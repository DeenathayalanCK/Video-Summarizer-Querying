"""
event_generator.py — Phase 6A lifecycle event producer + motion analysis.

Three improvements in this version:

1. MOTION SAMPLE COLLECTION (Bug 2)
   TrackState now stores a MotionSample per frame: (second, cx, cy, w, h)
   all in normalised 0-1 space.  After the frame loop, _analyse_motion()
   derives per-frame states:
       standing  → displacement < 4% frame-width per second
       walking   → 4–15%
       running   → > 15%
   And detects special events purely from bbox signals:
       fall_proxy        — bbox flips from tall-narrow to wide-flat (H/W ratio drops)
       sudden_stop       — motion then two+ still frames in a row
       direction_change  — x-vector reverses
   Dominant state + event list stored in GeneratedEvent.motion_summary.

2. REID-AWARE TRACK MERGING (Bug 3)
   generate() now accepts reid_embeddings={track_id: np.ndarray}.
   When provided, two tracks within merge_gap are only merged when their
   cosine similarity ≥ reid_threshold (default 0.75).
   Falls back to original class-only time-gap merge when embeddings absent.

3. GHOST TRACK FILTER + CONTINUITY MERGE (original, preserved)
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger


# ── Motion thresholds (normalised bbox-width units per second) ────────────────
_WALK_MIN  = 0.04   # > 4%  → walking
_RUN_MIN   = 0.15   # > 15% → running
_FALL_AR   = 0.50   # aspect-ratio (H/W) drop > 0.5 in one step → fall proxy
_STOP_N    = 2      # consecutive still frames after motion → sudden_stop


@dataclass
class MotionSample:
    """One frame's bbox data for a tracked object — collected during frame loop."""
    second: float
    cx: float     # centre-x, normalised 0-1
    cy: float     # centre-y, normalised 0-1
    w: float      # bbox width, normalised
    h: float      # bbox height, normalised

    @property
    def ar(self) -> float:
        """Aspect ratio H/W.  Upright person ≈ 2.5.  Fallen person ≈ 0.5."""
        return self.h / self.w if self.w > 0 else 1.0


@dataclass
class TrackState:
    """Internal state for one tracked object across a video."""
    track_id: int
    object_class: str
    first_seen: float
    last_seen: float
    frame_count: int = 1
    best_second: float = 0.0
    best_confidence: float = 0.0
    best_crop_path: Optional[str] = None
    all_seconds: list = field(default_factory=list)
    motion_samples: list = field(default_factory=list)   # list[MotionSample]


@dataclass
class GeneratedEvent:
    """A lifecycle event produced by EventGenerator → TrackEvent DB row."""
    track_id: int
    object_class: str
    event_type: str          # "entry" | "exit" | "dwell"
    first_seen_second: float
    last_seen_second: float
    duration_seconds: float
    best_frame_second: float
    best_crop_path: Optional[str]
    best_confidence: float
    rag_text: str
    motion_summary: dict = field(default_factory=dict)
    # {dominant_state, motion_events, direction, frame_states}


class EventGenerator:
    """
    Processes TrackState records after the full frame loop and produces
    lifecycle events (entry, exit, dwell) enriched with motion semantics.

    Pipeline:
      1. Filter ghost tracks
      2. Merge fragmented tracks (ReID-aware when embeddings provided)
      3. Analyse per-track motion sequence → motion_summary
      4. Generate entry / dwell / exit GeneratedEvents
    """

    def __init__(
        self,
        dwell_threshold_seconds: float = 10.0,
        exit_gap_seconds: float = 3.0,
        min_visible_frames: int = 2,
        merge_gap_seconds: float = 30.0,
        reid_threshold: float = 0.75,
    ):
        self.dwell_threshold = dwell_threshold_seconds
        self.exit_gap = exit_gap_seconds
        self.min_visible_frames = min_visible_frames
        self.merge_gap = merge_gap_seconds
        self.reid_threshold = reid_threshold
        self.logger = get_logger()

    def generate(
        self,
        track_states: dict,
        video_duration: float,
        reid_embeddings: dict = None,   # {track_id: np.ndarray} — optional
    ) -> list:
        # ── 1. Ghost filter ───────────────────────────────────────────────────
        valid = {
            tid: s for tid, s in track_states.items()
            if s.frame_count >= self.min_visible_frames
        }
        removed = len(track_states) - len(valid)
        if removed:
            self.logger.info("ghost_tracks_filtered", removed=removed)

        # ── 2. Merge fragmented tracks ────────────────────────────────────────
        if reid_embeddings:
            merged = self._merge_with_reid(valid, reid_embeddings)
            method = "reid"
        else:
            merged = self._merge_fragmented_tracks(valid)
            method = "time_gap"

        delta = len(valid) - len(merged)
        if delta:
            self.logger.info("tracks_merged", count=delta, method=method)

        # ── 3. Motion analysis + event generation ────────────────────────────
        events = []
        for track_id, state in merged.items():
            duration = state.last_seen - state.first_seen
            motion = self._analyse_motion(state)

            events.append(GeneratedEvent(
                track_id=track_id,
                object_class=state.object_class,
                event_type="entry",
                first_seen_second=state.first_seen,
                last_seen_second=state.last_seen,
                duration_seconds=duration,
                best_frame_second=state.best_second,
                best_crop_path=state.best_crop_path,
                best_confidence=state.best_confidence,
                rag_text=self._rag_entry(state, duration, motion),
                motion_summary=motion,
            ))

            if duration >= self.dwell_threshold:
                events.append(GeneratedEvent(
                    track_id=track_id,
                    object_class=state.object_class,
                    event_type="dwell",
                    first_seen_second=state.first_seen,
                    last_seen_second=state.last_seen,
                    duration_seconds=duration,
                    best_frame_second=state.best_second,
                    best_crop_path=state.best_crop_path,
                    best_confidence=state.best_confidence,
                    rag_text=self._rag_dwell(state, duration, motion),
                    motion_summary=motion,
                ))

            if (video_duration - state.last_seen) > self.exit_gap:
                events.append(GeneratedEvent(
                    track_id=track_id,
                    object_class=state.object_class,
                    event_type="exit",
                    first_seen_second=state.first_seen,
                    last_seen_second=state.last_seen,
                    duration_seconds=duration,
                    best_frame_second=state.best_second,
                    best_crop_path=state.best_crop_path,
                    best_confidence=state.best_confidence,
                    rag_text=self._rag_exit(state, duration, motion),
                    motion_summary=motion,
                ))

        self.logger.info(
            "events_generated",
            total=len(events),
            tracks=len(merged),
            entry=sum(1 for e in events if e.event_type == "entry"),
            exit=sum(1 for e in events if e.event_type == "exit"),
            dwell=sum(1 for e in events if e.event_type == "dwell"),
        )
        return events

    # ── Bug 2: Frame-sequence motion analysis ─────────────────────────────────

    def _analyse_motion(self, state: TrackState) -> dict:
        """
        Derive semantic motion labels from MotionSample sequence.

        At 1 FPS, displacement between consecutive samples is displacement/sec.
        Uses only normalised bbox geometry — no pose estimation, no GPU.

        Returns:
            dominant_state : str  ("standing" | "walking" | "running" | "stationary")
            motion_events  : list[str]  e.g. ["fall_proxy @ 32.0s", "sudden_stop @ 14.0s"]
            direction      : str  ("stationary" | "left_to_right" | "right_to_left" | "mixed")
            frame_states   : list[{second, state, disp}]
        """
        samples = state.motion_samples
        if len(samples) < 2:
            return {
                "dominant_state": "stationary",
                "motion_events": [],
                "direction": "stationary",
                "frame_states": [],
            }

        frame_states = []
        motion_events = []
        prev_state = None
        still_run = 0

        for i in range(1, len(samples)):
            prev = samples[i - 1]
            curr = samples[i]
            dt = max(curr.second - prev.second, 0.5)

            dx = curr.cx - prev.cx
            dy = curr.cy - prev.cy
            disp = math.sqrt(dx * dx + dy * dy) / dt

            if disp < _WALK_MIN:
                s = "standing"
            elif disp < _RUN_MIN:
                s = "walking"
            else:
                s = "running"

            frame_states.append({"second": round(curr.second, 1), "state": s, "disp": round(disp, 3)})

            # Fall proxy: tall upright bbox → wide flat bbox in one step
            if state.object_class == "person" and prev.ar > 1.3 and curr.ar < (prev.ar - _FALL_AR):
                motion_events.append(
                    f"fall_proxy @ {curr.second:.1f}s "
                    f"(AR {prev.ar:.1f}→{curr.ar:.1f})"
                )

            # Sudden stop: was moving, now still for _STOP_N+ frames
            if s == "standing":
                still_run += 1
            else:
                still_run = 0

            if still_run == _STOP_N and prev_state in ("walking", "running"):
                motion_events.append(f"sudden_stop @ {curr.second:.1f}s")

            # Direction reversal: x-vector flips
            if i >= 2:
                prev_dx = samples[i - 1].cx - samples[i - 2].cx
                if prev_dx * dx < 0 and abs(dx) > _WALK_MIN * dt:
                    motion_events.append(f"direction_change @ {curr.second:.1f}s")

            prev_state = s

        dominant = Counter(f["state"] for f in frame_states).most_common(1)[0][0] \
            if frame_states else "stationary"

        dx_total = samples[-1].cx - samples[0].cx
        if abs(dx_total) < 0.05:
            direction = "stationary"
        elif dx_total > 0:
            direction = "left_to_right"
        else:
            direction = "right_to_left"

        return {
            "dominant_state": dominant,
            "motion_events": motion_events,
            "direction": direction,
            "frame_states": frame_states,
        }

    # ── Bug 3: ReID-aware merge ───────────────────────────────────────────────

    def _merge_with_reid(self, states: dict, reid_embeddings: dict) -> dict:
        """
        Same time-window logic as _merge_fragmented_tracks but adds a
        cosine-similarity gate using crop appearance embeddings.

        Two tracks within merge_gap are merged ONLY when:
          - both have embeddings AND similarity ≥ reid_threshold, OR
          - one or both lack embeddings (falls back to time-gap)
        """
        import numpy as np

        def cosine(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

        by_class = {}
        for s in states.values():
            by_class.setdefault(s.object_class, []).append(s)

        merged = {}
        for cls, group in by_class.items():
            group.sort(key=lambda s: s.first_seen)
            canonical = group[0]

            for current in group[1:]:
                gap = current.first_seen - canonical.last_seen
                if gap > self.merge_gap:
                    merged[canonical.track_id] = canonical
                    canonical = current
                    continue

                ea = reid_embeddings.get(canonical.track_id)
                eb = reid_embeddings.get(current.track_id)

                if ea is not None and eb is not None:
                    sim = cosine(ea, eb)
                    ok = sim >= self.reid_threshold
                    self.logger.debug(
                        "reid_check",
                        a=canonical.track_id, b=current.track_id,
                        gap=round(gap, 1), sim=round(sim, 3), merge=ok,
                    )
                else:
                    ok = True   # no embedding → trust time-gap

                if ok:
                    canonical = self._absorb(canonical, current)
                else:
                    merged[canonical.track_id] = canonical
                    canonical = current

            merged[canonical.track_id] = canonical

        return merged

    def _merge_fragmented_tracks(self, states: dict) -> dict:
        """Original class-only time-gap merge (used when no ReID embeddings)."""
        by_class = {}
        for s in states.values():
            by_class.setdefault(s.object_class, []).append(s)

        merged = {}
        for cls, group in by_class.items():
            group.sort(key=lambda s: s.first_seen)
            canonical = group[0]
            for current in group[1:]:
                if current.first_seen - canonical.last_seen <= self.merge_gap:
                    canonical = self._absorb(canonical, current)
                else:
                    merged[canonical.track_id] = canonical
                    canonical = current
            merged[canonical.track_id] = canonical

        return merged

    @staticmethod
    def _absorb(canonical: TrackState, current: TrackState) -> TrackState:
        """Merge current into canonical in-place, return canonical."""
        canonical.last_seen = max(canonical.last_seen, current.last_seen)
        canonical.frame_count += current.frame_count
        canonical.all_seconds.extend(current.all_seconds)
        canonical.motion_samples.extend(current.motion_samples)
        if current.best_confidence > canonical.best_confidence:
            canonical.best_confidence = current.best_confidence
            canonical.best_second = current.best_second
            if current.best_crop_path:
                canonical.best_crop_path = current.best_crop_path
        return canonical

    # ── RAG text builders ─────────────────────────────────────────────────────

    def _motion_clause(self, motion: dict) -> str:
        dom = motion.get("dominant_state", "")
        events = motion.get("motion_events", [])
        parts = []
        if dom and dom != "stationary":
            parts.append(f"predominantly {dom}")
        if events:
            parts.append("; ".join(events[:2]))
        return (". " + ", ".join(parts) + ".") if parts else "."

    def _rag_entry(self, state, duration, motion) -> str:
        cls = state.object_class
        return (
            f"{cls.capitalize()} (track #{state.track_id}) appeared at "
            f"{state.first_seen:.1f}s and was visible until {state.last_seen:.1f}s "
            f"(duration: {duration:.1f}s, confidence: {state.best_confidence:.0%})"
            f"{self._motion_clause(motion)}"
        )

    def _rag_dwell(self, state, duration, motion) -> str:
        cls = state.object_class
        dom = motion.get("dominant_state", "present")
        events = motion.get("motion_events", [])
        ev_str = f" Notable: {'; '.join(events[:2])}." if events else ""
        return (
            f"{cls.capitalize()} (track #{state.track_id}) was {dom} from "
            f"{state.first_seen:.1f}s to {state.last_seen:.1f}s "
            f"({duration:.1f}s total). Prolonged presence.{ev_str}"
        )

    def _rag_exit(self, state, duration, motion) -> str:
        cls = state.object_class
        dom = motion.get("dominant_state", "")
        suffix = f" (was {dom})" if dom else ""
        return (
            f"{cls.capitalize()} (track #{state.track_id}) left at "
            f"{state.last_seen:.1f}s after {duration:.1f}s{suffix}."
        )


# ── Module-level helper (imported by processor) ───────────────────────────────

def build_detection_rag_text(
    object_class: str,
    track_id,
    frame_second: float,
    confidence: float,
    quadrant: str,
) -> str:
    track_info = f"track #{track_id}" if track_id is not None else "untracked"
    return (
        f"{object_class.capitalize()} ({track_info}) detected at {frame_second:.1f}s "
        f"in {quadrant} of frame. Confidence: {confidence:.0%}."
    )