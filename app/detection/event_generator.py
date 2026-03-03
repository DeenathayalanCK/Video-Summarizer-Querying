"""
event_generator.py — Phase 6A lifecycle event producer.

Two key improvements over the original:

1. GHOST TRACK FILTER
   ByteTrack at 1 FPS can create very short tracks (1-2 detections) from
   partial occlusion, reflections, or objects briefly crossing frame edge.
   We drop tracks seen in fewer than MIN_VISIBLE_FRAMES frames.

2. TRACK CONTINUITY MERGING
   A single physical person in a 10-min room video may lose their track ID
   multiple times (moves fast, sits still, occlusion, brief exit).
   ByteTrack assigns a NEW track_id each time.
   We merge tracks of the SAME class whose time gaps < merge_gap_seconds
   into a single canonical track (first track_id, best crop kept).
"""

from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger


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


@dataclass
class GeneratedEvent:
    """
    A lifecycle event produced by the EventGenerator.
    Maps directly to a TrackEvent row in the DB.
    """
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


class EventGenerator:
    """
    Processes all TrackState records for a video and produces
    meaningful lifecycle events (entry, exit, dwell).

    Runs AFTER the full frame loop completes.

    Pipeline:
      1. Filter ghost tracks (frame_count < min_visible_frames)
      2. Merge fragmented tracks of same class within merge_gap_seconds
      3. Generate entry / dwell / exit events for surviving tracks
    """

    def __init__(
        self,
        dwell_threshold_seconds: float = 10.0,
        exit_gap_seconds: float = 3.0,
        min_visible_frames: int = 2,
        merge_gap_seconds: float = 30.0,
    ):
        self.dwell_threshold = dwell_threshold_seconds
        self.exit_gap = exit_gap_seconds
        self.min_visible_frames = min_visible_frames
        self.merge_gap = merge_gap_seconds
        self.logger = get_logger()

    def generate(
        self,
        track_states: dict[int, TrackState],
        video_duration: float,
    ) -> list[GeneratedEvent]:
        """
        Convert accumulated TrackState data into GeneratedEvent objects.
        """
        # Step 1: Filter ghost tracks
        valid_states = {
            tid: state for tid, state in track_states.items()
            if state.frame_count >= self.min_visible_frames
        }
        ghost_count = len(track_states) - len(valid_states)
        if ghost_count:
            self.logger.info(
                "ghost_tracks_filtered",
                removed=ghost_count,
                min_frames=self.min_visible_frames,
            )

        # Step 2: Merge fragmented tracks
        merged_states = self._merge_fragmented_tracks(valid_states)
        merge_count = len(valid_states) - len(merged_states)
        if merge_count:
            self.logger.info(
                "fragmented_tracks_merged",
                before=len(valid_states),
                after=len(merged_states),
                merged=merge_count,
            )

        # Step 3: Generate events
        events = []
        for track_id, state in merged_states.items():
            duration = state.last_seen - state.first_seen

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
                rag_text=self._build_entry_rag_text(state, duration),
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
                    rag_text=self._build_dwell_rag_text(state, duration),
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
                    rag_text=self._build_exit_rag_text(state, duration),
                ))

        self.logger.info(
            "events_generated",
            total=len(events),
            tracks=len(merged_states),
            entry=sum(1 for e in events if e.event_type == "entry"),
            exit=sum(1 for e in events if e.event_type == "exit"),
            dwell=sum(1 for e in events if e.event_type == "dwell"),
        )
        return events

    def _merge_fragmented_tracks(
        self,
        states: dict[int, TrackState],
    ) -> dict[int, TrackState]:
        """
        Merge tracks of same class that are close in time.

        ByteTrack re-assigns a new track_id when it "loses" someone —
        e.g. a security guard who sits still, sleeps, walks behind a desk.
        If the gap between track A ending and track B starting is within
        merge_gap_seconds, they are combined into a single canonical track.

        The canonical track keeps the FIRST track_id and the BEST crop.
        """
        if not states:
            return states

        by_class: dict[str, list[TrackState]] = {}
        for state in states.values():
            by_class.setdefault(state.object_class, []).append(state)

        merged: dict[int, TrackState] = {}

        for cls, group in by_class.items():
            group.sort(key=lambda s: s.first_seen)
            canonical = group[0]

            for current in group[1:]:
                gap = current.first_seen - canonical.last_seen

                if gap <= self.merge_gap:
                    # Absorb current into canonical
                    canonical.last_seen = max(canonical.last_seen, current.last_seen)
                    canonical.frame_count += current.frame_count
                    canonical.all_seconds.extend(current.all_seconds)
                    if current.best_confidence > canonical.best_confidence:
                        canonical.best_confidence = current.best_confidence
                        canonical.best_second = current.best_second
                        if current.best_crop_path:
                            canonical.best_crop_path = current.best_crop_path
                    self.logger.debug(
                        "tracks_merged",
                        canonical_id=canonical.track_id,
                        absorbed_id=current.track_id,
                        gap_seconds=round(gap, 1),
                        cls=cls,
                    )
                else:
                    merged[canonical.track_id] = canonical
                    canonical = current

            merged[canonical.track_id] = canonical

        return merged

    def _build_entry_rag_text(self, state: TrackState, duration: float) -> str:
        cls = state.object_class
        return (
            f"{cls.capitalize()} (track #{state.track_id}) appeared at {state.first_seen:.1f}s "
            f"and was visible until {state.last_seen:.1f}s (duration: {duration:.1f}s). "
            f"Detected with {state.best_confidence:.0%} confidence."
        )

    def _build_dwell_rag_text(self, state: TrackState, duration: float) -> str:
        cls = state.object_class
        return (
            f"{cls.capitalize()} (track #{state.track_id}) remained stationary or "
            f"loitered from {state.first_seen:.1f}s to {state.last_seen:.1f}s "
            f"({duration:.1f} seconds total). This is a prolonged presence event."
        )

    def _build_exit_rag_text(self, state: TrackState, duration: float) -> str:
        cls = state.object_class
        return (
            f"{cls.capitalize()} (track #{state.track_id}) left the scene at "
            f"{state.last_seen:.1f}s after being present for {duration:.1f} seconds "
            f"(first seen at {state.first_seen:.1f}s)."
        )


def build_detection_rag_text(
    object_class: str,
    track_id: Optional[int],
    frame_second: float,
    confidence: float,
    quadrant: str,
) -> str:
    """Build per-detection RAG text (for DetectedObject row)."""
    track_info = f"track #{track_id}" if track_id is not None else "untracked"
    return (
        f"{object_class.capitalize()} ({track_info}) detected at {frame_second:.1f}s "
        f"in {quadrant} of frame. Confidence: {confidence:.0%}."
    )