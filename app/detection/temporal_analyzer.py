"""
temporal_analyzer.py — Phase 7A: Temporal behaviour analysis.

Upgraded in this version to consume motion_summary from EventGenerator

  fall_detected   ← fall_proxy event from bbox AR change
  running         ← dominant_state == "running"
  sudden_stop     ← sudden_stop event in motion_events
  ... plus all original duration/quadrant-based classes

Priority order (highest wins):
  1. fall_detected  (safety-critical — always surfaced first)
  2. running
  3. sudden_stop
  4. patrolling     (appearances-based)
  5. frequent_entry
  6. Duration-based: passing_through / active / stopped / loitering / stationary

Stored in TrackEvent.attributes["temporal"] as a flat JSON dict.
"""

from dataclasses import dataclass, asdict
from app.core.logging import get_logger
from app.detection.event_generator import TrackState, EventGenerator


# ── Duration thresholds ───────────────────────────────────────────────────────
PASSING_MAX   = 30.0    # < 30s  → passing_through
LOITERING_MIN = 60.0    # > 60s in limited area → loitering
PARKED_MIN    = 120.0   # > 2min vehicle → parked
PATROL_MIN_N  = 3       # 3+ separate appearances → patrolling
FREQUENT_MIN  = 2       # > 2 entries → frequent_entry


@dataclass
class TemporalBehaviour:
    track_id: int
    object_class: str
    behaviour: str
    duration_seconds: float
    first_seen: float
    last_seen: float
    appearance_count: int
    avg_gap_seconds: float
    quadrants_visited: list
    movement_pattern: str
    confidence: float
    notes: str
    # New: motion detail from Bug 2
    dominant_motion: str = "unknown"
    motion_events: list = None

    def __post_init__(self):
        if self.motion_events is None:
            self.motion_events = []

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalAnalyzer:
    """
    Classifies behaviour for every track after EventGenerator runs.
    Now reads motion_summary from TrackState.motion_samples directly.
    """

    def __init__(self):
        self.logger = get_logger()
        self._eg = EventGenerator()   # used to call _analyse_motion()

    def analyze(
        self,
        track_states: dict,
        detected_objects_by_track: dict,
    ) -> list:
        results = []
        for track_id, state in track_states.items():
            beh = self._classify(state, detected_objects_by_track.get(track_id, []))
            results.append(beh)
            self.logger.debug(
                "behaviour_classified",
                track_id=track_id,
                cls=state.object_class,
                behaviour=beh.behaviour,
                motion=beh.dominant_motion,
            )

        summary = {}
        for b in results:
            summary[b.behaviour] = summary.get(b.behaviour, 0) + 1
        self.logger.info("temporal_analysis_complete", tracks=len(results), summary=summary)
        return results

    def _classify(self, state: TrackState, detections: list) -> TemporalBehaviour:
        duration = state.last_seen - state.first_seen
        appearances, gaps = _count_appearances(state.all_seconds)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

        # Quadrant data from DetectedObject rows
        quadrants = list({d.frame_quadrant for d in detections if d.frame_quadrant})

        # ── Motion signals from frame-sequence analysis (Bug 2) ───────────────
        motion = self._eg._analyse_motion(state)
        dom    = motion.get("dominant_state", "unknown")
        mevts  = motion.get("motion_events", [])

        # Movement pattern: prefer motion dom over quadrant count
        if dom in ("walking", "running"):
            movement = "moving"
        elif dom == "standing":
            movement = "stationary"
        elif len(quadrants) <= 1:
            movement = "stationary"
        elif len(quadrants) <= 2:
            movement = "mixed"
        else:
            movement = "moving"

        # ── Classify ──────────────────────────────────────────────────────────
        if state.object_class in ("car", "truck", "bus", "motorcycle", "bicycle"):
            behaviour, notes = self._vehicle(duration, appearances, avg_gap, movement)
        else:
            behaviour, notes = self._person(
                duration, appearances, avg_gap, movement, quadrants, dom, mevts)

        return TemporalBehaviour(
            track_id=state.track_id,
            object_class=state.object_class,
            behaviour=behaviour,
            duration_seconds=round(duration, 1),
            first_seen=round(state.first_seen, 1),
            last_seen=round(state.last_seen, 1),
            appearance_count=appearances,
            avg_gap_seconds=round(avg_gap, 1),
            quadrants_visited=sorted(set(quadrants)),
            movement_pattern=movement,
            confidence=round(state.best_confidence, 3),
            notes=notes,
            dominant_motion=dom,
            motion_events=mevts,
        )

    # ── Vehicle ───────────────────────────────────────────────────────────────

    def _vehicle(self, duration, appearances, avg_gap, movement):
        if appearances > FREQUENT_MIN:
            return ("repeated_visit",
                    f"Vehicle appeared {appearances} times (avg gap {avg_gap:.0f}s).")
        if duration < PASSING_MAX:
            return ("passing",
                    f"Vehicle passed through in {duration:.0f}s.")
        if duration < PARKED_MIN:
            return ("waiting",
                    f"Vehicle stopped briefly for {duration:.0f}s.")
        return ("parked",
                f"Vehicle stationary {duration:.0f}s ({duration/60:.1f} min). Likely parked.")

    # ── Person — priority ladder ──────────────────────────────────────────────

    def _person(self, duration, appearances, avg_gap, movement, quadrants, dom, mevts):
        # 1. Safety-critical: fall
        fall_evts = [e for e in mevts if "fall_proxy" in e]
        if fall_evts:
            return (
                "fall_detected",
                f"Bbox aspect-ratio changed from upright to horizontal — "
                f"possible fall. Event: {fall_evts[0]}. Review footage immediately.",
            )

        # 2. Running
        if dom == "running":
            return (
                "running",
                f"Person was running for most of their {duration:.0f}s presence "
                f"(high inter-frame displacement).",
            )

        # 3. Sudden stop
        stop_evts = [e for e in mevts if "sudden_stop" in e]
        if stop_evts:
            return (
                "sudden_stop",
                f"Person stopped abruptly after moving. {stop_evts[0]}.",
            )

        # 4. Patrol (appearances-based)
        if appearances >= PATROL_MIN_N and avg_gap < 60:
            return (
                "patrolling",
                f"Person appeared {appearances}× across "
                f"{len(set(quadrants))} area(s) (avg gap {avg_gap:.0f}s). Patrol pattern.",
            )

        # 5. Frequent entry/exit
        if appearances > FREQUENT_MIN:
            return (
                "frequent_entry",
                f"Person entered/exited {appearances} times (avg gap {avg_gap:.0f}s).",
            )

        # 6. Duration-based
        if duration < PASSING_MAX:
            desc = "Walked across scene." if movement == "moving" else "Passed by."
            return ("passing_through", f"Brief visit {duration:.0f}s. {desc}")

        if duration < LOITERING_MIN:
            if movement == "moving":
                return ("active",
                        f"Person moving around for {duration:.0f}s across "
                        f"{len(set(quadrants))} area(s).")
            return ("stopped",
                    f"Person stopped for {duration:.0f}s "
                    f"({'one spot' if movement == 'stationary' else 'limited area'}).")

        if movement == "stationary":
            return ("stationary",
                    f"Person in one position for {duration:.0f}s "
                    f"({duration/60:.1f} min). Sitting/working/resting.")

        if len(set(quadrants)) >= 3:
            return ("patrolling",
                    f"Person covered {len(set(quadrants))} frame areas in {duration:.0f}s.")

        return (
            "loitering",
            f"Person present {duration:.0f}s ({duration/60:.1f} min) "
            f"in limited area ({len(set(quadrants))} zone(s)). No clear destination.",
        )


# ── Helper ────────────────────────────────────────────────────────────────────

def _count_appearances(all_seconds: list, gap: float = 10.0):
    if not all_seconds:
        return 1, []
    s = sorted(all_seconds)
    n, gaps = 1, []
    for i in range(1, len(s)):
        d = s[i] - s[i - 1]
        if d > gap:
            n += 1
            gaps.append(d)
    return n, gaps