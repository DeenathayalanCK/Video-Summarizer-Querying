"""
temporal_analyzer.py — Phase 7A: Temporal behaviour analysis.

Runs after EventGenerator has produced TrackEvents for a video.
Analyses the track_states (accumulated during the frame loop) to produce
rich behavioural inferences for persons and vehicles:

PERSON behaviours:
  - stationary     : present continuously, position changes < threshold
  - loitering      : dwell present in a single area, no clear destination
  - patrolling     : repeated appearances across different frame quadrants
  - passing_through: short visit (< 30s), single appearance
  - frequent_entry : enters/exits the scene multiple times

VEHICLE behaviours:
  - parked         : present for > 2 min with minimal movement
  - passing        : short visit < 30s, entered and exited quickly
  - waiting        : stopped briefly (30s–2min), then left
  - repeated_visit : appears more than once with a gap

The output is stored as a new TemporalAnalysis JSON blob per track
in the TrackEvent.attributes["temporal"] field, and used by:
  - The video summarizer (richer narrative)
  - The Q&A engine (answers "what was the person doing?")
  - The new Timeline/Temporal tab in the UI
"""

from dataclasses import dataclass, asdict, field
from typing import Optional

from app.core.logging import get_logger
from app.detection.event_generator import TrackState


# ── Behaviour classification thresholds ───────────────────────────────────────

PASSING_THROUGH_MAX_S   = 30.0   # present < 30s = passing through
WAITING_MAX_S           = 120.0  # 30–120s = waiting / brief stop
PARKED_MIN_S            = 120.0  # > 2 min vehicle = parked
LOITERING_MIN_S         = 60.0   # person in one area > 1 min = loitering
PATROL_MIN_APPEARANCES  = 3      # appears 3+ separate times = patrolling
FREQUENT_ENTRY_MIN      = 2      # enters more than N times = frequent entry


@dataclass
class TemporalBehaviour:
    """Behavioural classification for one tracked object."""
    track_id: int
    object_class: str
    behaviour: str              # primary label (see module docstring)
    duration_seconds: float
    first_seen: float
    last_seen: float
    appearance_count: int       # how many separate appearances / gaps
    avg_gap_seconds: float      # mean gap between appearances (0 if one block)
    quadrants_visited: list     # frame quadrants seen in
    movement_pattern: str       # "stationary" | "moving" | "mixed"
    confidence: float           # detection confidence
    notes: str                  # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalAnalyzer:
    """
    Analyses accumulated TrackState data to classify object behaviour.

    Called by VideoIntelligenceProcessor after EventGenerator.generate()
    and before VideoSummarizer. Writes results into TrackEvent.attributes.
    """

    def __init__(self):
        self.logger = get_logger()

    def analyze(
        self,
        track_states: dict,          # {track_id: TrackState}
        detected_objects_by_track: dict,  # {track_id: [DetectedObject]}
    ) -> list[TemporalBehaviour]:
        """
        Classify behaviour for every track.

        detected_objects_by_track is used to determine quadrant movement.
        It can be empty — the analyzer degrades gracefully to time-only analysis.
        """
        results = []
        for track_id, state in track_states.items():
            behaviour = self._classify(state, detected_objects_by_track.get(track_id, []))
            results.append(behaviour)
            self.logger.debug(
                "temporal_behaviour_classified",
                track_id=track_id,
                cls=state.object_class,
                behaviour=behaviour.behaviour,
                duration=behaviour.duration_seconds,
            )

        self.logger.info(
            "temporal_analysis_complete",
            tracks=len(results),
            behaviours={b.behaviour: sum(1 for r in results if r.behaviour == b.behaviour)
                        for b in results},
        )
        return results

    def _classify(self, state: TrackState, detections: list) -> TemporalBehaviour:
        duration = state.last_seen - state.first_seen

        # Count separate "appearances" — gaps in all_seconds > 10s = separate block
        appearances, gaps = self._count_appearances(state.all_seconds)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

        # Quadrant analysis
        quadrants = list({d.frame_quadrant for d in detections
                          if d.frame_quadrant}) if detections else []

        # Movement pattern: how many unique quadrants were visited
        if len(quadrants) <= 1:
            movement = "stationary"
        elif len(quadrants) == 2:
            movement = "mixed"
        else:
            movement = "moving"

        # ── Classify by object class ──────────────────────────────────────────
        if state.object_class in ("car", "truck", "bus", "motorcycle", "bicycle"):
            behaviour, notes = self._classify_vehicle(
                duration, appearances, avg_gap, movement, quadrants)
        else:
            behaviour, notes = self._classify_person(
                duration, appearances, avg_gap, movement, quadrants, state)

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
        )

    def _classify_vehicle(self, duration, appearances, avg_gap, movement, quadrants):
        if appearances > FREQUENT_ENTRY_MIN:
            return (
                "repeated_visit",
                f"Vehicle appeared {appearances} separate times "
                f"(avg gap {avg_gap:.0f}s between visits).",
            )
        if duration < PASSING_THROUGH_MAX_S:
            return (
                "passing",
                f"Vehicle passed through quickly ({duration:.0f}s). "
                f"{'Moved across frame.' if movement == 'moving' else 'Briefly visible.'}",
            )
        if duration < PARKED_MIN_S:
            return (
                "waiting",
                f"Vehicle stopped briefly for {duration:.0f}s "
                f"({'then left' if appearances == 1 else 'with brief gaps'}).",
            )
        return (
            "parked",
            f"Vehicle stationary for {duration:.0f}s "
            f"({duration / 60:.1f} min). Likely parked.",
        )

    def _classify_person(self, duration, appearances, avg_gap, movement, quadrants, state):
        # Frequent re-entry
        if appearances >= PATROL_MIN_APPEARANCES and avg_gap < 60:
            return (
                "patrolling",
                f"Person appeared {appearances} times across "
                f"{len(set(quadrants))} frame area(s) "
                f"(avg gap {avg_gap:.0f}s). Consistent patrol pattern.",
            )

        if appearances > FREQUENT_ENTRY_MIN:
            return (
                "frequent_entry",
                f"Person entered and left the scene {appearances} times "
                f"(avg gap {avg_gap:.0f}s). Frequent movement in/out.",
            )

        # Single continuous presence
        if duration < PASSING_THROUGH_MAX_S:
            return (
                "passing_through",
                f"Person briefly in frame for {duration:.0f}s. "
                f"{'Walked across scene.' if movement == 'moving' else 'Passed by.'}",
            )

        if duration < LOITERING_MIN_S:
            if movement == "moving":
                return (
                    "active",
                    f"Person moving around for {duration:.0f}s across "
                    f"{len(set(quadrants))} area(s).",
                )
            return (
                "stopped",
                f"Person stopped for {duration:.0f}s "
                f"({'in one spot' if movement == 'stationary' else 'with some movement'}).",
            )

        # Long presence
        if movement == "stationary":
            return (
                "stationary",
                f"Person remained in one position for {duration:.0f}s "
                f"({duration / 60:.1f} min). Likely sitting, working, or resting.",
            )

        if len(set(quadrants)) >= 3:
            return (
                "patrolling",
                f"Person moved through {len(set(quadrants))} different frame areas "
                f"over {duration:.0f}s. Consistent with patrol or active movement.",
            )

        return (
            "loitering",
            f"Person present for {duration:.0f}s ({duration / 60:.1f} min) "
            f"in a limited area ({len(set(quadrants))} zone(s)). "
            f"No clear destination — possible loitering.",
        )

    @staticmethod
    def _count_appearances(all_seconds: list, gap_threshold: float = 10.0):
        """
        Count how many separate 'blocks' of presence exist in the timestamp list.
        Returns (appearance_count, list_of_gaps).

        Example: [1, 2, 3, 45, 46, 47] with gap_threshold=10
          → 2 appearances, [42.0] gap
        """
        if not all_seconds:
            return 1, []

        sorted_s = sorted(all_seconds)
        appearances = 1
        gaps = []
        for i in range(1, len(sorted_s)):
            gap = sorted_s[i] - sorted_s[i - 1]
            if gap > gap_threshold:
                appearances += 1
                gaps.append(gap)
        return appearances, gaps