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
    Processes all DetectedObject records for a video and produces
    meaningful lifecycle events (entry, exit, dwell).

    Runs AFTER the full frame loop completes — it needs the complete
    picture of when each track_id appeared/disappeared.

    Design decisions:
    - entry:  generated for every track_id (always fires)
    - exit:   generated when a track disappears for > exit_gap_seconds
    - dwell:  generated additionally when duration > dwell_threshold_seconds
              (a long entry is both an entry AND a dwell event)
    """

    def __init__(
        self,
        dwell_threshold_seconds: float = 10.0,
        exit_gap_seconds: float = 3.0,
    ):
        self.dwell_threshold = dwell_threshold_seconds
        self.exit_gap = exit_gap_seconds
        self.logger = get_logger()

    def generate(
        self,
        track_states: dict[int, TrackState],
        video_duration: float,
    ) -> list[GeneratedEvent]:
        """
        Convert accumulated TrackState data into GeneratedEvent objects.

        Args:
            track_states:    dict of track_id → TrackState, built during frame loop
            video_duration:  total duration of the processed video in seconds

        Returns:
            List of GeneratedEvent objects ready to be written to TrackEvent table
        """
        events = []

        for track_id, state in track_states.items():
            duration = state.last_seen - state.first_seen

            # Always generate an entry event
            entry = GeneratedEvent(
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
            )
            events.append(entry)

            # Dwell event if object was present for a meaningful time
            if duration >= self.dwell_threshold:
                dwell = GeneratedEvent(
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
                )
                events.append(dwell)

            # Exit event if the object disappeared before end of video
            disappeared_before_end = (video_duration - state.last_seen) > self.exit_gap
            if disappeared_before_end:
                exit_ev = GeneratedEvent(
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
                )
                events.append(exit_ev)

        self.logger.info(
            "events_generated",
            total=len(events),
            tracks=len(track_states),
            entry=sum(1 for e in events if e.event_type == "entry"),
            exit=sum(1 for e in events if e.event_type == "exit"),
            dwell=sum(1 for e in events if e.event_type == "dwell"),
        )
        return events

    def _build_entry_rag_text(self, state: TrackState, duration: float) -> str:
        """
        Build the RAG embedding text for an entry event.
        This text is what gets embedded into pgvector and searched against.
        It is human-readable and query-friendly — written to match how
        a user would ask about this event.
        """
        cls = state.object_class
        appeared = f"{state.first_seen:.1f}s"
        last = f"{state.last_seen:.1f}s"
        dur = f"{duration:.1f}s"

        return (
            f"{cls.capitalize()} (track #{state.track_id}) appeared at {appeared} "
            f"and was visible until {last} (duration: {dur}). "
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
    """
    Build per-detection RAG text (for DetectedObject row).
    Simpler than track event text — describes one moment in time.
    """
    track_info = f"track #{track_id}" if track_id is not None else "untracked"
    return (
        f"{object_class.capitalize()} ({track_info}) detected at {frame_second:.1f}s "
        f"in {quadrant} of frame. Confidence: {confidence:.0%}."
    )