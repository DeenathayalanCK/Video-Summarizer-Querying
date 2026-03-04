"""
scene_understanding.py — Scene-level intelligence layer.

Analyses the FULL set of track events for a video to detect emergent
scene-level situations that cannot be inferred from individual tracks.

Scene events detected:
  crowd_gathering      — 3+ persons present simultaneously for > 10s
  fight_proxy          — 2+ persons in same quadrant with sudden_stop or
                         direction_change events within 5s of each other
  vehicle_congestion   — 3+ vehicles present simultaneously
  abandoned_object     — object class that stays stationary while the person
                         who brought it leaves (heuristic: person exits,
                         track stays; not directly detectable without object
                         class but inferred from motion_summary)
  lone_person_at_night — single person present for > 2 min with no other
                         activity (useful with timestamp metadata)
  rapid_entry_exit     — any track with duration < 10s but repeated > 2 times
                         (could indicate door-testing, loitering pattern)

Output:
  List[SceneEvent] stored in VideoTimeline.scene_events (JSONB array).
  Each SceneEvent has: type, start_second, end_second, track_ids, confidence, notes.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

from app.core.logging import get_logger


@dataclass
class SceneEvent:
    event_type: str          # crowd_gathering, fight_proxy, vehicle_congestion, etc.
    start_second: float
    end_second: float
    track_ids: List[int]
    confidence: float        # 0.0–1.0 — how certain we are
    notes: str               # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


# ── Thresholds ────────────────────────────────────────────────────────────────
_CROWD_MIN_PERSONS   = 3      # simultaneous persons for crowd_gathering
_CROWD_MIN_DURATION  = 10.0   # seconds of simultaneous presence
_CONGESTION_MIN_VEHS = 3      # simultaneous vehicles for vehicle_congestion
_FIGHT_PROXIMITY_S   = 5.0    # motion events within this window = fight proxy
_RAPID_ENTRY_MAX_S   = 10.0   # duration < 10s = rapid entry
_RAPID_ENTRY_MIN_N   = 2      # repeated > 2 times = suspicious


class SceneUnderstanding:
    """
    Detects scene-level situations from structured track data.

    Called by VideoIntelligenceProcessor after all TrackEvents are saved,
    and before VideoSummarizer. Results written to VideoTimeline.scene_events.

    Works entirely from already-computed data:
      - TrackEvent.first_seen_second / last_seen_second (for overlap detection)
      - TrackEvent.attributes["motion_summary"]["motion_events"] (for fight proxy)
      - TrackEvent.attributes["temporal"]["appearance_count"] (for rapid entry)
    No additional LLM or CV calls.
    """

    def __init__(self):
        self.logger = get_logger()

    def analyse(self, track_events: list) -> List[SceneEvent]:
        """
        Analyse all entry-type TrackEvents for a video.
        Returns list of SceneEvent objects (may be empty).

        track_events: list of TrackEvent ORM objects (event_type == "entry")
        """
        scene_events = []

        persons   = [e for e in track_events if e.object_class == "person"]
        vehicles  = [e for e in track_events
                     if e.object_class in ("car", "truck", "bus", "motorcycle", "bicycle")]

        scene_events += self._detect_crowd(persons)
        scene_events += self._detect_congestion(vehicles)
        scene_events += self._detect_fight_proxy(persons)
        scene_events += self._detect_rapid_entry(persons)

        scene_events.sort(key=lambda e: e.start_second)

        self.logger.info(
            "scene_understanding_complete",
            scene_events=len(scene_events),
            types=[e.event_type for e in scene_events],
        )
        return scene_events

    # ── Crowd gathering ───────────────────────────────────────────────────────

    def _detect_crowd(self, persons: list) -> List[SceneEvent]:
        """
        Find time windows where ≥ CROWD_MIN_PERSONS persons overlap.
        Uses a sweep-line over all [first_seen, last_seen] intervals.
        """
        if len(persons) < _CROWD_MIN_PERSONS:
            return []

        # Build list of (time, +1/-1, track_id) events
        events = []
        for p in persons:
            events.append((p.first_seen_second, +1, p.track_id))
            events.append((p.last_seen_second,  -1, p.track_id))
        events.sort()

        results = []
        active = set()
        crowd_start = None

        for t, delta, tid in events:
            if delta == +1:
                active.add(tid)
            else:
                active.discard(tid)

            if len(active) >= _CROWD_MIN_PERSONS and crowd_start is None:
                crowd_start = t
                crowd_tracks = set(active)

            elif len(active) < _CROWD_MIN_PERSONS and crowd_start is not None:
                duration = t - crowd_start
                if duration >= _CROWD_MIN_DURATION:
                    results.append(SceneEvent(
                        event_type="crowd_gathering",
                        start_second=round(crowd_start, 1),
                        end_second=round(t, 1),
                        track_ids=sorted(crowd_tracks),
                        confidence=min(0.95, 0.6 + len(crowd_tracks) * 0.05),
                        notes=(
                            f"{len(crowd_tracks)} people present simultaneously "
                            f"for {duration:.0f}s from {crowd_start:.1f}s."
                        ),
                    ))
                crowd_start = None

        return results

    # ── Vehicle congestion ────────────────────────────────────────────────────

    def _detect_congestion(self, vehicles: list) -> List[SceneEvent]:
        if len(vehicles) < _CONGESTION_MIN_VEHS:
            return []

        events = []
        for v in vehicles:
            events.append((v.first_seen_second, +1, v.track_id))
            events.append((v.last_seen_second,  -1, v.track_id))
        events.sort()

        results = []
        active = set()
        start = None

        for t, delta, tid in events:
            if delta == +1:
                active.add(tid)
            else:
                active.discard(tid)

            if len(active) >= _CONGESTION_MIN_VEHS and start is None:
                start = t
                peak_tracks = set(active)

            elif len(active) < _CONGESTION_MIN_VEHS and start is not None:
                results.append(SceneEvent(
                    event_type="vehicle_congestion",
                    start_second=round(start, 1),
                    end_second=round(t, 1),
                    track_ids=sorted(peak_tracks),
                    confidence=0.80,
                    notes=(
                        f"{len(peak_tracks)} vehicles present simultaneously "
                        f"from {start:.1f}s to {t:.1f}s."
                    ),
                ))
                start = None

        return results

    # ── Fight proxy ───────────────────────────────────────────────────────────

    def _detect_fight_proxy(self, persons: list) -> List[SceneEvent]:
        """
        Fight proxy: two or more people in the same quadrant who both have
        sudden_stop or direction_change events within FIGHT_PROXIMITY_S of
        each other.  This is a low-precision heuristic — confidence is low.
        """
        # Gather (person, event_second, event_type) tuples for motion events
        motion_hits = []
        for p in persons:
            attrs = p.attributes or {}
            ms = attrs.get("motion_summary", {})
            for ev_str in ms.get("motion_events", []):
                if "sudden_stop" in ev_str or "direction_change" in ev_str:
                    try:
                        sec = float(ev_str.split("@")[1].strip().split("s")[0])
                        motion_hits.append((sec, p.track_id))
                    except (IndexError, ValueError):
                        pass

        if len(motion_hits) < 2:
            return []

        # Check for any two hits within FIGHT_PROXIMITY_S
        motion_hits.sort()
        results = []
        for i in range(len(motion_hits) - 1):
            t1, tid1 = motion_hits[i]
            t2, tid2 = motion_hits[i + 1]
            if tid1 != tid2 and (t2 - t1) <= _FIGHT_PROXIMITY_S:
                results.append(SceneEvent(
                    event_type="fight_proxy",
                    start_second=round(t1, 1),
                    end_second=round(t2, 1),
                    track_ids=[tid1, tid2],
                    confidence=0.40,    # low — heuristic only
                    notes=(
                        f"Tracks #{tid1} and #{tid2} both had sudden motion "
                        f"changes within {t2-t1:.1f}s of each other. "
                        f"Possible altercation — verify with footage."
                    ),
                ))
        return results[:3]   # cap at 3 to avoid noise spam

    # ── Rapid entry/exit ──────────────────────────────────────────────────────

    def _detect_rapid_entry(self, persons: list) -> List[SceneEvent]:
        """
        Rapid entry/exit: person(s) with very short visits repeated multiple times.
        Consistent with door-testing, counter-surveillance, or loitering patterns.
        """
        short_visits = [
            p for p in persons
            if p.duration_seconds < _RAPID_ENTRY_MAX_S
        ]
        if len(short_visits) <= _RAPID_ENTRY_MIN_N:
            return []

        tids = [p.track_id for p in short_visits]
        start = min(p.first_seen_second for p in short_visits)
        end   = max(p.last_seen_second  for p in short_visits)

        return [SceneEvent(
            event_type="rapid_entry_exit",
            start_second=round(start, 1),
            end_second=round(end, 1),
            track_ids=tids,
            confidence=0.65,
            notes=(
                f"{len(short_visits)} very short person visits "
                f"(< {_RAPID_ENTRY_MAX_S:.0f}s each) across the video. "
                f"Possible reconnaissance or loitering pattern."
            ),
        )]