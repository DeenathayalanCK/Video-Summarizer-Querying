"""
timeline_builder.py — Builds the VideoTimeline event graph.

Converts TrackEvent + motion_summary data into a human-readable
ordered list of {second, event, track_id, object_class, detail} entries.

Example output:
  00:01  person #3  enters      "Person enters from top-left quadrant"
  00:05  person #3  walking     "Walking left-to-right"
  00:09  person #3  picks_up    "Sudden stop detected — possible object interaction"
  00:14  person #3  exits       "Person leaves scene"
  00:05  car    #1  enters      "Blue sedan enters (top-right)"
  00:07  car    #1  exits       "Vehicle exits after 2s"

This makes Q&A queries like "what happened at 0:09?" directly answerable
from structured data without re-running the LLM on raw frame data.
"""

from app.core.logging import get_logger
from app.storage.models import TrackEvent, VideoTimeline
from app.detection.scene_understanding import SceneUnderstanding


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


class TimelineBuilder:
    """
    Builds and persists VideoTimeline for a video.
    Called after all TrackEvents and temporal attributes are written.
    """

    def __init__(self, db):
        self.db = db
        self.logger = get_logger()

    def build(self, video_filename: str, camera_id: str) -> VideoTimeline:
        """
        Build VideoTimeline from TrackEvent rows.
        Replaces any existing timeline for this video.
        Returns the saved VideoTimeline object.
        """
        # Load all entry-type events (one per physical track)
        events = (
            self.db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",
            )
            .order_by(TrackEvent.first_seen_second)
            .all()
        )

        if not events:
            self.logger.warning("timeline_build_no_events", video=video_filename)
            return None

        entries = []

        for ev in events:
            attrs = ev.attributes or {}
            ms    = attrs.get("motion_summary", {})
            temp  = attrs.get("temporal", {})

            obj   = ev.object_class
            tid   = ev.track_id
            t_in  = ev.first_seen_second
            t_out = ev.last_seen_second
            dur   = ev.duration_seconds

            # ── Entry event ───────────────────────────────────────────────────
            entries.append({
                "second":       round(t_in, 1),
                "time_label":   _fmt(t_in),
                "event":        "enters",
                "track_id":     tid,
                "object_class": obj,
                "detail":       self._entry_detail(ev, attrs, temp),
            })

            # ── Motion-derived mid-events ─────────────────────────────────────
            motion_evts = ms.get("motion_events", [])
            dom_state   = ms.get("dominant_state", "")

            for mev in motion_evts:
                try:
                    sec = float(mev.split("@")[1].strip().split("s")[0])
                except (IndexError, ValueError):
                    sec = t_in + dur / 2

                if "fall_proxy" in mev:
                    label  = "fall_detected"
                    detail = f"ALERT: Possible fall — bounding box shifted from upright to horizontal. {mev}"
                elif "sudden_stop" in mev:
                    label  = "sudden_stop"
                    detail = f"Person stopped abruptly — possible object interaction or confrontation. {mev}"
                elif "direction_change" in mev:
                    label  = "direction_change"
                    detail = f"Direction reversed — possible hesitation or reaction to stimulus."
                else:
                    continue

                entries.append({
                    "second":       round(sec, 1),
                    "time_label":   _fmt(sec),
                    "event":        label,
                    "track_id":     tid,
                    "object_class": obj,
                    "detail":       detail,
                })

            # ── Dominant motion state (mid-presence) ──────────────────────────
            if dom_state and dom_state not in ("stationary", "standing", "unknown"):
                mid = t_in + dur * 0.4
                entries.append({
                    "second":       round(mid, 1),
                    "time_label":   _fmt(mid),
                    "event":        dom_state,   # "walking" | "running"
                    "track_id":     tid,
                    "object_class": obj,
                    "detail":       self._motion_detail(dom_state, ms, attrs),
                })

            # ── Behaviour label (from temporal analysis) ──────────────────────
            behaviour = temp.get("behaviour", "")
            if behaviour and behaviour not in ("passing_through", "passing", "unknown"):
                mid2 = t_in + dur * 0.6
                entries.append({
                    "second":       round(mid2, 1),
                    "time_label":   _fmt(mid2),
                    "event":        behaviour,
                    "track_id":     tid,
                    "object_class": obj,
                    "detail":       temp.get("notes", f"Behaviour: {behaviour}"),
                })

            # ── Exit event ────────────────────────────────────────────────────
            entries.append({
                "second":       round(t_out, 1),
                "time_label":   _fmt(t_out),
                "event":        "exits",
                "track_id":     tid,
                "object_class": obj,
                "detail":       f"{obj.capitalize()} leaves scene after {dur:.0f}s.",
            })

        # Sort all entries by time
        entries.sort(key=lambda e: e["second"])

        # ── Scene-level events ────────────────────────────────────────────────
        scene_analyser = SceneUnderstanding()
        scene_evts = scene_analyser.analyse(events)
        scene_dicts = [s.to_dict() for s in scene_evts]

        # Also inject scene events into the flat timeline
        for se in scene_evts:
            entries.append({
                "second":       se.start_second,
                "time_label":   _fmt(se.start_second),
                "event":        se.event_type,
                "track_id":     None,
                "object_class": "scene",
                "detail":       se.notes,
            })
        entries.sort(key=lambda e: e["second"])

        # ── Persist ───────────────────────────────────────────────────────────
        existing = (
            self.db.query(VideoTimeline)
            .filter(VideoTimeline.video_filename == video_filename)
            .first()
        )
        duration = max(ev.last_seen_second for ev in events) if events else 0.0

        if existing:
            existing.timeline_entries = entries
            existing.scene_events = scene_dicts
            existing.total_duration_seconds = duration
            existing.entry_count = len(entries)
            tl = existing
        else:
            tl = VideoTimeline(
                video_filename=video_filename,
                camera_id=camera_id,
                timeline_entries=entries,
                scene_events=scene_dicts,
                total_duration_seconds=duration,
                entry_count=len(entries),
            )
            self.db.add(tl)

        self.db.commit()
        self.logger.info(
            "timeline_built",
            video=video_filename,
            entries=len(entries),
            scene_events=len(scene_dicts),
        )
        return tl

    # ── Detail string builders ────────────────────────────────────────────────

    def _entry_detail(self, ev, attrs: dict, temp: dict) -> str:
        obj = ev.object_class
        parts = []

        if obj == "person":
            gender = attrs.get("gender_estimate", "")
            age    = attrs.get("age_estimate", "")
            top    = attrs.get("clothing_top", "")
            vis    = attrs.get("visible_text", "")
            if gender and gender != "unknown": parts.append(gender)
            if age    and age    != "unknown": parts.append(age)
            if top    and top    != "unknown": parts.append(top)
            if vis    and vis not in ("none", "unknown", ""): parts.append(f'[{vis}]')
        else:
            color = attrs.get("color", "")
            vtype = attrs.get("type", "")
            make  = attrs.get("make_estimate", "")
            if color and color != "unknown": parts.append(color)
            if vtype and vtype != "unknown": parts.append(vtype)
            if make  and make  != "unknown": parts.append(make)

        desc = " ".join(parts) if parts else obj.capitalize()
        return f"{desc} enters scene at {_fmt(ev.first_seen_second)}."

    def _motion_detail(self, dom_state: str, ms: dict, attrs: dict) -> str:
        direction = ms.get("direction", "")
        dir_str   = direction.replace("_", " ") if direction != "stationary" else ""
        base = f"Person is {dom_state}"
        return f"{base} ({dir_str})." if dir_str else f"{base}."