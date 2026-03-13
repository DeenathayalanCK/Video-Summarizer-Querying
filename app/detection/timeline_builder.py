"""
timeline_builder.py — Builds the VideoTimeline event graph.

Each entry in the flat spine:
  second, time_label, event, track_id, object_class, detail, crop_path

crop_path is included so the UI can render a thumbnail per event.
Motion labels are class-aware:
  person  → enters / walking / running / sudden_stop / fall_detected / exits
  vehicle → enters / moving / exits   (never "walking" — vehicles don't walk)
"""

from app.core.logging import get_logger
from app.storage.models import TrackEvent, VideoTimeline
from app.detection.scene_understanding import SceneUnderstanding

# Vehicle classes — motion label vocabulary differs from persons
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Person motion states — direct mapping to human-readable labels
_PERSON_MOTION_LABEL = {
    "walking":    "walking",
    "running":    "running",
    "stationary": "standing still",
    "standing":   "standing still",
}

# Vehicle motion states — different vocabulary
_VEHICLE_MOTION_LABEL = {
    "walking":    "moving",      # displacement → vehicle moving, NOT "walking"
    "running":    "moving fast",
    "stationary": "stationary",
    "standing":   "stationary",
}


def _fmt(seconds: float) -> str:
    """
    Convert a timestamp to a display string.
    TrackEvent.first_seen_second stores a Unix epoch timestamp (float seconds).
    We convert to IST (UTC+5:30) for display.
    If somehow the value is small (< 86400 * 365, i.e. looks like a relative offset
    rather than a unix timestamp) we fall back to mm:ss relative display.
    """
    import datetime as _dt
    # Heuristic: unix timestamps for 2024+ are > 1.7 billion
    if seconds > 1_000_000_000:
        try:
            utc = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=seconds)
            ist = utc + _dt.timedelta(hours=5, minutes=30)
            return ist.strftime("%H:%M:%S")
        except Exception:
            pass
    # Fallback: relative mm:ss
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


class TimelineBuilder:

    def __init__(self, db):
        self.db = db
        self.logger = get_logger()

    def build(self, video_filename: str, camera_id: str) -> VideoTimeline:
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
            attrs  = ev.attributes or {}
            ms     = attrs.get("motion_summary", {})
            temp   = attrs.get("temporal", {})

            obj    = ev.object_class
            tid    = ev.track_id
            t_in   = ev.first_seen_second
            t_out  = ev.last_seen_second
            dur    = ev.duration_seconds
            crop   = ev.best_crop_path   # ← thumbnail for all events of this track

            is_vehicle = obj in _VEHICLE_CLASSES

            # ── Entry ─────────────────────────────────────────────────────────
            person_label = attrs.get("person_label") or f"track#{tid}"

            entries.append({
                "second":       round(t_in, 1),
                "time_label":   _fmt(t_in),
                "event":        "enters",
                "track_id":     tid,
                "person_label": person_label,
                "object_class": obj,
                "detail":       self._entry_detail(ev, attrs),
                "crop_path":    crop,
            })

            # ── Plate number event (vehicles only, when plate was OCR'd) ─────
            plate_num = attrs.get("plate_number", "unknown")
            if is_vehicle and plate_num and plate_num not in ("unknown", ""):
                entries.append({
                    "second":       round(t_in + 0.1, 1),   # just after entry
                    "time_label":   _fmt(t_in),
                    "event":        "plate_read",
                    "track_id":     tid,
                    "object_class": obj,
                    "detail":       f"License plate identified: {plate_num}",
                    "crop_path":    crop,
                })

            # ── Motion-event mid-points (fall / sudden_stop / direction_change)
            # These are person-only signals — skip for vehicles
            if not is_vehicle:
                motion_evts = ms.get("motion_events", [])
                for mev in motion_evts:
                    try:
                        sec = float(mev.split("@")[1].strip().split("s")[0])
                    except (IndexError, ValueError):
                        sec = t_in + dur / 2

                    if "fall_proxy" in mev:
                        label  = "fall_detected"
                        detail = f"ALERT: Possible fall — bounding box flipped to horizontal. {mev}"
                    elif "sudden_stop" in mev:
                        label  = "sudden_stop"
                        detail = f"Person stopped abruptly — possible object interaction. {mev}"
                    elif "direction_change" in mev:
                        label  = "direction_change"
                        detail = "Direction reversed — possible hesitation or reaction."
                    else:
                        continue

                    entries.append({
                        "second":       round(sec, 1),
                        "time_label":   _fmt(sec),
                        "event":        label,
                        "track_id":     tid,
                    "person_label": person_label,
                        "object_class": obj,
                        "detail":       detail,
                        "crop_path":    crop,
                    })

            # ── Dominant motion state mid-event ───────────────────────────────
            dom_state = ms.get("dominant_state", "")
            if dom_state and dom_state not in ("stationary", "standing", "unknown", ""):
                mid = t_in + dur * 0.4

                if is_vehicle:
                    # Vehicles: only emit "moving" / "moving fast" — skip stationary
                    motion_label = _VEHICLE_MOTION_LABEL.get(dom_state, "moving")
                    detail = self._vehicle_motion_detail(dom_state, ms, attrs)
                else:
                    motion_label = _PERSON_MOTION_LABEL.get(dom_state, dom_state)
                    detail = self._person_motion_detail(dom_state, ms)

                entries.append({
                    "second":       round(mid, 1),
                    "time_label":   _fmt(mid),
                    "event":        motion_label,
                    "track_id":     tid,
                    "person_label": person_label,
                    "object_class": obj,
                    "detail":       detail,
                    "crop_path":    crop,
                })

            # ── Behaviour label (temporal analysis, persons mostly) ───────────
            behaviour = temp.get("behaviour", "")
            skip_behaviours = {"passing_through", "passing", "unknown", ""}
            if behaviour not in skip_behaviours:
                mid2 = t_in + dur * 0.6
                entries.append({
                    "second":       round(mid2, 1),
                    "time_label":   _fmt(mid2),
                    "event":        behaviour,
                    "track_id":     tid,
                    "object_class": obj,
                    "detail":       temp.get("notes", f"Behaviour classified as: {behaviour}."),
                    "crop_path":    crop,
                })

            # ── Exit ──────────────────────────────────────────────────────────
            entries.append({
                "second":       round(t_out, 1),
                "time_label":   _fmt(t_out),
                "event":        "exits",
                "track_id":     tid,
                "object_class": obj,
                "detail":       f"{obj.capitalize()} leaves scene after {dur:.0f}s.",
                "crop_path":    crop,
            })

        entries.sort(key=lambda e: e["second"])

        # ── Scene-level events ────────────────────────────────────────────────
        scene_analyser = SceneUnderstanding()
        scene_evts     = scene_analyser.analyse(events)
        scene_dicts    = [s.to_dict() for s in scene_evts]

        for se in scene_evts:
            entries.append({
                "second":       se.start_second,
                "time_label":   _fmt(se.start_second),
                "event":        se.event_type,
                "track_id":     None,
                "object_class": "scene",
                "detail":       se.notes,
                "crop_path":    None,   # scene events have no crop
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
            existing.timeline_entries          = entries
            existing.scene_events              = scene_dicts
            existing.total_duration_seconds    = duration
            existing.entry_count               = len(entries)
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

    # ── Detail builders ───────────────────────────────────────────────────────

    def _entry_detail(self, ev, attrs: dict) -> str:
        obj   = ev.object_class
        parts = []
        if obj == "person":
            for k in ("gender_estimate", "age_estimate", "clothing_top", "visible_text"):
                v = attrs.get(k, "")
                if v and v not in ("unknown", "none", ""):
                    parts.append(f"[{v}]" if k == "visible_text" else v)
        else:
            for k in ("color", "type", "make_estimate"):
                v = attrs.get(k, "")
                if v and v not in ("unknown", "none", ""):
                    parts.append(v)
            # Include plate number in entry detail if available
            plate = attrs.get("plate_number", "unknown")
            if plate and plate not in ("unknown", ""):
                parts.append(f"[plate:{plate}]")
        desc = " ".join(parts) if parts else obj.capitalize()
        return f"{desc} enters scene at {_fmt(ev.first_seen_second)}."

    def _person_motion_detail(self, dom_state: str, ms: dict) -> str:
        direction = ms.get("direction", "")
        dir_str   = direction.replace("_", " ") if direction not in ("stationary", "") else ""
        label     = _PERSON_MOTION_LABEL.get(dom_state, dom_state)
        return f"Person is {label}{' (' + dir_str + ')' if dir_str else ''}."

    def _vehicle_motion_detail(self, dom_state: str, ms: dict, attrs: dict) -> str:
        direction = ms.get("direction", "")
        dir_str   = direction.replace("_", " ") if direction not in ("stationary", "") else ""
        color     = attrs.get("color", "")
        vtype     = attrs.get("type", "")
        desc      = " ".join(p for p in [color, vtype] if p and p != "unknown") or "Vehicle"
        motion    = _VEHICLE_MOTION_LABEL.get(dom_state, "moving")
        return f"{desc} is {motion}{' (' + dir_str + ')' if dir_str else ''}."