"""
fast_path.py — Direct DB answers for factual queries, bypassing the LLM.

For simple factual questions the answer already exists in the database as
structured data. Calling the LLM for these is wasteful — it takes 2-7 minutes
and produces the same answer a DB query gives in milliseconds.

Fast-path triggers:
  plate     → "plate 7272", "plate number", "registration"
  count     → "how many cars", "how many people"
  identity  → "what colour is car", "what was person wearing"
  time      → "when did X enter", "what time did X leave"
  presence  → "was there a car", "did anyone enter", "is there a person"
  behaviour → "was anyone loitering", "did anyone fall", "was anyone running"

Each resolver returns:
  {"answered": True,  "answer": "...", "sources": [...]}   ← skip LLM
  {"answered": False}                                       ← fall through to LLM
"""

import re
from typing import Optional
from sqlalchemy.orm import Session

from app.storage.models import TrackEvent, SemanticMemoryGraph, VideoTimeline
from app.core.logging import get_logger

_log = get_logger()

# ── Intent patterns ──────────────────────────────────────────────────────────

_PLATE_PATTERNS = [
    r"\bplate\b", r"\bregistration\b", r"\bnumber plate\b",
    r"\blicen[sc]e\b", r"\brego\b",
]
_COUNT_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal\b",
]
_PRESENCE_PATTERNS = [
    r"\bwas there\b", r"\bis there\b", r"\bdid (?:any|a)\b",
    r"\bwere there\b", r"\banyone\b", r"\bany (?:car|vehicle|person|truck|bus)\b",
    r"\bdid (?:anyone|somebody)\b",
]
_BEHAVIOUR_PATTERNS = [
    r"\bloiter", r"\bfall\b", r"\bfell\b", r"\brun\b", r"\brunning\b",
    r"\bsudden.?stop\b", r"\bparked?\b", r"\bwait", r"\bpatrol",
    r"\bstationary\b", r"\bstanding\b",
]
_WHEN_PATTERNS = [
    r"\bwhen\b.*\benter", r"\bwhen\b.*\bleav", r"\bwhat time\b",
    r"\bfirst seen\b", r"\blast seen\b", r"\barriv", r"\bdeparted?\b",
]
_IDENTITY_PATTERNS = [
    r"\bwhat colou?r\b", r"\bwhat.+wear", r"\bcloth", r"\bshirt\b",
    r"\bdescrib", r"\bappearance\b", r"\bwhat.+look",
]

_OBJECT_CLASSES = {
    "person": ["person", "people", "man", "woman", "someone", "individual"],
    "car":    ["car", "vehicle", "sedan", "automobile"],
    "truck":  ["truck", "lorry", "van"],
    "bus":    ["bus", "coach"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "bicycle": ["bicycle", "cycle", "cyclist"],
}


def _fmt(s: float) -> str:
    return f"{int(s)//60}:{int(s)%60:02d}"


def _match(patterns: list, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _detect_class(q: str) -> Optional[str]:
    for cls, keywords in _OBJECT_CLASSES.items():
        if any(k in q.lower() for k in keywords):
            return cls
    return None


def _entry_events(db: Session, video_filename: Optional[str], obj_class: Optional[str]) -> list:
    q = db.query(TrackEvent).filter(TrackEvent.event_type == "entry")
    if video_filename:
        q = q.filter(TrackEvent.video_filename == video_filename)
    if obj_class:
        q = q.filter(TrackEvent.object_class == obj_class)
    return q.order_by(TrackEvent.first_seen_second).all()


def _memory_nodes(db: Session, video_filename: Optional[str], node_type: Optional[str] = None) -> list:
    q = db.query(SemanticMemoryGraph)
    if video_filename:
        q = q.filter(SemanticMemoryGraph.video_filename == video_filename)
    if node_type:
        q = q.filter(SemanticMemoryGraph.node_type == node_type)
    return q.order_by(SemanticMemoryGraph.start_second).all()


def _source_from_event(ev: TrackEvent) -> dict:
    return {
        "video_filename": ev.video_filename,
        "track_id": ev.track_id,
        "object_class": ev.object_class,
        "first_seen": ev.first_seen_second,
        "last_seen": ev.last_seen_second,
        "best_crop_path": ev.best_crop_path,
        "retrieval_reason": "fast_path",
    }


# ── Resolvers ────────────────────────────────────────────────────────────────

def _resolve_plate(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer 'is there a car with plate X?' directly from attributes."""
    # Extract digits/letters that look like a plate fragment
    # Match things like "7272", "TN09", "AB1234"
    plate_fragments = re.findall(r"\b([A-Z0-9]{2,8})\b", q.upper())
    # Remove common English words that got uppercased
    stop_upper = {"IS","ARE","ANY","CAR","THE","WITH","THAT","HAS","HAVE",
                  "WHAT","WAS","THERE","PLATE","NUMBER","CONTAINS","IN","IT"}
    plate_fragments = [f for f in plate_fragments if f not in stop_upper]

    events = _entry_events(db, video_filename, None)
    vehicle_classes = {"car","truck","bus","motorcycle","bicycle"}

    matches = []
    for ev in events:
        if ev.object_class not in vehicle_classes:
            continue
        attrs = ev.attributes or {}
        plate = str(attrs.get("plate_number", "")).upper()
        if not plate or plate in ("UNKNOWN", ""):
            # Also check plate_visible flag
            if attrs.get("plate_visible"):
                plate = "visible but unread"
            else:
                continue
        # Check if any fragment matches
        if plate_fragments:
            if any(frag in plate for frag in plate_fragments):
                matches.append((ev, plate))
        else:
            # No specific plate asked — list all plates
            matches.append((ev, plate))

    if not matches:
        # Also check memory graph for plate nodes
        nodes = _memory_nodes(db, video_filename, "identity")
        plate_nodes = [n for n in nodes if "plate" in n.semantic_text.lower()
                       and (not plate_fragments or
                            any(f in n.semantic_text.upper() for f in plate_fragments))]
        if plate_nodes:
            lines = [n.semantic_text for n in plate_nodes[:5]]
            return {
                "answered": True,
                "answer": "Yes. " + " ".join(lines),
                "sources": [{"video_filename": n.video_filename,
                              "track_id": n.track_id,
                              "retrieval_reason": "fast_path_memory_graph"}
                            for n in plate_nodes[:3]],
            }
        frag_str = ", ".join(plate_fragments) if plate_fragments else "any plate"
        return {
            "answered": True,
            "answer": f"No vehicle with a plate matching '{frag_str}' was found in the footage. "
                      f"Plates may not have been visible or readable in the available crops.",
            "sources": [],
        }

    lines = []
    sources = []
    for ev, plate in matches[:5]:
        attrs = ev.attributes or {}
        color = attrs.get("color", "")
        vtype = attrs.get("type", ev.object_class)
        desc = " ".join(p for p in [color, vtype] if p and p != "unknown")
        lines.append(
            f"Yes — {desc or ev.object_class} (track #{ev.track_id}) "
            f"with plate **{plate}** was seen at {_fmt(ev.first_seen_second)} "
            f"in {ev.video_filename}."
        )
        sources.append(_source_from_event(ev))

    return {"answered": True, "answer": "\n".join(lines), "sources": sources}


def _resolve_count(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer 'how many X' by counting unique track IDs."""
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class)

    if not events:
        cls_str = obj_class or "objects"
        return {
            "answered": True,
            "answer": f"No {cls_str} were detected in the footage.",
            "sources": [],
        }

    # Group by video
    by_video: dict = {}
    for ev in events:
        by_video.setdefault(ev.video_filename, []).append(ev)

    lines = []
    sources = []
    for vf, evs in sorted(by_video.items()):
        cls_str = obj_class or "unique objects"
        lines.append(f"**{len(evs)}** {cls_str}(s) detected in `{vf}`.")
        # Breakdown by class if no class filter
        if not obj_class:
            class_counts: dict = {}
            for ev in evs:
                class_counts[ev.object_class] = class_counts.get(ev.object_class, 0) + 1
            breakdown = ", ".join(f"{v} {k}" for k, v in sorted(class_counts.items()))
            lines.append(f"  Breakdown: {breakdown}")
        sources.extend(_source_from_event(ev) for ev in evs[:3])

    return {"answered": True, "answer": "\n".join(lines), "sources": sources[:6]}


def _resolve_presence(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer 'was there a X / did anyone X' quickly."""
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class)

    if not events:
        cls_str = obj_class or "the object"
        return {
            "answered": True,
            "answer": f"No — {cls_str} was not detected in the footage.",
            "sources": [],
        }

    cls_str = obj_class or "objects"
    count = len(events)
    first = events[0]
    last  = events[-1]

    answer = (
        f"Yes — **{count}** {cls_str}(s) were detected. "
        f"First appearance at {_fmt(first.first_seen_second)}, "
        f"last at {_fmt(last.first_seen_second)} "
        f"(in `{first.video_filename}`)."
    )
    sources = [_source_from_event(ev) for ev in events[:4]]
    return {"answered": True, "answer": answer, "sources": sources}


def _resolve_behaviour(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer behaviour questions from memory graph behaviour nodes."""
    # Map query words to behaviour labels
    behaviour_map = {
        "loiter": "loitering", "fall": "fall_detected", "fell": "fall_detected",
        "run": "running", "running": "running", "sudden": "sudden_stop",
        "stop": "sudden_stop", "park": "parked", "wait": "waiting",
        "patrol": "patrolling", "station": "stationary", "stand": "stationary",
    }
    q_lower = q.lower()
    target_behaviours = []
    for word, label in behaviour_map.items():
        if word in q_lower:
            target_behaviours.append(label)

    # Search memory graph behaviour + timeline nodes
    nodes = _memory_nodes(db, video_filename, "behaviour")
    nodes += _memory_nodes(db, video_filename, "timeline")

    matching = []
    for node in nodes:
        text_lower = node.semantic_text.lower()
        label_lower = node.node_label.lower()
        if target_behaviours:
            if any(b.replace("_", " ") in text_lower or b in text_lower
                   or b.replace("_"," ") in label_lower for b in target_behaviours):
                matching.append(node)
        else:
            matching.append(node)  # no specific behaviour — return all

    if not matching:
        # Also check TrackEvent.attributes.temporal directly
        events = _entry_events(db, video_filename, None)
        for ev in events:
            beh = (ev.attributes or {}).get("temporal", {}).get("behaviour", "")
            if beh and any(b in beh for b in target_behaviours):
                attrs = ev.attributes or {}
                color = attrs.get("color", "") or attrs.get("clothing_top", "")
                desc = f"{color} {ev.object_class}".strip()
                matching_answer = (
                    f"Yes — {desc} (track #{ev.track_id}) showed **{beh}** behaviour "
                    f"from {_fmt(ev.first_seen_second)} to {_fmt(ev.last_seen_second)} "
                    f"in `{ev.video_filename}`."
                )
                return {
                    "answered": True,
                    "answer": matching_answer,
                    "sources": [_source_from_event(ev)],
                }

        beh_str = " or ".join(set(target_behaviours)) if target_behaviours else "notable behaviour"
        return {
            "answered": True,
            "answer": f"No {beh_str} was detected in the footage.",
            "sources": [],
        }

    lines = [n.semantic_text for n in matching[:5]]
    sources = [{"video_filename": n.video_filename, "track_id": n.track_id,
                "retrieval_reason": "fast_path_memory_graph"}
               for n in matching[:5] if n.track_id]

    return {"answered": True, "answer": "\n\n".join(lines), "sources": sources}


def _resolve_when(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer 'when did X enter/leave?' from track timestamps."""
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class)

    if not events:
        return {"answered": True,
                "answer": "No matching tracks found in the footage.", "sources": []}

    lines = []
    for ev in events[:6]:
        attrs = ev.attributes or {}
        color = attrs.get("color", "") or attrs.get("clothing_top", "")
        desc = f"{color} {ev.object_class}".strip() if color else ev.object_class
        lines.append(
            f"**{desc} #{ev.track_id}**: entered {_fmt(ev.first_seen_second)}, "
            f"left {_fmt(ev.last_seen_second)}, "
            f"present for {ev.duration_seconds:.0f}s — `{ev.video_filename}`"
        )

    return {
        "answered": True,
        "answer": "\n".join(lines),
        "sources": [_source_from_event(ev) for ev in events[:4]],
    }


def _resolve_identity(db, q: str, video_filename: Optional[str]) -> dict:
    """Answer 'what colour is X / what was person wearing' from attributes."""
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class)

    matches = [ev for ev in events if ev.attributes]
    if not matches:
        return {"answered": False}  # Let LLM handle if no attributes

    lines = []
    for ev in matches[:5]:
        attrs = ev.attributes or {}
        if ev.object_class in ("car","truck","bus","motorcycle","bicycle"):
            color   = attrs.get("color", "unknown")
            vtype   = attrs.get("type", "vehicle")
            make    = attrs.get("make_estimate", "")
            plate   = attrs.get("plate_number", "")
            desc = f"**{color} {vtype}**"
            if make and make != "unknown":
                desc += f" ({make})"
            if plate and plate != "unknown":
                desc += f", plate: {plate}"
            lines.append(
                f"Track #{ev.track_id}: {desc} — "
                f"seen {_fmt(ev.first_seen_second)}–{_fmt(ev.last_seen_second)} "
                f"in `{ev.video_filename}`"
            )
        else:
            gender  = attrs.get("gender_estimate", "")
            age     = attrs.get("age_estimate", "")
            top     = attrs.get("clothing_top", "")
            bottom  = attrs.get("clothing_bottom", "")
            carry   = attrs.get("carrying", "")
            vis     = attrs.get("visible_text", "")
            parts = [p for p in [gender, age, top, bottom]
                     if p and p not in ("unknown","none","")]
            if carry and carry not in ("unknown","none"):
                parts.append(f"carrying {carry}")
            if vis and vis not in ("none","unknown"):
                parts.append(f'badge/text: "{vis}"')
            desc = ", ".join(parts) if parts else "appearance unknown"
            lines.append(
                f"Track #{ev.track_id}: **{desc}** — "
                f"seen {_fmt(ev.first_seen_second)}–{_fmt(ev.last_seen_second)} "
                f"in `{ev.video_filename}`"
            )

    return {
        "answered": True,
        "answer": "\n".join(lines),
        "sources": [_source_from_event(ev) for ev in matches[:4]],
    }


# ── Public entry point ────────────────────────────────────────────────────────

def try_fast_path(
    db: Session,
    question: str,
    video_filename: Optional[str] = None,
) -> dict:
    """
    Try to answer the question directly from the database.

    Returns:
      {"answered": True,  "answer": "...", "sources": [...], "fast_path_type": "..."}
      {"answered": False}   ← caller should proceed to LLM
    """
    q = question.strip()

    # Plate query — highest priority, very specific
    if _match(_PLATE_PATTERNS, q):
        _log.info("fast_path_triggered", type="plate", question=q)
        result = _resolve_plate(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "plate"
            return result

    # Behaviour query — memory graph nodes answer directly
    if _match(_BEHAVIOUR_PATTERNS, q):
        _log.info("fast_path_triggered", type="behaviour", question=q)
        result = _resolve_behaviour(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "behaviour"
            return result

    # Count query
    if _match(_COUNT_PATTERNS, q):
        _log.info("fast_path_triggered", type="count", question=q)
        result = _resolve_count(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "count"
            return result

    # When/time query
    if _match(_WHEN_PATTERNS, q):
        _log.info("fast_path_triggered", type="when", question=q)
        result = _resolve_when(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "when"
            return result

    # Identity/appearance query
    if _match(_IDENTITY_PATTERNS, q):
        _log.info("fast_path_triggered", type="identity", question=q)
        result = _resolve_identity(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "identity"
            return result

    # Presence query — simple yes/no
    if _match(_PRESENCE_PATTERNS, q):
        _log.info("fast_path_triggered", type="presence", question=q)
        result = _resolve_presence(db, q, video_filename)
        if result["answered"]:
            result["fast_path_type"] = "presence"
            return result

    _log.info("fast_path_miss", question=q)
    return {"answered": False}