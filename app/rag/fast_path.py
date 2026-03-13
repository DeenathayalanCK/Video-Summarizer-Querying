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

# Keywords that indicate a carrying/accessory query — NOT a YOLO object_class
# but stored in TrackEvent.attributes['carrying'] via moondream + YOLO wiring
_CARRYING_KEYWORDS = {
    "backpack": ["backpack", "rucksack", "knapsack"],
    "bag":      ["bag", "handbag", "purse", "tote"],
    "luggage":  ["luggage", "suitcase", "trolley", "baggage"],
    "briefcase":["briefcase"],
}


# ── Time-range helpers ───────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}


def _parse_time_range(q: str):
    """
    Extract a UTC epoch [min, max) range and matching window-name suffixes
    from a natural-language question like "Mar 13 between 10:30-10:40".

    Returns (min_epoch, max_epoch, window_suffixes: list[str]) or None.
    window_suffixes is a list of "YYYYMMDD_HHMM" strings that overlap the range
    (one per 5-minute boundary inside [start, end)).
    """
    import datetime as _dt
    q_lower = q.lower()

    # ── Date extraction ───────────────────────────────────────────────────────
    year = _dt.datetime.now(_dt.timezone.utc).year
    month = day = None

    m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec\w*)\s+(\d{1,2})\b', q_lower)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3])
        day   = int(m.group(2))
    else:
        m = re.search(r'\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec\w*)\b', q_lower)
        if m:
            day   = int(m.group(1))
            month = _MONTH_MAP.get(m.group(2)[:3])
    if not month:
        m = re.search(r'\b(\d{1,2})[/-](\d{1,2})\b', q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= 12 and b <= 31:
                month, day = a, b
            elif b <= 12 and a <= 31:
                day, month = a, b

    # ── Time range extraction ─────────────────────────────────────────────────
    times = re.findall(r'(\d{1,2}):(\d{2})', q)
    if len(times) < 2:
        return None
    t_start = (int(times[0][0]), int(times[0][1]))
    t_end   = (int(times[1][0]), int(times[1][1]))

    if not (month and day):
        return None

    try:
        # Times in questions are in IST (UTC+5:30) — the UI, filenames, and
        # display all use IST.  Convert to UTC epoch for DB comparison.
        IST_OFFSET = _dt.timedelta(hours=5, minutes=30)
        tz  = _dt.timezone.utc
        dts_ist = _dt.datetime(year, month, day, t_start[0], t_start[1])
        dte_ist = _dt.datetime(year, month, day, t_end[0],   t_end[1])
        dts = (dts_ist - IST_OFFSET).replace(tzinfo=tz)   # IST → UTC
        dte = (dte_ist - IST_OFFSET).replace(tzinfo=tz)   # IST → UTC
    except ValueError:
        return None

    # Build 5-min window suffixes that overlap [dts, dte)
    suffixes = []
    cur = dts.replace(minute=(dts.minute // 5) * 5, second=0, microsecond=0)
    while cur < dte:
        suffixes.append(cur.strftime('%Y%m%d_%H%M'))
        cur += _dt.timedelta(minutes=5)

    return dts.timestamp(), dte.timestamp(), suffixes


def _filter_events_by_time(events: list, min_epoch: float, max_epoch: float) -> list:
    """Keep only events whose track overlaps the given epoch range."""
    return [
        ev for ev in events
        if ev.first_seen_second < max_epoch and ev.last_seen_second >= min_epoch
    ]


def _fmt(s: float) -> str:
    """Format a timestamp for display.

    TrackEvent.first_seen_second stores Unix epoch seconds (e.g. 1773135152).
    Displaying epoch//60 as minutes produces nonsense like '29552252:32'.
    Detect epoch values (> 1_000_000_000) and convert to wall-clock HH:MM:SS.
    Legacy offset-seconds data (< 86400) is formatted as M:SS.
    """
    import datetime as _dt
    s = float(s)
    if s > 1_000_000_000:  # Unix epoch timestamp
        return _dt.datetime.fromtimestamp(s, tz=_dt.timezone.utc).strftime("%H:%M:%S")
    si = int(s)
    return f"{si//60}:{si%60:02d}"


def _match(patterns: list, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _detect_class(q: str) -> Optional[str]:
    for cls, keywords in _OBJECT_CLASSES.items():
        if any(k in q.lower() for k in keywords):
            return cls
    return None


def _entry_events(
    db: Session,
    video_filename: Optional[str],
    obj_class: Optional[str],
    min_second: Optional[float] = None,
    max_second: Optional[float] = None,
) -> list:
    q = db.query(TrackEvent).filter(TrackEvent.event_type == "entry")
    if video_filename:
        q = q.filter(TrackEvent.video_filename == video_filename)
    if obj_class:
        q = q.filter(TrackEvent.object_class == obj_class)
    # Apply time range filter when provided (epoch seconds in first_seen_second)
    if min_second is not None:
        q = q.filter(TrackEvent.last_seen_second >= min_second)
    if max_second is not None:
        q = q.filter(TrackEvent.first_seen_second < max_second)
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

def _resolve_plate(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """Answer 'is there a car with plate X?' directly from attributes."""
    # Extract digits/letters that look like a plate fragment
    # Match things like "7272", "TN09", "AB1234"
    plate_fragments = re.findall(r"\b([A-Z0-9]{2,8})\b", q.upper())
    # Remove common English words that got uppercased
    stop_upper = {"IS","ARE","ANY","CAR","THE","WITH","THAT","HAS","HAVE",
                  "WHAT","WAS","THERE","PLATE","NUMBER","CONTAINS","IN","IT"}
    plate_fragments = [f for f in plate_fragments if f not in stop_upper]

    events = _entry_events(db, video_filename, None, min_second, max_second)
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


def _resolve_count(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """Answer 'how many X' by counting unique track IDs.

    Produces a clean natural-language answer directly in Python — no LLM needed.
    Sets no_curate=True so qa_engine skips the Ollama curate call entirely,
    avoiding the 25s timeout when llama3.2 is busy with pipeline attribute calls.
    """
    import datetime as _dt
    _IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))

    def _fmt_ist(epoch: float) -> str:
        """Format epoch as HH:MM in IST for display."""
        return _dt.datetime.fromtimestamp(epoch, tz=_IST).strftime("%H:%M")

    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class, min_second, max_second)

    # IST time context for display ("between 10:30-10:40")
    time_ctx = ""
    if min_second is not None and max_second is not None:
        time_ctx = f" between {_fmt_ist(min_second)}\u2013{_fmt_ist(max_second)}"

    if not events:
        cls_str = obj_class or "persons or objects"
        return {
            "answered": True,
            "answer": f"No {cls_str} were detected{time_ctx}.",
            "no_curate": True,
            "sources": [],
        }

    total = len(events)
    cls_str = obj_class or "people"

    # Group by video window
    by_video: dict = {}
    for ev in events:
        by_video.setdefault(ev.video_filename, []).append(ev)
    nw = len(by_video)

    sources = []
    for vf, evs in sorted(by_video.items()):
        sources.extend(_source_from_event(ev) for ev in evs[:2])

    # Clean natural-language answer — no markdown, no backticks, no LLM needed
    window_phrase = "1 window" if nw == 1 else f"{nw} windows"
    answer = f"{total} {cls_str} were detected{time_ctx} across {window_phrase}."

    # Structured raw_facts for DB Evidence panel (uses markdown, not sent to LLM)
    t_prefix = (f"Between {_fmt_ist(min_second)}\u2013{_fmt_ist(max_second)}: "
                if (min_second and max_second) else "")
    raw_lines = [f"{t_prefix}**{total}** {cls_str} detected across {nw} window(s)."]
    for vf, evs in sorted(by_video.items())[:5]:
        raw_lines.append(f"  {len(evs)} in `{vf}`")
    if nw > 5:
        raw_lines.append(f"  ... and {nw - 5} more window(s)")

    return {
        "answered": True,
        "answer": answer,               # clean NL shown to user directly
        "raw_facts": "\n".join(raw_lines),  # structured shown in DB Evidence panel
        "no_curate": True,              # skip Ollama — answer already ready
        "sources": sources[:6],
    }


def _resolve_presence(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """Answer 'was there a X / did anyone X' quickly.

    Open-ended questions like 'was there anything suspicious?' must NOT be
    answered here — they require LLM reasoning. Only answer simple yes/no
    presence checks for a specific object class or a specific video.
    """
    # Bail out on open-ended / vague queries that need LLM reasoning
    _open_ended = [
        r"\bsuspicious\b", r"\banything\b", r"\bsomething\b", r"\bunusual\b",
        r"\bwrong\b", r"\bnothing\b", r"\beverything\b",
        r"\bany.{0,5}(issue|problem|concern)\b",
    ]
    if any(re.search(p, q, re.I) for p in _open_ended):
        return {"answered": False}

    obj_class = _detect_class(q)

    # Without a specific class AND no specific video, counting every track in
    # the DB is meaningless — fall through to LLM instead.
    if not obj_class and not video_filename:
        return {"answered": False}

    events = _entry_events(db, video_filename, obj_class, min_second, max_second)

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


def _resolve_when(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """Answer 'when did X enter/leave?' from track timestamps."""
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class, min_second, max_second)

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


def _resolve_identity(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """Answer 'what colour is X / what was person wearing' from attributes.
    
    Bug fix: When video_filename is None (All videos), old code returned
    tracks from all windows including old ones with attr_has_data=False,
    showing 'appearance unknown' for tracks whose LLM timed out.
    Now filters to prefer tracks with real attribute data.
    Also uses Tier-1 CV color fields (clothing_top_color/bottom_color) as fallback.
    """
    obj_class = _detect_class(q)
    events = _entry_events(db, video_filename, obj_class, min_second, max_second)

    if not events:
        return {"answered": False}

    # Tier priority:
    #   rich  = LLM ran and got data (attr_has_data=True)
    #   cv    = CV tier-1 only (attr_tier1_only=True, clothing_top_color set)
    #   none  = no attributes at all (old window, extraction not run)
    # When All videos is selected, only show rich+cv — never show "appearance unknown"
    # from old windows alongside good results from a specific window.
    rich_events = [ev for ev in events if (ev.attributes or {}).get("attr_has_data")]
    cv_events   = [ev for ev in events if (ev.attributes or {}).get("attr_tier1_only")]

    if rich_events:
        # Best case: full LLM data available somewhere
        matches = rich_events
    elif cv_events:
        # CV-only data (color histogram), still useful
        matches = cv_events
    else:
        # No processed attrs at all — let LLM handle rather than show "unknown"
        # Unless they specifically asked "is there a person" (presence, not identity)
        matches_any = [ev for ev in events if ev.attributes]
        if not matches_any:
            return {"answered": False}
        # Check if any have visible non-unknown attrs
        has_real = [ev for ev in matches_any if any(
            (ev.attributes.get(k) or "") not in ("unknown", "none", "")
            for k in ("clothing_top", "clothing_bottom", "clothing_top_color",
                      "clothing_bottom_color", "color", "gender_estimate")
        )]
        if not has_real:
            return {"answered": False}
        matches = has_real

    lines = []
    for ev in matches[:5]:
        attrs = ev.attributes or {}
        if ev.object_class in ("car","truck","bus","motorcycle","bicycle"):
            color  = attrs.get("color", "unknown")
            vtype  = attrs.get("type", "vehicle")
            make   = attrs.get("make_estimate", "")
            plate  = attrs.get("plate_number", "")
            desc   = f"**{color} {vtype}**"
            if make and make != "unknown":
                desc += f" ({make})"
            if plate and plate != "unknown":
                desc += f", plate: {plate}"
            lines.append(
                f"Track #{ev.track_id}: {desc} — "
                f"seen {_fmt(ev.first_seen_second)}\u2013{_fmt(ev.last_seen_second)} "
                f"in `{ev.video_filename}`"
            )
        else:
            gender = attrs.get("gender_estimate", "")
            age    = attrs.get("age_estimate", "")
            top    = attrs.get("clothing_top", "")
            bottom = attrs.get("clothing_bottom", "")
            # Fall back to CV color fields if LLM clothing is unknown
            if not top or top in ("unknown", "none"):
                top = attrs.get("clothing_top_color", "")
            if not bottom or bottom in ("unknown", "none"):
                bottom = attrs.get("clothing_bottom_color", "")
            carry  = attrs.get("carrying", "")
            vis    = attrs.get("visible_text", "")
            parts  = [p for p in [gender, age, top, bottom]
                      if p and p not in ("unknown", "none", "")]
            if carry and carry not in ("unknown", "none"):
                parts.append(f"carrying {carry}")
            if vis and vis not in ("none", "unknown"):
                parts.append(f'badge/text: "{vis}"')
            if not parts:
                continue  # skip genuinely empty tracks
            desc = ", ".join(parts)
            lines.append(
                f"Track #{ev.track_id}: **{desc}** — "
                f"seen {_fmt(ev.first_seen_second)}\u2013{_fmt(ev.last_seen_second)} "
                f"in `{ev.video_filename}`"
            )

    if not lines:
        return {"answered": False}

    return {
        "answered": True,
        "answer": "\n".join(lines),
        "sources": [_source_from_event(ev) for ev in matches[:4]],
    }


def _resolve_carrying(db, q: str, video_filename: Optional[str], min_second: Optional[float] = None, max_second: Optional[float] = None) -> dict:
    """
    Answer 'is there anyone with a backpack / bag / luggage?' type questions.

    YOLO detects backpack/handbag/suitcase as nearby objects and wires them
    into TrackEvent.attributes['carrying'].  moondream also extracts carrying
    from crop images.  Neither ends up as a TrackEvent.object_class, so the
    normal _detect_class path misses them entirely.

    This resolver searches the JSONB attributes field directly.
    """
    q_lower = q.lower()

    # Detect which carrying keyword is being asked about
    carrying_type: Optional[str] = None
    for label, keywords in _CARRYING_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            carrying_type = label
            break

    if carrying_type is None:
        return {"answered": False}

    # Search TrackEvent.attributes->>'carrying' for the keyword group
    from app.storage.models import TrackEvent as _TE
    from sqlalchemy import cast, String

    q_db = db.query(_TE).filter(_TE.event_type == "entry")
    if video_filename:
        q_db = q_db.filter(_TE.video_filename == video_filename)
    if min_second is not None:
        q_db = q_db.filter(_TE.last_seen_second >= min_second)
    if max_second is not None:
        q_db = q_db.filter(_TE.first_seen_second < max_second)
    all_entry_events = q_db.order_by(_TE.first_seen_second).all()

    keywords_for_type = _CARRYING_KEYWORDS[carrying_type]
    matches = [
        ev for ev in all_entry_events
        if any(
            k in str((ev.attributes or {}).get("carrying", "")).lower()
            for k in keywords_for_type
        )
    ]

    if not matches:
        return {
            "answered": True,
            "answer": f"No — no person carrying a {carrying_type} was detected in the footage.",
            "sources": [],
        }

    lines = []
    for ev in matches[:5]:
        attrs = ev.attributes or {}
        carrying_val = attrs.get("carrying", carrying_type)
        top    = attrs.get("clothing_top", "") or attrs.get("clothing_top_color", "")
        gender = attrs.get("gender_estimate", "")
        parts  = [p for p in [gender, top] if p and p not in ("unknown", "none", "")]
        desc   = ", ".join(parts) if parts else "person"
        lines.append(
            f"Track #{ev.track_id}: **{desc}** carrying **{carrying_val}** — "
            f"seen {_fmt(ev.first_seen_second)}–{_fmt(ev.last_seen_second)} "
            f"in `{ev.video_filename}`"
        )

    answer = (
        f"Yes — **{len(matches)}** person(s) carrying a {carrying_type} detected:\n"
        + "\n".join(lines)
    )
    return {
        "answered": True,
        "answer": answer,
        "sources": [_source_from_event(ev) for ev in matches[:4]],
    }


# ── Public entry point ────────────────────────────────────────────────────────

def try_fast_path(
    db: Session,
    question: str,
    video_filename: Optional[str] = None,
    min_second: Optional[float] = None,
    max_second: Optional[float] = None,
) -> dict:
    """
    Try to answer the question directly from the database.

    min_second / max_second are Unix epoch floats that gate which TrackEvents
    are considered.  If not supplied, they are parsed from the question text
    (e.g. "Mar 13 between 10:30-10:40").

    Returns:
      {"answered": True,  "answer": "...", "sources": [...], "fast_path_type": "..."}
      {"answered": False}   ← caller should proceed to LLM
    """
    q = question.strip()

    # ── Auto-detect time range from question if not provided by caller ─────────
    _time_range = _parse_time_range(q)
    if _time_range:
        _min_e, _max_e, _win_suffixes = _time_range
        # Only override caller values if they weren't explicitly supplied
        if min_second is None:
            min_second = _min_e
        if max_second is None:
            max_second = _max_e
        # If no specific video selected, narrow to matching windows only
        if video_filename is None and _win_suffixes:
            # We'll pass this through to resolvers via the min/max epoch filter
            # (window names embed the time, so filtering by first_seen_second
            # already scopes to the right windows — no extra join needed)
            pass
    _log.info("fast_path_time_range", min=min_second, max=max_second, question=q)

    # Plate query — highest priority, very specific
    if _match(_PLATE_PATTERNS, q):
        _log.info("fast_path_triggered", type="plate", question=q)
        result = _resolve_plate(db, q, video_filename, min_second, max_second)
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
        result = _resolve_count(db, q, video_filename, min_second, max_second)
        if result["answered"]:
            result["fast_path_type"] = "count"
            return result

    # When/time query
    if _match(_WHEN_PATTERNS, q):
        _log.info("fast_path_triggered", type="when", question=q)
        result = _resolve_when(db, q, video_filename, min_second, max_second)
        if result["answered"]:
            result["fast_path_type"] = "when"
            return result

    # Identity/appearance query
    if _match(_IDENTITY_PATTERNS, q):
        _log.info("fast_path_triggered", type="identity", question=q)
        result = _resolve_identity(db, q, video_filename, min_second, max_second)
        if result["answered"]:
            result["fast_path_type"] = "identity"
            return result

    # Carrying/accessory query — backpack, bag, luggage etc.
    # Must run before presence so "anyone with backpack" hits here first
    carrying_keywords = ["backpack", "rucksack", "bag", "handbag", "purse",
                         "luggage", "suitcase", "briefcase", "carrying"]
    if any(k in q.lower() for k in carrying_keywords):
        _log.info("fast_path_triggered", type="carrying", question=q)
        result = _resolve_carrying(db, q, video_filename, min_second, max_second)
        if result["answered"]:
            result["fast_path_type"] = "carrying"
            return result

    # Presence query — simple yes/no
    if _match(_PRESENCE_PATTERNS, q):
        _log.info("fast_path_triggered", type="presence", question=q)
        result = _resolve_presence(db, q, video_filename, min_second, max_second)
        if result["answered"]:
            result["fast_path_type"] = "presence"
            return result

    _log.info("fast_path_miss", question=q)
    return {"answered": False}