"""
Phase 6B: Attribute extraction prompts.

Model strategy:
- Primary: moondream2 (1.7GB, CPU-optimized, ~15-30s per crop on CPU)
- Fallback: minicpm-v / llava (heavier, GPU recommended)

moondream2 works best with SHORT, direct questions — NOT complex JSON schemas.
We ask one simple question and parse the plain-text answer into structured fields.
This avoids JSON parse failures that are common with moondream2's output style.
"""

# ── Vehicle attribute extraction ───────────────────────────────────────────────

VEHICLE_ATTRIBUTE_PROMPT = """Analyze this security camera crop of a vehicle.
Reply with ONLY a JSON object, no other text.
{"color": "dominant body color or unknown", "type": "sedan|suv|van|truck|bus|motorcycle|bicycle|hatchback|pickup|unknown", "make_estimate": "brand if visible or unknown", "plate_visible": true or false}
Example: {"color": "white", "type": "van", "make_estimate": "unknown", "plate_visible": false}"""


# ── License plate OCR ─────────────────────────────────────────────────────────

PLATE_OCR_PROMPT = """Read the license plate in this image.
Reply with ONLY: {"plate_number": "the text or unknown"}"""


# ── Person attribute extraction ────────────────────────────────────────────────
# Designed for moondream2 and minicpm-v on CPU.
# Shorter prompt = fewer input tokens = faster inference.
# Plain-text fallback parser handles non-JSON responses from moondream2.

PERSON_ATTRIBUTE_PROMPT = """Describe this person from a security camera image.
Reply with ONLY a JSON object, no other text.
{"gender_estimate": "male|female|unknown", "age_estimate": "child|teenager|young adult|adult|senior|unknown", "clothing_top": "color and type or unknown", "clothing_bottom": "color and type or unknown", "head_covering": "hat/helmet/hood/none/unknown", "carrying": "bags/backpack/object or none", "visible_text": "text on clothing or none"}
Example: {"gender_estimate": "male", "age_estimate": "adult", "clothing_top": "black jacket", "clothing_bottom": "blue jeans", "head_covering": "none", "carrying": "backpack", "visible_text": "none"}"""


# ── Moondream2 fallback: plain-text question ──────────────────────────────────
# moondream2 often ignores JSON instructions and returns plain text.
# This prompt gets a prose answer we parse with _parse_moondream_person().

PERSON_ATTRIBUTE_PROMPT_SIMPLE = """Describe the person in this image briefly: gender, approximate age, top clothing color and type, bottom clothing color and type, any hat or head covering, anything they are carrying, any visible text on their clothing."""


# ── RAG text builders ─────────────────────────────────────────────────────────

def build_vehicle_rag_text(
    track_id: int,
    object_class: str,
    event_type: str,
    first_seen: float,
    last_seen: float,
    duration: float,
    confidence: float,
    color: str = "unknown",
    vehicle_type: str = "unknown",
    make_estimate: str = "unknown",
    plate_number: str = "unknown",
) -> str:
    parts = []
    if color and color != "unknown":
        parts.append(color)
    if vehicle_type and vehicle_type != "unknown":
        parts.append(vehicle_type)
    elif object_class:
        parts.append(object_class)

    description = " ".join(parts) if parts else object_class
    make_str  = f", possibly {make_estimate}" if make_estimate and make_estimate != "unknown" else ""
    plate_str = f" License plate: {plate_number}." if plate_number and plate_number not in ("unknown", "") else ""

    return (
        f"{description.capitalize()}{make_str} (track #{track_id}) {event_type} event: "
        f"appeared at {first_seen:.1f}s, last seen at {last_seen:.1f}s "
        f"(duration: {duration:.1f}s). Detected with {confidence:.0%} confidence.{plate_str}"
    )


def build_person_rag_text(
    track_id: int,
    event_type: str,
    first_seen: float,
    last_seen: float,
    duration: float,
    confidence: float,
    gender_estimate: str = "unknown",
    age_estimate: str = "unknown",
    clothing_top: str = "unknown",
    clothing_bottom: str = "unknown",
    head_covering: str = "unknown",
    carrying: str = "unknown",
    visible_text: str = "none",
) -> str:
    appearance_parts = []
    if gender_estimate and gender_estimate != "unknown":
        appearance_parts.append(gender_estimate)
    else:
        appearance_parts.append("person")
    if age_estimate and age_estimate != "unknown":
        appearance_parts.append(age_estimate)

    clothing_parts = []
    if clothing_top and clothing_top != "unknown":
        clothing_parts.append(f"wearing {clothing_top}")
    if clothing_bottom and clothing_bottom != "unknown":
        clothing_parts.append(clothing_bottom)
    if head_covering and head_covering not in ("unknown", "none"):
        clothing_parts.append(f"with {head_covering}")
    if carrying and carrying not in ("unknown", "none"):
        clothing_parts.append(f"carrying {carrying}")
    if visible_text and visible_text not in ("unknown", "none"):
        clothing_parts.append(f'wearing text "{visible_text}"')

    appearance = " ".join(appearance_parts)
    clothing_str = ", ".join(clothing_parts)
    if clothing_str:
        appearance = f"{appearance} ({clothing_str})"

    return (
        f"{appearance.capitalize()} (track #{track_id}) {event_type} event: "
        f"appeared at {first_seen:.1f}s, last seen at {last_seen:.1f}s "
        f"(duration: {duration:.1f}s). Detected with {confidence:.0%} confidence."
    )


def build_person_rag_text_live(
    person_label: str,
    event_type: str,
    entry_wall_time: str,
    exit_wall_time: str,
    duration: float,
    confidence: float,
    gender_estimate: str = "unknown",
    age_estimate: str = "unknown",
    clothing_top: str = "unknown",
    clothing_bottom: str = "unknown",
    head_covering: str = "unknown",
    carrying: str = "unknown",
    visible_text: str = "none",
) -> str:
    appearance_parts = []
    if gender_estimate and gender_estimate != "unknown":
        appearance_parts.append(gender_estimate)
    else:
        appearance_parts.append("person")
    if age_estimate and age_estimate != "unknown":
        appearance_parts.append(age_estimate)

    clothing_parts = []
    if clothing_top and clothing_top != "unknown":
        clothing_parts.append(f"wearing {clothing_top}")
    if clothing_bottom and clothing_bottom != "unknown":
        clothing_parts.append(clothing_bottom)
    if head_covering and head_covering not in ("unknown", "none"):
        clothing_parts.append(f"with {head_covering}")
    if carrying and carrying not in ("unknown", "none"):
        clothing_parts.append(f"carrying {carrying}")
    if visible_text and visible_text not in ("unknown", "none"):
        clothing_parts.append(f'wearing text "{visible_text}"')

    appearance = " ".join(appearance_parts)
    clothing_str = ", ".join(clothing_parts)
    if clothing_str:
        appearance = f"{appearance} ({clothing_str})"

    time_str = f"entered at {entry_wall_time}"
    if exit_wall_time:
        time_str += f", exited at {exit_wall_time}"
    if duration > 0:
        time_str += f" (duration: {duration:.0f}s)"

    return (
        f"{appearance.capitalize()} ({person_label}) {event_type} event: "
        f"{time_str}. Detected with {confidence:.0%} confidence."
    )