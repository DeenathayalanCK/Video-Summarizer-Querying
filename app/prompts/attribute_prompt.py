"""
Phase 6B: Attribute extraction prompts for minicpm-v.

Design principles:
- Ask for JSON output only — no prose, no preamble
- Short, specific fields — minicpm-v on CPU is ~30s/crop, we don't want essays
- Use "unknown" as the null value — easier to filter than None in RAG text
- Keep prompts tight — fewer tokens = faster inference
"""

# ── Vehicle attribute extraction ───────────────────────────────────────────────

VEHICLE_ATTRIBUTE_PROMPT = """You are analyzing a cropped image of a vehicle from a security camera.

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

JSON fields:
- "color": dominant color of the vehicle body (e.g. "white", "black", "silver", "red", "blue", "grey", "unknown")
- "type": vehicle body style (choose one: "sedan", "suv", "van", "truck", "bus", "motorcycle", "bicycle", "hatchback", "pickup", "unknown")
- "make_estimate": manufacturer if visible (e.g. "Toyota", "Ford", "unknown") — only if clearly identifiable, otherwise "unknown"
- "plate_visible": true or false — is any license plate visible?

Example output:
{"color": "white", "type": "van", "make_estimate": "unknown", "plate_visible": false}

Now analyze the vehicle in this image:"""



# ── License plate OCR ─────────────────────────────────────────────────────────

PLATE_OCR_PROMPT = """You are reading a license plate from a security camera image of a vehicle.

Your ONLY job is to read the license plate number/text exactly as it appears.

Rules:
- Look carefully at the plate area — it may be partially visible or at an angle
- Read characters left-to-right exactly as printed
- Include letters, numbers, spaces, and hyphens as they appear
- Do NOT guess or invent characters you cannot read clearly
- If the plate is too blurry, occluded, or unreadable, output: {"plate_number": "unknown"}

Respond ONLY with a valid JSON object. No explanation, no preamble.

JSON field:
- "plate_number": the exact plate text (e.g. "TN 09 AB 1234", "KA-01-MF-7890", "unknown")

Examples:
{"plate_number": "TN 09 AB 1234"}
{"plate_number": "KA 01 MF 7890"}
{"plate_number": "unknown"}

Now read the license plate in this image:"""

# ── Person attribute extraction ────────────────────────────────────────────────

PERSON_ATTRIBUTE_PROMPT = """You are analyzing a cropped image of a person from a security camera.

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

JSON fields:
- "gender_estimate": "male", "female", or "unknown" — only if clearly visible
- "clothing_top": color and type of upper body clothing (e.g. "black jacket", "white shirt", "unknown")
- "clothing_bottom": color and type of lower body clothing (e.g. "blue jeans", "dark trousers", "unknown")
- "head_covering": any hat, helmet, hood, or "none" or "unknown"
- "carrying": any visible bags, backpacks, objects or "none" or "unknown"
- "age_estimate": approximate age bracket — choose one: "child" (under 13), "teenager" (13-18), "young adult" (18-30), "adult" (30-55), "senior" (55+), or "unknown". Base on body size, posture, and hair if visible. Do NOT base on face alone.
- "visible_text": any readable text ON clothing, badges, vests, hats (e.g. "SECURITY", "POLICE", "STAFF", a name or number). Use "none" if no text is visible.

Example output:
{"gender_estimate": "male", "age_estimate": "adult", "clothing_top": "white shirt", "clothing_bottom": "dark trousers", "head_covering": "none", "carrying": "none", "visible_text": "SECURITY"}

Now analyze the person in this image:"""


# ── RAG text builders — called after attributes are extracted ──────────────────

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
    """
    Build enriched RAG text for a vehicle TrackEvent.
    Called after Phase 6B attributes are extracted.

    Format designed to match natural language queries like:
      "red van that entered"
      "white Toyota near gate"
      "vehicle present for 30 seconds"
    """
    # Build description prefix — skip "unknown" values to keep text clean
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
    """
    Build enriched RAG text for a person TrackEvent.
    Called after Phase 6B attributes are extracted.

    Format designed to match natural language queries like:
      "person in black jacket"
      "male with backpack"
      "person who entered at 30 seconds"
    """
    # Build appearance description
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