"""
Phase 6B: Attribute extractors for vehicle and person crops.

Uses minicpm-v (via Ollama) to extract structured attributes from
the best crop image saved for each tracked object.

Design:
- One minicpm-v call per unique track (not per frame) — ~30s each
- Returns typed dataclasses, not raw dicts — callers don't parse JSON
- All fields have "unknown" as fallback — never raises on bad LLM output
- Timeout/parse failures are logged and return empty attributes (not raised)
"""

import json
import base64
import re
import requests
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompts.attribute_prompt import VEHICLE_ATTRIBUTE_PROMPT, PERSON_ATTRIBUTE_PROMPT


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class VehicleAttributes:
    color: str = "unknown"
    vehicle_type: str = "unknown"
    make_estimate: str = "unknown"
    plate_visible: bool = False
    plate_number: str = "unknown"  # OCR result — "unknown" if unreadable

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "type": self.vehicle_type,
            "make_estimate": self.make_estimate,
            "plate_visible": self.plate_visible,
            "plate_number": self.plate_number,
        }

    @property
    def has_data(self) -> bool:
        """True if any meaningful attribute was extracted (not all unknown)."""
        return any(
            v != "unknown"
            for v in [self.color, self.vehicle_type, self.make_estimate]
        )


@dataclass
class PersonAttributes:
    gender_estimate: str = "unknown"
    age_estimate: str = "unknown"    # "child"|"teenager"|"young adult"|"adult"|"senior"
    clothing_top: str = "unknown"
    clothing_bottom: str = "unknown"
    head_covering: str = "unknown"
    carrying: str = "unknown"
    visible_text: str = "none"   # text printed on clothing/badge e.g. "SECURITY"
    # Tier-1 CV fields — always populated from color histogram, no LLM needed
    clothing_top_color: str = "unknown"
    clothing_bottom_color: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "gender_estimate": self.gender_estimate,
            "age_estimate": self.age_estimate,
            "clothing_top": self.clothing_top,
            "clothing_bottom": self.clothing_bottom,
            "head_covering": self.head_covering,
            "carrying": self.carrying,
            "visible_text": self.visible_text,
            "clothing_top_color": self.clothing_top_color,
            "clothing_bottom_color": self.clothing_bottom_color,
        }

    @property
    def has_data(self) -> bool:
        # True if ANY meaningful field extracted — includes Tier-1 CV color
        return any(
            v not in ("unknown", "none")
            for v in [self.gender_estimate, self.clothing_top,
                      self.clothing_bottom, self.clothing_top_color,
                      self.clothing_bottom_color]
        )


# ── Base extractor ─────────────────────────────────────────────────────────────

class BaseAttributeExtractor:
    """
    Shared logic for calling minicpm-v on a crop image.
    Subclasses provide the prompt and parse the response.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.base_url = self.settings.ollama_host
        self.model = self.settings.multimodal_model

    def _load_and_encode_crop(self, crop_path: str) -> Optional[str]:
        """
        Load crop from disk, resize if needed, return base64 JPEG string.
        Returns None if file can't be read.
        """
        try:
            img = cv2.imread(crop_path)
            if img is None:
                self.logger.warning("crop_image_unreadable", path=crop_path)
                return None

            h, w = img.shape[:2]
            # For attribute extraction (person/vehicle crops), 224px is enough
            # to identify clothing color and type. Larger images massively increase
            # llava inference time on CPU (576 tokens vs ~64 tokens at 224px).
            # Use ATTR_MAX_DIM instead of the global caption_max_image_dim.
            ATTR_MAX_DIM = 224
            if max(h, w) > ATTR_MAX_DIM:
                scale = ATTR_MAX_DIM / max(h, w)
                img = cv2.resize(
                    img,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode("utf-8")

        except Exception as e:
            self.logger.warning("crop_encode_failed", path=crop_path, error=str(e))
            return None

    def _call_vision_model(self, image_b64: str, prompt: str) -> Optional[str]:
        """
        Call minicpm-v with a crop image and prompt.
        Returns raw text response or None on failure.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_predict": 80,    # JSON attrs are short — 80 tokens enough
                "temperature": 0.05, # low temp = consistent structured output
                "num_ctx": 2048,     # smaller context = faster KV cache on CPU
            },
        }

        # Attribute extraction uses small crops with num_predict=150, so a 90s
        # timeout is more than enough.  The full caption_timeout_seconds (300s)
        # is too long — holding an inactive DB connection for 300 s causes the
        # PostgreSQL server to close it, and the subsequent flush() fails with
        # OperationalError, aborting all remaining tracks in the window.
        # Cap read timeout at 90s; use separate 5s connect timeout so a
        # cold/unreachable Ollama fails fast rather than blocking the whole pipeline.
        # Tier-1 CV gives color+carrying for free. LLM (tier-2) is optional enrichment
        # for gender/age/clothing type. With semaphore fix, 30s is sufficient — the
        # thread won't waste its timeout waiting for another thread's Ollama call.
        attr_timeout = 120  # 30s — if it doesn't finish in 30s, tier-1 data is enough

        # Skip entirely if no multimodal model configured
        if not self.model or not self.model.strip():
            self.logger.info("attribute_extraction_skipped", reason="no_multimodal_model")
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(5, attr_timeout),   # (connect_timeout, read_timeout)
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            self.logger.warning(
                "attribute_extraction_timeout",
                model=self.model,
                timeout=attr_timeout,
                hint="Model too slow for this hardware. Consider minicpm-v or moondream2.",
            )
            return None
        except Exception as e:
            self.logger.warning("attribute_extraction_failed", error=str(e))
            return None

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Parse JSON from model response.
        Handles cases where the model wraps JSON in markdown code blocks.
        """
        if not text:
            return None

        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } in the text
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        self.logger.warning(
            "attribute_json_parse_failed",
            raw_response=text[:200],
        )
        return None

    # ── Tier-1: OpenCV color histogram (zero LLM dependency, <5ms) ────────────

    # HSV hue/saturation/value bounds for dominant color classification.
    # Format: (color_name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
    _COLOR_RANGES = [
        ("black",  (0, 0, 0),     (180, 255, 60)),
        ("white",  (0, 0, 200),   (180, 30,  255)),
        ("red",    (0, 120, 70),  (10,  255, 255)),
        ("red",    (170, 120, 70),(180, 255, 255)),
        ("orange", (11, 100, 70), (25,  255, 255)),
        ("yellow", (26, 100, 70), (34,  255, 255)),
        ("green",  (35, 50,  50), (85,  255, 255)),
        ("blue",   (86, 50,  50), (130, 255, 255)),
        ("navy",   (86, 80,  20), (130, 255, 100)),
        ("purple", (131, 50, 50), (160, 255, 255)),
        ("pink",   (161, 50, 100),(169, 255, 255)),
        ("grey",   (0, 0,   70),  (180, 40,  200)),
        ("brown",  (10, 50, 50),  (20,  200, 150)),
    ]

    def extract_cv_colors(self, crop_path: str) -> tuple:
        """
        Tier-1: Extract dominant clothing color for top and bottom halves
        using OpenCV HSV color histograms. No LLM required. ~3ms per crop.

        Splits crop into top-third (shirt/jacket) and bottom-third (pants/skirt),
        skips middle third to reduce torso/waist overlap.

        Returns (top_color: str, bottom_color: str) — "unknown" if unreadable.
        """
        try:
            img = cv2.imread(crop_path)
            if img is None or img.shape[0] < 20 or img.shape[1] < 10:
                return "unknown", "unknown"

            h, w = img.shape[:2]
            top_region    = img[: h // 3, :]
            bottom_region = img[2 * h // 3 :, :]

            return self._dominant_color(top_region), self._dominant_color(bottom_region)

        except Exception as e:
            self.logger.debug("cv_color_extraction_failed", path=crop_path, error=str(e))
            return "unknown", "unknown"

    def _dominant_color(self, region: np.ndarray) -> str:
        """Return dominant clothing color from an image region via HSV matching."""
        if region is None or region.size == 0:
            return "unknown"
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        except Exception:
            return "unknown"

        pixels = hsv.reshape(-1, 3)
        total = len(pixels)
        if total == 0:
            return "unknown"

        color_counts: dict = {}
        for px in pixels:
            h_v, s_v, v_v = int(px[0]), int(px[1]), int(px[2])
            for color, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) in self._COLOR_RANGES:
                if (lo_h <= h_v <= hi_h and
                        lo_s <= s_v <= hi_s and
                        lo_v <= v_v <= hi_v):
                    color_counts[color] = color_counts.get(color, 0) + 1
                    break  # first matching range wins

        if not color_counts:
            return "unknown"
        best = max(color_counts, key=color_counts.get)
        # Require ≥15% pixel coverage to avoid noise
        if color_counts[best] / total < 0.15:
            return "unknown"
        return best

    def _parse_prose_person_attrs(self, text: str) -> dict:
        """
        Fallback for models (e.g. moondream2) that return prose instead of JSON.
        Extracts person attributes using keyword matching on plain-text response.
        Returns partial dict — missing keys stay "unknown" in the caller.
        """
        if not text:
            return {}
        t = text.lower()
        result = {}
        import re as _re

        if any(w in t for w in [" male", " man ", "boy", " his "]):
            result["gender_estimate"] = "male"
        elif any(w in t for w in [" female", " woman ", "girl", " her "]):
            result["gender_estimate"] = "female"

        if any(w in t for w in ["child", " kid "]):
            result["age_estimate"] = "child"
        elif "teen" in t:
            result["age_estimate"] = "teenager"
        elif "young adult" in t or "young man" in t or "young woman" in t:
            result["age_estimate"] = "young adult"
        elif any(w in t for w in ["senior", "elderly", "older person"]):
            result["age_estimate"] = "senior"
        elif "adult" in t:
            result["age_estimate"] = "adult"

        colors = ["black", "white", "red", "blue", "green", "yellow",
                  "grey", "gray", "brown", "orange", "purple", "pink",
                  "navy", "dark blue", "light blue", "dark", "light"]
        top_garments    = ["jacket", "shirt", "hoodie", "sweater", "t-shirt",
                           "coat", "blazer", "top", "blouse", "vest", "jumper"]
        bottom_garments = ["jeans", "trousers", "pants", "shorts", "skirt", "leggings"]

        for color in colors:
            for garment in top_garments:
                if color in t and garment in t:
                    result["clothing_top"] = f"{color} {garment}"
                    break
            if "clothing_top" in result:
                break

        for color in colors:
            for garment in bottom_garments:
                if color in t and garment in t:
                    result["clothing_bottom"] = f"{color} {garment}"
                    break
            if "clothing_bottom" in result:
                break

        for word in ["hat", "cap", "helmet", "hood", "beanie", "turban", "hijab"]:
            if word in t:
                result["head_covering"] = word
                break

        for word in ["backpack", "bag", "handbag", "briefcase",
                     "luggage", "suitcase", "box", "package"]:
            if word in t:
                result["carrying"] = word
                break

        return result


# ── Vehicle extractor ──────────────────────────────────────────────────────────

class VehicleAttributeExtractor(BaseAttributeExtractor):
    """
    Extracts color, type, and make estimate from a vehicle crop image.
    Returns VehicleAttributes (all fields default to "unknown" on any failure).
    """

    def extract(self, crop_path: str) -> VehicleAttributes:
        """
        Run minicpm-v on a vehicle crop and return structured attributes.
        Never raises — returns default VehicleAttributes on any failure.
        """
        self.logger.info("vehicle_attribute_extraction_start", crop=crop_path)

        image_b64 = self._load_and_encode_crop(crop_path)
        if image_b64 is None:
            return VehicleAttributes()

        raw = self._call_vision_model(image_b64, VEHICLE_ATTRIBUTE_PROMPT)
        if raw is None:
            return VehicleAttributes()

        data = self._extract_json(raw)
        if data is None:
            return VehicleAttributes()

        attrs = VehicleAttributes(
            color=str(data.get("color", "unknown")).lower().strip(),
            vehicle_type=str(data.get("type", "unknown")).lower().strip(),
            make_estimate=str(data.get("make_estimate", "unknown")).strip(),
            plate_visible=bool(data.get("plate_visible", False)),
        )

        # ── Second call: OCR plate number if plate was detected ───────────────
        if attrs.plate_visible:
            plate_num = self._extract_plate_number(image_b64, crop_path)
            attrs.plate_number = plate_num
            self.logger.info(
                "plate_ocr_done",
                crop=crop_path,
                plate_number=plate_num,
            )

        self.logger.info(
            "vehicle_attribute_extraction_done",
            crop=crop_path,
            color=attrs.color,
            type=attrs.vehicle_type,
            make=attrs.make_estimate,
            plate_visible=attrs.plate_visible,
            plate_number=attrs.plate_number,
        )
        return attrs

    def _extract_plate_number(self, image_b64: str, crop_path: str) -> str:
        """
        Dedicated second vision model call focused on reading the license plate.
        Uses a tighter prompt with explicit OCR instructions.
        Returns the plate number string, or "unknown" if unreadable.
        """
        from app.prompts.attribute_prompt import PLATE_OCR_PROMPT
        raw = self._call_vision_model(image_b64, PLATE_OCR_PROMPT)
        if not raw:
            return "unknown"

        data = self._extract_json(raw)
        if not data:
            # Model may return plain text instead of JSON — use it directly if short
            raw_stripped = raw.strip()
            if raw_stripped and len(raw_stripped) < 15 and raw_stripped.lower() != "unknown":
                return raw_stripped.upper()
            return "unknown"

        plate = str(data.get("plate_number", "unknown")).strip().upper()
        # Sanity check: plate numbers are typically 4–10 characters
        if len(plate) < 2 or len(plate) > 12:
            return "unknown"
        return plate


# ── Person extractor ───────────────────────────────────────────────────────────

class PersonAttributeExtractor(BaseAttributeExtractor):
    """
    Extracts gender estimate and clothing description from a person crop image.
    Returns PersonAttributes (all fields default to "unknown" on any failure).
    """

    def extract(self, crop_path: str) -> PersonAttributes:
        """
        Extract person attributes using a two-tier pipeline:

        Tier-1 (always, <5ms): OpenCV HSV color histogram → clothing_top_color,
          clothing_bottom_color. Always succeeds if crop is readable.

        Tier-2 (optional, up to 30s): Ollama LLM → gender, age, clothing type,
          carrying, visible_text. If it times out, Tier-1 data is still returned.
          Only runs if a multimodal model is configured.

        Never raises — returns default PersonAttributes on any failure.
        """
        self.logger.info("person_attribute_extraction_start", crop=crop_path)

        # ── Tier-1: CV color extraction (always, <5ms) ───────────────────────
        top_color, bottom_color = self.extract_cv_colors(crop_path)
        self.logger.info(
            "person_cv_colors_done",
            crop=crop_path,
            top_color=top_color,
            bottom_color=bottom_color,
        )

        # ── Tier-2: LLM extraction (optional) ───────────────────────────────
        llm_attrs: dict = {}
        if self.model and self.model.strip():
            image_b64 = self._load_and_encode_crop(crop_path)
            if image_b64 is not None:
                raw = self._call_vision_model(image_b64, PERSON_ATTRIBUTE_PROMPT)
                if raw is not None:
                    data = self._extract_json(raw)
                    if data is None:
                        # JSON failed — try prose fallback (moondream2 style)
                        data = self._parse_prose_person_attrs(raw)
                        if data:
                            self.logger.info(
                                "person_attribute_prose_fallback",
                                crop=crop_path,
                                fields_found=list(data.keys()),
                            )
                    if data:
                        llm_attrs = data

        # ── Merge: Tier-1 color fills in where LLM didn't extract ────────────
        # If LLM got clothing_top (e.g. "black jacket"), use that — it has type info.
        # If LLM got nothing or timed out, fall back to CV color-only.
        clothing_top    = str(llm_attrs.get("clothing_top", "unknown")).lower().strip()
        clothing_bottom = str(llm_attrs.get("clothing_bottom", "unknown")).lower().strip()

        # If LLM clothing field is unknown but CV got a color, use the CV color
        if (clothing_top == "unknown" or not clothing_top) and top_color != "unknown":
            clothing_top = top_color  # just the color — no type suffix
        if (clothing_bottom == "unknown" or not clothing_bottom) and bottom_color != "unknown":
            clothing_bottom = bottom_color

        attrs = PersonAttributes(
            gender_estimate=str(llm_attrs.get("gender_estimate", "unknown")).lower().strip(),
            age_estimate=str(llm_attrs.get("age_estimate", "unknown")).lower().strip(),
            clothing_top=clothing_top,
            clothing_bottom=clothing_bottom,
            head_covering=str(llm_attrs.get("head_covering", "unknown")).lower().strip(),
            carrying=str(llm_attrs.get("carrying", "unknown")).lower().strip(),
            visible_text=str(llm_attrs.get("visible_text", "none")).strip(),
            clothing_top_color=top_color,
            clothing_bottom_color=bottom_color,
        )

        self.logger.info(
            "person_attribute_extraction_done",
            crop=crop_path,
            gender=attrs.gender_estimate,
            top=attrs.clothing_top,
            bottom=attrs.clothing_bottom,
            top_cv=top_color,
            bottom_cv=bottom_color,
        )
        return attrs