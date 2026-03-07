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

    def to_dict(self) -> dict:
        return {
            "gender_estimate": self.gender_estimate,
            "age_estimate": self.age_estimate,
            "clothing_top": self.clothing_top,
            "clothing_bottom": self.clothing_bottom,
            "head_covering": self.head_covering,
            "carrying": self.carrying,
            "visible_text": self.visible_text,
        }

    @property
    def has_data(self) -> bool:
        return any(
            v not in ("unknown", "none")
            for v in [self.gender_estimate, self.clothing_top, self.clothing_bottom]
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
            max_dim = self.settings.caption_max_image_dim  # 768 default
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
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
                "num_predict": 150,   # attributes are short — cap tokens
                "temperature": 0.05,  # low temp = consistent structured output
            },
        }

        # Attribute extraction uses small crops with num_predict=150, so a 90s
        # timeout is more than enough.  The full caption_timeout_seconds (300s)
        # is too long — holding an inactive DB connection for 300 s causes the
        # PostgreSQL server to close it, and the subsequent flush() fails with
        # OperationalError, aborting all remaining tracks in the window.
        attr_timeout = min(self.settings.caption_timeout_seconds, 90)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=attr_timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            self.logger.warning(
                "attribute_extraction_timeout",
                model=self.model,
                timeout=attr_timeout,
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
        Run minicpm-v on a person crop and return structured attributes.
        Never raises — returns default PersonAttributes on any failure.
        """
        self.logger.info("person_attribute_extraction_start", crop=crop_path)

        image_b64 = self._load_and_encode_crop(crop_path)
        if image_b64 is None:
            return PersonAttributes()

        raw = self._call_vision_model(image_b64, PERSON_ATTRIBUTE_PROMPT)
        if raw is None:
            return PersonAttributes()

        data = self._extract_json(raw)
        if data is None:
            return PersonAttributes()

        attrs = PersonAttributes(
            gender_estimate=str(data.get("gender_estimate", "unknown")).lower().strip(),
            age_estimate=str(data.get("age_estimate", "unknown")).lower().strip(),
            clothing_top=str(data.get("clothing_top", "unknown")).lower().strip(),
            clothing_bottom=str(data.get("clothing_bottom", "unknown")).lower().strip(),
            head_covering=str(data.get("head_covering", "unknown")).lower().strip(),
            carrying=str(data.get("carrying", "unknown")).lower().strip(),
            # visible_text keeps its original case — "SECURITY" not "security"
            visible_text=str(data.get("visible_text", "none")).strip(),
        )

        self.logger.info(
            "person_attribute_extraction_done",
            crop=crop_path,
            gender=attrs.gender_estimate,
            top=attrs.clothing_top,
            bottom=attrs.clothing_bottom,
        )
        return attrs