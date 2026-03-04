"""
appearance_reid.py — CPU-only appearance embeddings for track Re-Identification.

Bug 3 fix: ByteTrack re-assigns a new track_id every time it loses a person
(occlusion, fast motion, sitting still for a long time).  The original merge
logic matched any two same-class tracks within a 30s window — so two DIFFERENT
people standing in the same area could be merged into one identity.

This module extracts a 512-dim appearance vector from each track's best crop
using ResNet18 (torchvision — already in requirements.txt, no new deps).
The EventGenerator then uses cosine similarity to gate merge decisions.

Architecture:
  - Model    : ResNet18 pretrained on ImageNet, fc layer → Identity()
  - Input    : 128×256 px crop (standard person ReID aspect ratio)
  - Output   : 512-dim L2-normalised float32 vector
  - Speed    : ~40–60ms per crop on CPU
  - Accuracy : Not a purpose-built ReID model.  Cosine similarity > 0.75
               reliably indicates same-clothing same-person in the same scene.
               For cross-camera ReID (Phase 8) use OSNet/torchreid.

Fails gracefully: if torch import fails or any crop is missing/corrupt,
embed_tracks() returns an empty dict and EventGenerator falls back to the
original time-gap-only merge.
"""

import os
from typing import Optional

import numpy as np

from app.core.logging import get_logger


class AppearanceReID:
    """
    Extracts 512-dim L2-normalised appearance vectors from crop images.
    One instance is created per VideoIntelligenceProcessor and reused.
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    _INPUT_HW = (256, 128)   # (H, W) — tall portrait for people

    def __init__(self):
        self.logger = get_logger()
        self._model = None
        self._transform = None
        self._available = None   # None = not yet tried; True/False after first attempt

    # ── Lazy model load ───────────────────────────────────────────────────────

    def _load(self) -> bool:
        """Load model once.  Returns True if ready, False if unavailable."""
        if self._available is not None:
            return self._available

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            self.logger.info("reid_model_loading", model="resnet18_imagenet_cpu")
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            m.fc = torch.nn.Identity()
            m.eval()
            self._model = m

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self._INPUT_HW),
                T.ToTensor(),
                T.Normalize(mean=self._MEAN, std=self._STD),
            ])
            self._available = True
            self.logger.info("reid_model_ready", dim=512)
        except Exception as e:
            self.logger.warning(
                "reid_model_unavailable", error=str(e),
                note="ReID disabled — falling back to time-gap merge",
            )
            self._available = False

        return self._available

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_crop(self, crop_path: str) -> Optional[np.ndarray]:
        """Return 512-dim L2-normed vector for one crop image, or None on failure."""
        if not self._load():
            return None
        try:
            import cv2, torch
            img = cv2.imread(crop_path)
            if img is None or img.size == 0:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self._transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                feat = self._model(tensor).squeeze().numpy().astype(np.float32)
            n = np.linalg.norm(feat)
            return feat / n if n > 0 else feat
        except Exception as e:
            self.logger.debug("reid_embed_failed", path=crop_path, error=str(e))
            return None

    def embed_tracks(
        self,
        track_states: dict,          # {track_id: TrackState}
        classes: set = None,         # only embed these classes; default {"person"}
    ) -> dict:
        """
        Embed every track that has a best_crop_path.
        Returns {track_id: np.ndarray(512,)} for all successfully embedded tracks.
        Vehicles are skipped by default — time-gap merge is reliable for them.
        """
        if classes is None:
            classes = {"person"}

        if not self._load():
            return {}

        results = {}
        for tid, state in track_states.items():
            if state.object_class not in classes:
                continue
            if not state.best_crop_path or not os.path.exists(state.best_crop_path):
                continue
            vec = self.embed_crop(state.best_crop_path)
            if vec is not None:
                results[tid] = vec

        self.logger.info(
            "reid_embeddings_done",
            requested=sum(1 for s in track_states.values()
                          if s.object_class in classes and s.best_crop_path),
            embedded=len(results),
        )
        return results