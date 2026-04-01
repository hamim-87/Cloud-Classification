"""
Cloud Segmentation Inference Module
Loads a trained UNet++ model and runs inference on satellite images.
Returns per-class segmentation masks and classification results.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

# ── Class labels (index order matches training) ──────────────────────────────
CLASS_NAMES = ["Fish", "Flower", "Gravel", "Sugar"]

# ── Preprocessing constants (must match training) ────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_HEIGHT = 320
INPUT_WIDTH = 480

# ── Detection thresholds ─────────────────────────────────────────────────────
PIXEL_THRESHOLD = 0.5        # probability above which a pixel is "positive"
MIN_AREA_PIXELS = 100        # minimum positive pixels to consider a class present


class CloudSegmentationModel:
    """Singleton wrapper around the trained UNet++ model."""

    _instance: Optional["CloudSegmentationModel"] = None

    def __init__(self, weights_path: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Build the same architecture used during training
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b4",
            encoder_weights=None,          # we load our own weights
            in_channels=3,
            classes=len(CLASS_NAMES),
        )

        # Load trained weights
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Place your best_model.pth in the backend/model/ directory."
            )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Strip "model." prefix if present (common when saved from a training wrapper)
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.removeprefix("model.")
            cleaned[new_key] = v
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully from %s", weights_path)

    @classmethod
    def get_instance(cls, weights_path: str | Path) -> "CloudSegmentationModel":
        """Return (or create) the singleton model instance."""
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    # ── Preprocessing ────────────────────────────────────────────────────────
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Resize, normalise, and convert a PIL image to a model-ready tensor."""
        image = image.convert("RGB")
        image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)

        arr = np.array(image, dtype=np.float32) / 255.0          # (H, W, 3)

        # ImageNet normalisation
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

        # HWC → CHW → batch dim
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    # ── Mask encoding ────────────────────────────────────────────────────────
    @staticmethod
    def _mask_to_base64_png(mask: np.ndarray) -> str:
        """Convert a 2-D uint8 mask (0/255) to a base64-encoded PNG string."""
        img = Image.fromarray(mask, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── Public inference method ──────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """
        Run inference on a single image.

        Returns
        -------
        dict  with keys:
            image_size   : {width, height} of the *original* image
            results      : per-class dict with mask_base64, confidence, coverage, present
            classes_detected : list of class names that are present
        """
        image_path = Path(image_path)
        original_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = original_image.size

        # Preprocess & run model
        tensor = self._preprocess(original_image)
        logits = self.model(tensor)                      # (1, 4, H, W)
        probs = torch.sigmoid(logits).squeeze(0)         # (4, H, W)
        probs_np = probs.cpu().numpy()                   # (4, H, W)

        results = {}
        classes_detected = []

        for idx, class_name in enumerate(CLASS_NAMES):
            prob_map = probs_np[idx]                      # (H, W) float32 0-1

            # Binary mask at model resolution
            binary_mask = (prob_map >= PIXEL_THRESHOLD).astype(np.uint8)

            # Resize mask back to original image dimensions
            mask_pil = Image.fromarray(binary_mask * 255, mode="L")
            mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
            mask_full = np.array(mask_pil)

            # Statistics
            positive_pixels = int(np.sum(mask_full > 0))
            total_pixels = orig_w * orig_h
            coverage = round((positive_pixels / total_pixels) * 100, 2)
            confidence = round(float(prob_map.mean()), 4)
            present = positive_pixels >= MIN_AREA_PIXELS

            if present:
                classes_detected.append(class_name)

            results[class_name] = {
                "present": present,
                "confidence": confidence,
                "coverage_percent": coverage,
                "mask_base64": self._mask_to_base64_png(mask_full),
            }

        return {
            "image_size": {"width": orig_w, "height": orig_h},
            "results": results,
            "classes_detected": classes_detected,
        }
