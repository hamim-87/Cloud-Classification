"""
Cloud Segmentation Inference Module
Loads a trained TimmUNetWithCls model and runs inference on satellite images.
Returns per-class segmentation masks, classification confidence, and coverage.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Fish", "Flower", "Gravel", "Sugar"]

BACKBONE_NAME_MAP = {
    "resnet34": "resnet34",
    "efficientnet-b1": "efficientnet_b1",
    "resnext101_32x8d_wsl": "resnext101_32x8d",
    "resnext101_32x16d_wsl": "resnext101_32x16d",
}

IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONE = "efficientnet-b1"
PIXEL_THRESHOLD = 0.5
CLS_THRESHOLD = 0.5
MIN_AREA_PIXELS = 100


# ── Model Architecture ───────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TimmUNetWithCls(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=4,
        use_cls_head=True,
        pretrained=False,
        decoder_channels=(256, 128, 64, 32),
    ):
        super().__init__()
        timm_name = BACKBONE_NAME_MAP[backbone]
        self.encoder = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_ch = self.encoder.feature_info.channels()

        self.center = ConvBlock(enc_ch[-1], decoder_channels[0])
        up_blocks = []
        in_ch = decoder_channels[0]
        for idx, skip_ch in enumerate(reversed(enc_ch[:-1])):
            out_ch = decoder_channels[min(idx + 1, len(decoder_channels) - 1)]
            up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)
        self.seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)

        self.use_cls_head = use_cls_head
        self.cls_head = nn.Linear(enc_ch[-1], num_classes) if use_cls_head else None

    def forward(self, x):
        inp_size = x.shape[-2:]
        feats = self.encoder(x)
        y = self.center(feats[-1])
        for up, skip in zip(self.up_blocks, reversed(feats[:-1])):
            y = up(y, skip)
        seg = self.seg_head(y)
        seg = F.interpolate(seg, size=inp_size, mode="bilinear", align_corners=False)
        cls = None
        if self.use_cls_head and self.cls_head is not None:
            pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
            cls = self.cls_head(pooled)
        return {"seg_logits": seg, "cls_logits": cls}


# ── Inference Wrapper ─────────────────────────────────────────────────────────

class CloudSegmentationModel:
    """Singleton wrapper around the trained TimmUNetWithCls model."""

    _instance: Optional["CloudSegmentationModel"] = None

    def __init__(self, weights_path: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.model = TimmUNetWithCls(
            backbone=DEFAULT_BACKBONE,
            num_classes=len(CLASS_NAMES),
            use_cls_head=True,
            pretrained=False,
        )

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Place your best.pth in the backend/ml_project_inference/ directory."
            )

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Model loaded from %s (epoch=%s, score=%s)",
            weights_path, ckpt.get("epoch"), ckpt.get("score", "N/A"),
        )

        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    @classmethod
    def get_instance(cls, weights_path: str | Path) -> "CloudSegmentationModel":
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    @staticmethod
    def _mask_to_base64_png(mask: np.ndarray) -> str:
        img = Image.fromarray(mask, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _preprocess(self, image_path: Path):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        original_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=original_image)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        return tensor, original_image

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        tensor, original_image = self._preprocess(image_path)
        orig_h, orig_w = original_image.shape[:2]

        outputs = self.model(tensor)
        seg_logits = outputs["seg_logits"]
        cls_logits = outputs["cls_logits"]

        # Classification confidence from cls head
        cls_probs = torch.sigmoid(cls_logits[0]).cpu().numpy()

        # Segmentation masks resized to original dimensions
        seg_probs = torch.sigmoid(seg_logits[0])
        seg_probs_resized = F.interpolate(
            seg_probs.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0].cpu().numpy()

        results = {}
        classes_detected = []

        for idx, class_name in enumerate(CLASS_NAMES):
            binary_mask = (seg_probs_resized[idx] >= PIXEL_THRESHOLD).astype(np.uint8)
            mask_full = binary_mask * 255

            positive_pixels = int(np.sum(binary_mask > 0))
            total_pixels = orig_w * orig_h
            coverage = round((positive_pixels / total_pixels) * 100, 2)
            confidence = round(float(cls_probs[idx]), 4)
            present = cls_probs[idx] >= CLS_THRESHOLD and positive_pixels >= MIN_AREA_PIXELS

            if present:
                classes_detected.append(class_name)

            results[class_name] = {
                "present": bool(present),
                "confidence": confidence,
                "coverage_percent": coverage,
                "mask_base64": self._mask_to_base64_png(mask_full),
            }

        return {
            "image_size": {"width": orig_w, "height": orig_h},
            "results": results,
            "classes_detected": classes_detected,
        }
