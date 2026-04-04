"""
Inference script for Understanding Clouds from Satellite Images.
Produces both classification labels and segmentation masks.

Usage:
    # Single image inference
    python inference.py --image path/to/image.jpg --model-dir ./output

    # Batch inference on a directory
    python inference.py --image-dir path/to/images/ --model-dir ./output --output-dir ./results

    # With TTA (test-time augmentation)
    python inference.py --image path/to/image.jpg --model-dir ./output --tta 3
"""

import logging
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


# ===== Configuration =====

CLASS_NAMES = ["Fish", "Flower", "Gravel", "Sugar"]
NUM_CLASSES = 4
IMG_H = 384
IMG_W = 576
ORIG_H = 1400
ORIG_W = 2100
PIXEL_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]
AREA_THRESHOLDS = [0, 0, 0, 0]


# ===== RLE Utilities =====

def mask2rle(mask: np.ndarray) -> str:
    """Convert binary mask to run-length encoding."""
    pixels = mask.T.flatten()
    if pixels.sum() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(rle: str, height: int = 1400, width: int = 2100) -> np.ndarray:
    """Convert run-length encoding to binary mask."""
    mask = np.zeros(height * width, dtype=np.float32)
    if rle and rle != "":
        s = rle.split()
        starts = np.array(s[0::2], dtype=int) - 1
        lengths = np.array(s[1::2], dtype=int)
        for start, length in zip(starts, lengths):
            mask[start : start + length] = 1
    mask = mask.reshape(width, height).T
    return mask


# ===== Model Definitions =====

class SegmentationModel(nn.Module):
    """Segmentation model with auxiliary classification head."""

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        framework: str = "FPN",
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if framework == "Unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                classes=num_classes,
                activation=None,
            )
        elif framework == "FPN":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=None,
                classes=num_classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

        encoder_channels = self.model.encoder.out_channels
        out_planes = encoder_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(out_planes, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the model's own forward for segmentation
        seg_out = self.model(x)
        # Get encoder features for classification head
        features = self.model.encoder(x)
        cls_out = self.cls_head(features[-1])
        return seg_out, cls_out


class ClassificationModel(nn.Module):
    """Standalone classification model using EfficientNet."""

    def __init__(
        self,
        encoder_name: str = "efficientnet-b1",
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        import timm

        timm_name = encoder_name.replace("-", "_")
        self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ===== Transforms =====

def get_inference_transform():
    return albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )


# ===== Inference Engine =====

class CloudInferenceEngine:
    """
    Inference engine that loads trained models and produces both
    classification labels and segmentation masks.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        seg_threshold: Optional[List[float]] = None,
        cls_threshold: float = 0.5,
        area_thresholds: Optional[List[int]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.transform = get_inference_transform()
        self.seg_threshold = seg_threshold or PIXEL_THRESHOLDS
        self.cls_threshold = cls_threshold
        self.area_thresholds = area_thresholds or AREA_THRESHOLDS

        # Load config
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.seg_encoder = self.config.get("SEG_ENCODER", "efficientnet-b4")
            self.seg_framework = self.config.get("SEG_FRAMEWORK", "FPN")
            self.cls_encoder = self.config.get("CLS_ENCODER", "efficientnet-b1")
            self.train_folds = self.config.get("TRAIN_FOLDS", [0, 1])
        else:
            self.seg_encoder = "efficientnet-b4"
            self.seg_framework = "FPN"
            self.cls_encoder = "efficientnet-b1"
            self.train_folds = [0, 1]

        self.seg_models = self._load_seg_models()
        self.cls_models = self._load_cls_models()

        print(f"Loaded {len(self.seg_models)} seg models and {len(self.cls_models)} cls models on {self.device}")

    def _load_seg_models(self) -> List[nn.Module]:
        """Load all available segmentation model checkpoints."""
        models = []
        seg_dir = self.model_dir / f"seg_{self.seg_encoder}_{self.seg_framework}"
        if not seg_dir.exists():
            # Try to find any seg directory
            seg_dirs = list(self.model_dir.glob("seg_*"))
            if seg_dirs:
                seg_dir = seg_dirs[0]

        for fold in self.train_folds:
            ckpt_path = seg_dir / f"best-seg-{fold}.pt"
            if not ckpt_path.exists():
                ckpt_path = seg_dir / f"last-seg-{fold}.pt"
            if ckpt_path.exists():
                model = SegmentationModel(
                    encoder_name=self.seg_encoder,
                    framework=self.seg_framework,
                )
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"  Loaded seg model: {ckpt_path.name}")

        return models

    def _load_cls_models(self) -> List[nn.Module]:
        """Load all available classification model checkpoints."""
        models = []
        cls_dir = self.model_dir / f"cls_{self.cls_encoder}"
        if not cls_dir.exists():
            cls_dirs = list(self.model_dir.glob("cls_*"))
            if cls_dirs:
                cls_dir = cls_dirs[0]

        for fold in self.train_folds:
            ckpt_path = cls_dir / f"best-cls-{fold}.pt"
            if not ckpt_path.exists():
                ckpt_path = cls_dir / f"last-cls-{fold}.pt"
            if ckpt_path.exists():
                model = ClassificationModel(encoder_name=self.cls_encoder)
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"  Loaded cls model: {ckpt_path.name}")

        return models

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image (BGR or RGB numpy array)."""
        if image.shape[2] == 4:  # BGRA -> BGR
            image = image[:, :, :3]

        # Ensure RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (IMG_W, IMG_H))

        # Apply transforms
        augmented = self.transform(image=image_resized)
        tensor = augmented["image"].unsqueeze(0)  # (1, 3, H, W)
        return tensor

    @torch.no_grad()
    def _predict_seg_single(
        self, model: nn.Module, tensor: torch.Tensor, tta: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get segmentation and classification predictions from one model."""
        tensor = tensor.to(self.device)

        with autocast():
            seg_out, cls_out = model(tensor)

        B, C, H, W = seg_out.shape
        mask_pred = torch.sigmoid(seg_out).cpu().numpy().reshape(-1, H, W)
        cls_pred = torch.sigmoid(cls_out).cpu().numpy().reshape(-1)

        if tta >= 2:
            # Horizontal flip
            with autocast():
                seg_flip, cls_flip = model(tensor.flip(3))
            seg_flip = seg_flip.flip(3)
            mask_pred += torch.sigmoid(seg_flip).cpu().numpy().reshape(-1, H, W)
            cls_pred += torch.sigmoid(cls_flip).cpu().numpy().reshape(-1)

        if tta >= 3:
            # Vertical flip
            with autocast():
                seg_flip, cls_flip = model(tensor.flip(2))
            seg_flip = seg_flip.flip(2)
            mask_pred += torch.sigmoid(seg_flip).cpu().numpy().reshape(-1, H, W)
            cls_pred += torch.sigmoid(cls_flip).cpu().numpy().reshape(-1)

        if tta > 1:
            mask_pred /= tta
            cls_pred /= tta

        return mask_pred, cls_pred

    @torch.no_grad()
    def _predict_cls_single(
        self, model: nn.Module, tensor: torch.Tensor, tta: int = 1
    ) -> np.ndarray:
        """Get classification predictions from one model."""
        tensor = tensor.to(self.device)

        with autocast():
            cls_out = model(tensor)
        cls_pred = torch.sigmoid(cls_out).cpu().numpy().reshape(-1)

        if tta >= 2:
            with autocast():
                cls_flip = model(tensor.flip(3))
            cls_pred += torch.sigmoid(cls_flip).cpu().numpy().reshape(-1)

        if tta >= 3:
            with autocast():
                cls_flip = model(tensor.flip(2))
            cls_pred += torch.sigmoid(cls_flip).cpu().numpy().reshape(-1)

        if tta > 1:
            cls_pred /= tta

        return cls_pred

    def predict(
        self,
        image: np.ndarray,
        tta: int = 1,
        return_original_size: bool = True,
    ) -> Dict:
        """
        Run full inference pipeline on a single image.

        Args:
            image: Input image as numpy array (BGR, any size).
            tta: Test-time augmentation level (1=none, 2=hflip, 3=hflip+vflip).
            return_original_size: If True, resize masks back to original image size.

        Returns:
            Dictionary containing:
                - 'class_names': List of detected class names
                - 'class_probs': Dict mapping class name -> probability
                - 'class_labels': Binary array [Fish, Flower, Gravel, Sugar]
                - 'segmentation_masks': Dict mapping class name -> binary mask (H, W)
                - 'segmentation_probs': Dict mapping class name -> probability mask (H, W)
                - 'rles': Dict mapping class name -> run-length encoding string
        """
        orig_h, orig_w = image.shape[:2]
        tensor = self.preprocess(image)

        # --- Segmentation predictions (averaged over folds) ---
        mask_pred = np.zeros((NUM_CLASSES, IMG_H, IMG_W), dtype=np.float32)
        seg_cls_pred = np.zeros(NUM_CLASSES, dtype=np.float32)

        if self.seg_models:
            for model in self.seg_models:
                m, c = self._predict_seg_single(model, tensor, tta=tta)
                mask_pred += m
                seg_cls_pred += c
            mask_pred /= len(self.seg_models)
            seg_cls_pred /= len(self.seg_models)

        # --- Classification predictions (averaged over folds) ---
        cls_pred = np.zeros(NUM_CLASSES, dtype=np.float32)

        if self.cls_models:
            for model in self.cls_models:
                c = self._predict_cls_single(model, tensor, tta=tta)
                cls_pred += c
            cls_pred /= len(self.cls_models)

        # Combine classification from both heads (average)
        if self.seg_models and self.cls_models:
            final_cls_pred = (seg_cls_pred + cls_pred) / 2
        elif self.cls_models:
            final_cls_pred = cls_pred
        else:
            final_cls_pred = seg_cls_pred

        # --- Build output ---
        class_labels = (final_cls_pred > self.cls_threshold).astype(int)
        class_probs = {}
        class_names_detected = []
        seg_masks = {}
        seg_probs = {}
        rles = {}

        for c in range(NUM_CLASSES):
            name = CLASS_NAMES[c]
            class_probs[name] = float(final_cls_pred[c])

            if class_labels[c]:
                class_names_detected.append(name)

            # Apply pixel threshold and area filtering
            binary_mask = (mask_pred[c] > self.seg_threshold[c]).astype(np.uint8)

            # Apply classification filter: if class not detected, zero out mask
            if not class_labels[c]:
                binary_mask = np.zeros_like(binary_mask)

            # Remove small connected components
            if self.area_thresholds[c] > 0 and binary_mask.sum() > 0:
                num_comp, comp = cv2.connectedComponents(binary_mask)
                filtered = np.zeros_like(binary_mask)
                for comp_id in range(1, num_comp):
                    region = (comp == comp_id)
                    if region.sum() >= self.area_thresholds[c]:
                        filtered[region] = 1
                binary_mask = filtered

            if return_original_size:
                prob_full = cv2.resize(
                    mask_pred[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )
                mask_full = (
                    cv2.resize(
                        binary_mask.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0.5
                ).astype(np.uint8)
            else:
                prob_full = mask_pred[c]
                mask_full = binary_mask

            seg_masks[name] = mask_full
            seg_probs[name] = prob_full

            # RLE at 350x525 for competition submission format
            mask_sub = (
                cv2.resize(binary_mask.astype(np.float32), (525, 350)) > 0.5
            ).astype(np.uint8)
            rles[name] = mask2rle(mask_sub)

        return {
            "class_names": class_names_detected,
            "class_probs": class_probs,
            "class_labels": class_labels,
            "segmentation_masks": seg_masks,
            "segmentation_probs": seg_probs,
            "rles": rles,
        }

    def predict_file(
        self, image_path: str, tta: int = 1, return_original_size: bool = True
    ) -> Dict:
        """Run inference on an image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict(image, tta=tta, return_original_size=return_original_size)


# ===== Visualization =====

def visualize_result(image: np.ndarray, result: Dict, save_path: Optional[str] = None):
    """Visualize inference results with overlays."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Original image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for c, name in enumerate(CLASS_NAMES):
        mask = result["segmentation_masks"][name]
        prob = result["class_probs"][name]
        detected = name in result["class_names"]

        overlay = img_rgb.copy()
        if mask.sum() > 0:
            color_mask = np.zeros_like(overlay)
            for ch in range(3):
                color_mask[:, :, ch] = mask * colors[c][ch]
            overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

        axes[c + 1].imshow(overlay)
        status = "DETECTED" if detected else "absent"
        axes[c + 1].set_title(f"{name}: {prob:.3f} ({status})")
        axes[c + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Visualization saved: {save_path}")

    plt.show()
    plt.close()


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(
        description="Cloud Segmentation & Classification Inference"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image"
    )
    parser.add_argument(
        "--image-dir", type=str, default=None, help="Directory of images for batch inference"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument("--tta", type=int, default=1, help="TTA level (1, 2, or 3)")
    parser.add_argument(
        "--cls-threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualizations"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize engine
    engine = CloudInferenceEngine(
        model_dir=args.model_dir,
        device=args.device,
        cls_threshold=args.cls_threshold,
    )

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        image_paths = sorted(
            str(p)
            for p in img_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        )
    else:
        parser.error("Provide either --image or --image-dir")

    print(f"\nProcessing {len(image_paths)} image(s)...")

    all_results = []

    for img_path in image_paths:
        img_name = Path(img_path).stem
        print(f"\n--- {img_name} ---")

        result = engine.predict_file(img_path, tta=args.tta)

        # Print classification results
        print(f"  Detected classes: {result['class_names']}")
        for name, prob in result["class_probs"].items():
            print(f"    {name}: {prob:.4f}")

        # Print mask statistics
        for name, mask in result["segmentation_masks"].items():
            if mask.sum() > 0:
                coverage = mask.sum() / mask.size * 100
                print(f"    {name} mask: {coverage:.1f}% coverage")

        # Save individual results
        result_entry = {
            "image": img_name,
            "class_probs": result["class_probs"],
            "class_names": result["class_names"],
            "rles": result["rles"],
        }
        all_results.append(result_entry)

        # Save masks
        for name, mask in result["segmentation_masks"].items():
            mask_path = output_dir / f"{img_name}_{name}_mask.png"
            cv2.imwrite(str(mask_path), mask * 255)

        # Visualize
        if args.visualize:
            image = cv2.imread(img_path)
            vis_path = str(output_dir / f"{img_name}_visualization.png")
            visualize_result(image, result, save_path=vis_path)

    # Save summary CSV
    if all_results:
        rows = []
        for entry in all_results:
            for name in CLASS_NAMES:
                rows.append(
                    {
                        "Image_Label": f"{entry['image']}_{name}",
                        "EncodedPixels": entry["rles"].get(name, ""),
                        "ClassProb": entry["class_probs"].get(name, 0),
                    }
                )

        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = output_dir / "submission.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSubmission CSV saved: {csv_path}")

    print(f"\nAll results saved to: {output_dir}")



# ===== Backend-compatible wrapper (used by main.py) =====

import base64
import io
from PIL import Image as PILImage


class CloudSegmentationModel:
    """
    Wrapper that provides the same interface as the original inference.py
    CloudSegmentationModel, so main.py can do:
        from inference2 import CloudSegmentationModel
        model = CloudSegmentationModel.get_instance(weights_path)
        result = model.predict(image_path)
    Internally uses UNet++ (matching the supplied .pth weights).
    """

    _instance: Optional["CloudSegmentationModel"] = None

    PIXEL_THRESHOLD = 0.5
    MIN_AREA_PIXELS = 100

    def __init__(self, weights_path: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger = logging.getLogger(__name__)
        logger.info("Using device: %s", self.device)

        # Build UNet++ (architecture matching the supplied weight file)
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Place your best_model.pth in the backend/model/ directory."
            )

        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        # Strip "model." prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("model.")] = v
        self.model.load_state_dict(cleaned, strict=False)

        self.model.to(self.device)
        self.model.eval()
        self.transform = get_inference_transform()
        logger.info("Model loaded successfully from %s", weights_path)

    @classmethod
    def get_instance(cls, weights_path: str | Path) -> "CloudSegmentationModel":
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _mask_to_base64_png(mask: np.ndarray) -> str:
        img = PILImage.fromarray(mask, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── public predict (matches main.py expectations) ────────────────────
    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess with albumentations
        resized = cv2.resize(image_rgb, (IMG_W, IMG_H))
        augmented = self.transform(image=resized)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)                         # (1, 4, H, W)
        probs = torch.sigmoid(logits).squeeze(0)            # (4, H, W)
        probs_np = probs.cpu().numpy()

        results = {}
        classes_detected = []

        for idx, class_name in enumerate(CLASS_NAMES):
            prob_map = probs_np[idx]

            # Binary mask at model resolution
            binary = (prob_map >= self.PIXEL_THRESHOLD).astype(np.uint8)

            # Resize mask back to original image size
            mask_full = cv2.resize(
                binary * 255, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            positive_pixels = int(np.sum(mask_full > 0))
            total_pixels = orig_w * orig_h
            coverage = round((positive_pixels / total_pixels) * 100, 2)
            confidence = round(float(prob_map.mean()), 4)
            present = positive_pixels >= self.MIN_AREA_PIXELS

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


if __name__ == "__main__":
    main()
