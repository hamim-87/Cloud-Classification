"""
Cloud-type segmentation and classification inference.

Model: TimmUNetWithCls (U-Net + EfficientNet-B1 encoder + classification head)
Classes: Fish, Flower, Gravel, Sugar
Trained on: Understanding Clouds from Satellite Images (Kaggle)
"""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

# ─── Constants ────────────────────────────────────────────────────

BACKBONE_NAME_MAP = {
    "resnet34": "resnet34",
    "efficientnet-b1": "efficientnet_b1",
    "resnext101_32x8d_wsl": "resnext101_32x8d",
    "resnext101_32x16d_wsl": "resnext101_32x16d",
}

CLASS_NAMES = ["Fish", "Flower", "Gravel", "Sugar"]

CLASS_COLORS = {
    "Fish": (0.255, 0.412, 0.882),
    "Flower": (1.000, 0.843, 0.000),
    "Gravel": (0.180, 0.545, 0.341),
    "Sugar": (0.863, 0.078, 0.235),
}

IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONE = "efficientnet-b1"
DEFAULT_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45


# ─── Model Architecture ──────────────────────────────────────────


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


# ─── Inference Functions ──────────────────────────────────────────


def load_model(checkpoint_path, backbone=DEFAULT_BACKBONE, device="cpu"):
    """Load a TimmUNetWithCls model from a training checkpoint."""
    device = torch.device(device) if isinstance(device, str) else device

    model = TimmUNetWithCls(
        backbone=backbone,
        num_classes=4,
        use_cls_head=True,
        pretrained=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
        f"score={ckpt.get('score', 'N/A')}"
    )
    return model


def build_inference_transform(image_size=IMAGE_SIZE):
    """Build the albumentations pipeline matching validation transforms from training."""
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def preprocess_image(image_path, transform):
    """Read an image and apply the inference transform.

    Returns:
        tensor: (1, 3, IMAGE_SIZE, IMAGE_SIZE) float32 tensor
        original_image: (H, W, 3) uint8 numpy array in RGB
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    original_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transformed = transform(image=original_image)
    tensor = transformed["image"].unsqueeze(0)

    return tensor, original_image


def postprocess_outputs(
    seg_logits,
    cls_logits,
    original_hw,
    seg_threshold=DEFAULT_THRESHOLD,
    cls_threshold=DEFAULT_THRESHOLD,
):
    """Convert raw model logits to predictions.

    Returns:
        dict with keys:
            'class_names':  list of str
            'cls_probs':    np.ndarray (4,) — sigmoid probabilities
            'cls_labels':   list of str — names of detected classes
            'seg_masks':    np.ndarray (4, orig_H, orig_W) — binary uint8
    """
    cls_probs = torch.sigmoid(cls_logits[0]).cpu().numpy()
    cls_labels = [CLASS_NAMES[i] for i in range(4) if cls_probs[i] >= cls_threshold]

    seg_probs = torch.sigmoid(seg_logits[0])
    orig_h, orig_w = original_hw
    seg_probs_resized = F.interpolate(
        seg_probs.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0]

    seg_masks = (seg_probs_resized.cpu().numpy() >= seg_threshold).astype(np.uint8)

    return {
        "class_names": CLASS_NAMES,
        "cls_probs": cls_probs,
        "cls_labels": cls_labels,
        "seg_masks": seg_masks,
    }


@torch.no_grad()
def run_inference(
    image_path,
    model,
    device,
    seg_threshold=DEFAULT_THRESHOLD,
    cls_threshold=DEFAULT_THRESHOLD,
):
    """Full inference pipeline: load image -> preprocess -> model -> postprocess.

    Returns the postprocess dict plus 'original_image'.
    """
    device = torch.device(device) if isinstance(device, str) else device

    transform = build_inference_transform()

    tensor, original_image = preprocess_image(image_path, transform)
    tensor = tensor.to(device)

    outputs = model(tensor)

    orig_hw = (original_image.shape[0], original_image.shape[1])
    results = postprocess_outputs(
        outputs["seg_logits"],
        outputs["cls_logits"],
        orig_hw,
        seg_threshold,
        cls_threshold,
    )
    results["original_image"] = original_image

    return results


# ─── Visualization ────────────────────────────────────────────────


def visualize_results(results, save_path=None, show=True):
    """Overlay segmentation masks on the original image with a legend."""
    original = results["original_image"]
    seg_masks = results["seg_masks"]
    cls_probs = results["cls_probs"]

    overlay = original.astype(np.float32) / 255.0
    legend_patches = []

    for i, name in enumerate(CLASS_NAMES):
        color = np.array(CLASS_COLORS[name])
        mask = seg_masks[i]

        if mask.sum() > 0:
            mask_3d = mask[:, :, np.newaxis]
            overlay = overlay * (1 - mask_3d * OVERLAY_ALPHA) + mask_3d * OVERLAY_ALPHA * color

    overlay = np.clip(overlay, 0, 1)

    for i, name in enumerate(CLASS_NAMES):
        prob = cls_probs[i]
        legend_patches.append(
            mpatches.Patch(color=CLASS_COLORS[name], label=f"{name}: {prob:.2%}")
        )

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title("Cloud Type Segmentation", fontsize=16, fontweight="bold")
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=12,
        framealpha=0.8,
        title="Class (confidence)",
        title_fontsize=13,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cloud-type segmentation & classification inference"
    )
    parser.add_argument("image_path", type=str, help="Path to the input cloud image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="Path to model checkpoint (default: best.pth)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        choices=list(BACKBONE_NAME_MAP.keys()),
        help=f"Encoder backbone (default: {DEFAULT_BACKBONE})",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Segmentation binarization threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--cls-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Classification confidence threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', or 'cuda:0'. Auto-detects if not set.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the visualization image (e.g., output.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the matplotlib window",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.backbone, device)

    results = run_inference(
        args.image_path,
        model,
        device,
        seg_threshold=args.seg_threshold,
        cls_threshold=args.cls_threshold,
    )

    print("\n=== Classification Results ===")
    for i, name in enumerate(CLASS_NAMES):
        prob = results["cls_probs"][i]
        detected = "  [DETECTED]" if name in results["cls_labels"] else ""
        print(f"  {name:8s}: {prob:.4f} ({prob:.1%}){detected}")

    print("\n=== Segmentation Summary ===")
    for i, name in enumerate(CLASS_NAMES):
        mask = results["seg_masks"][i]
        pixel_count = mask.sum()
        total = mask.shape[0] * mask.shape[1]
        pct = 100.0 * pixel_count / total
        print(f"  {name:8s}: {pixel_count:>8d} pixels ({pct:.2f}% of image)")

    visualize_results(results, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
