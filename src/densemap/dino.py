"""Frozen DINO ViT-B/8 backbone and token-grid feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from . import vision_transformer as vits

DINO_VITB8_URL = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"


def build_dino_vitb8(device: torch.device, weights_path: Path | None = None):
    """Build ViT-B/8 and load the public self-supervised DINO weights (downloaded if no local file is given)."""
    model = vits.__dict__["vit_base"](patch_size=8, num_classes=0)
    state = (
        torch.load(weights_path, map_location="cpu", weights_only=False)
        if weights_path and weights_path.exists()
        else torch.hub.load_state_dict_from_url(DINO_VITB8_URL, map_location="cpu")
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_transform(image_size: int = 1024):
    """Resize to image_size x image_size and normalize with ImageNet channel statistics."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def extract_feature_grid(image_path: Path, model, device: torch.device, image_size: int = 1024, patch_size: int = 8):
    """Return the (H/8, W/8, 768) grid of final-block patch tokens and the original image size."""
    transform = build_transform(image_size=image_size)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    x = transform(img).unsqueeze(0).to(device)
    _, _, h, w = x.shape
    h2 = h - (h % patch_size)
    w2 = w - (w % patch_size)
    x = x[:, :, :h2, :w2]
    layers = model.get_intermediate_layers(x, 1)
    feats = layers[0][0, 1:, :]
    feat_h = h2 // patch_size
    feat_w = w2 // patch_size
    feats = feats.reshape(feat_h, feat_w, -1).contiguous()
    return feats, (orig_w, orig_h)
