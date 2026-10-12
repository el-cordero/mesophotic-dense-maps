"""Point-feature classifier and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

REQUIRED_CHECKPOINT_KEYS = {"num_classes", "class_ids", "class_names", "x_mean", "x_std", "classifier"}


class PointFeatureClassifier(nn.Module):
    """MLP with hidden layers of 1024, 512 and 256 units, batch normalization, GELU and dropout."""

    def __init__(self, in_dim: int = 768, num_classes: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a trained classifier together with its class IDs and feature standardization."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing required keys: {', '.join(missing)}")
    num_classes = int(checkpoint["num_classes"])
    class_ids = list(checkpoint["class_ids"])
    if len(class_ids) != num_classes:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has num_classes={num_classes}, "
            f"but class_ids has length {len(class_ids)}."
        )
    model = PointFeatureClassifier(in_dim=768, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["classifier"])
    model.eval()
    return model, checkpoint
