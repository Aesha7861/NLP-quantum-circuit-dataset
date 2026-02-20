"""
Inference-only classifier for "circuit" vs "non_circuit" images.

This module keeps three things consistent across your project:
  1) the model architecture (ResNet-18 with a 2-class head)
  2) the image preprocessing pipeline
  3) a tiny API surface (load_circuit_model, predict_image)

That separation reduces "spaghetti" because the rest of the pipeline doesn't
need to know anything about PyTorch details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Default weights path used by the rest of the pipeline.
MODEL_PATH = PROJECT_ROOT / "resnet18_circuit_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same normalization as in training (ImageNet stats).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)

# Must match training's ImageFolder.classes order.
CLASS_NAMES = ["circuit", "non_circuit"]


def set_deterministic(seed: int = 0) -> None:
    """Try to make inference deterministic.

    Notes:
        - Determinism on GPU can still vary depending on driver/cuDNN kernels.
        - This function is harmless on CPU.
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some Torch versions don't support this.
        pass


def _build_model(num_classes: int = 2) -> nn.Module:
    """Create the ResNet-18 architecture used at training time."""

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_circuit_model(
    model_path: Path | str = MODEL_PATH,
    *,
    deterministic: bool = False,
) -> nn.Module:
    """Load trained weights and return an eval() model on DEVICE.

    Args:
        model_path: Path to .pth weights.
        deterministic: If True, enable deterministic settings for inference.

    Raises:
        FileNotFoundError: If the weights file is missing.
    """

    if deterministic:
        set_deterministic(0)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            "Put resnet18_circuit_classifier.pth in PROJECT_ROOT or pass model_path explicitly."
        )

    model = _build_model(num_classes=len(CLASS_NAMES))
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: nn.Module, img_path: str) -> Tuple[str, float]:
    """Predict (label, probability) for a single image file."""

    img = Image.open(img_path).convert("RGB")
    x = _EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)

    label = CLASS_NAMES[pred_idx.item()]
    return label, float(conf.item())
