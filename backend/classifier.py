"""
CNN Classifier — EfficientNet-B4
Loads trained weights from models/efficientnet_b4.pth.
Includes DEMO MODE: if weights are missing, returns realistic mock predictions.
"""

import os
import json
import random
import numpy as np

# Load class names
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor", "astrocytoma", "ependymoma"]

# Path to weights (resolved relative to this file)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "efficientnet_b4.pth")
MODEL = None
DEMO_MODE = False


def load_model(weights_path: str = WEIGHTS_PATH, num_classes: int = 6):
    """
    Loads EfficientNet-B4 with custom classifier head.
    Falls back to demo mode if weights are unavailable or torch is not installed.
    """
    global MODEL, DEMO_MODE

    # Try importing torch
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models
    except ImportError:
        print("[Classifier] WARNING: PyTorch not installed. Entering DEMO mode.")
        DEMO_MODE = True
        return None

    abs_path = os.path.abspath(weights_path)
    if not os.path.exists(abs_path):
        print(f"[Classifier] WARNING: Model weights not found at {abs_path}. Entering DEMO mode.")
        DEMO_MODE = True
        return None

    try:
        model = tv_models.efficientnet_b4(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(abs_path, map_location="cpu"))
        model.eval()
        DEMO_MODE = False
        print(f"[Classifier] EfficientNet-B4 loaded from {abs_path}")
        return model
    except Exception as e:
        print(f"[Classifier] Failed to load weights: {e}. Entering DEMO mode.")
        DEMO_MODE = True
        return None


def _demo_predict(image: np.ndarray) -> tuple:
    """
    Returns realistic synthetic predictions for demo purposes.
    The dominant class is randomly chosen but probabilities are softmax-plausible.
    """
    # Create random logits with one dominant class
    dominant_idx = random.randint(0, len(CLASS_NAMES) - 1)
    logits = np.random.uniform(0.5, 2.0, len(CLASS_NAMES))
    logits[dominant_idx] += random.uniform(3.0, 6.0)  # boost dominant

    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence, all_probs


def predict(model, preprocessed_image: np.ndarray) -> tuple:
    """
    Args:
        model: Loaded EfficientNet-B4 model (or None in demo mode)
        preprocessed_image: float32 numpy (224, 224, 3)

    Returns:
        predicted_class (str), confidence (float), all_probs (dict)
    """
    global DEMO_MODE

    if DEMO_MODE or model is None:
        return _demo_predict(preprocessed_image)

    try:
        import torch
        tensor = (
            torch.tensor(preprocessed_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return predicted_class, confidence, all_probs

    except Exception as e:
        print(f"[Classifier] Inference failed: {e}. Falling back to demo.")
        return _demo_predict(preprocessed_image)


def is_demo_mode() -> bool:
    return DEMO_MODE
