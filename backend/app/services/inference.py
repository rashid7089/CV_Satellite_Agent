"""
Inference service (TensorFlow / Keras).

The model is loaded ONCE at startup, not per request (instructor review
question #3). load_model() is called from the FastAPI lifespan handler in
app/main.py; run_inference() is called by POST /api/v1/predict.
"""

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings

log = logging.getLogger(__name__)

# Module-level state - populated once by load_model()
_model = None
_labels: list[str] = []
_model_loaded = False

# Must match the preprocessing used during training.
INPUT_SIZE = (128, 128)          # (width, height) for PIL resize

# Set to True ONLY if the Keras model does NOT already contain a
# Rescaling(1./255) layer. Applying it twice produces garbage predictions.
RESCALE_MANUALLY = False

FALLBACK_LABELS = ["cloudy", "desert", "green_area", "water"]


def _load_labels() -> list[str]:
    """
    Read models/labels.json. Accepts either {"0": "cloudy", ...} or
    ["cloudy", ...]. Falls back to a hardcoded list if the file is missing,
    empty, or malformed - a broken labels file should not take down the API.
    """
    path = Path(settings.labels_path)

    if not path.exists():
        log.warning("%s not found - using fallback labels", path)
        return FALLBACK_LABELS

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("%s is empty or invalid JSON - using fallback labels", path)
        return FALLBACK_LABELS

    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw, key=int)]
    if isinstance(raw, list):
        return list(raw)

    log.warning("Unexpected labels.json structure - using fallback labels")
    return FALLBACK_LABELS


def load_model() -> None:
    """Load the .keras artifact once at application startup."""
    global _model, _labels, _model_loaded

    _labels = _load_labels()

    path = Path(settings.model_path)
    if not path.exists():
        log.error("Model artifact not found at %s - /predict will fail", path)
        _model_loaded = False
        return

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(path)
        _model_loaded = True

        log.info(
            "Model loaded: %d classes, input %s, output shape %s",
            len(_labels), INPUT_SIZE, _model.output_shape,
        )

    except Exception:
        log.exception("Failed to load model from %s", path)
        _model = None
        _model_loaded = False


def is_loaded() -> bool:
    return _model_loaded


def get_labels() -> list[str]:
    return _labels or _load_labels()


def _preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> (1, H, W, 3) float32 batch, in Keras channel order."""
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32)

    if RESCALE_MANUALLY:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def run_inference(image_bytes: bytes) -> dict:
    """
    Classify raw image bytes.

    Returns the structured result defined by the project spec.
    Raises ValueError if the bytes are not a decodable image.
    Raises RuntimeError if the model never loaded.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                                    # catches corrupted files
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    if _model is None:
        raise RuntimeError("Model is not loaded")

    start = time.perf_counter()

    batch = _preprocess(img)
    probs = np.asarray(_model.predict(batch, verbose=0)[0], dtype=np.float64)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # If the final layer has no softmax, the outputs are raw logits.
    if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        exp = np.exp(probs - probs.max())
        probs = exp / exp.sum()

    k = min(5, len(_labels))
    top_idx = np.argsort(probs)[::-1][:k]

    top = [
        {"class_name": _labels[int(i)], "probability": round(float(probs[i]), 4)}
        for i in top_idx
    ]

    return {
        "predicted_class": top[0]["class_name"],
        "confidence": top[0]["probability"],
        "top_predictions": top,
        "inference_ms": round(elapsed_ms, 3),
        "model_version": settings.model_version,
    }