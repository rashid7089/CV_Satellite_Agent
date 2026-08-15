"""Keras inference service. The trained model is loaded once at startup."""
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from app.config import settings
import logging
log = logging.getLogger(__name__)

_model = None
_labels: list[str] = []
_model_loaded = False


def _load_labels() -> list[str]:
    fallback = ["annual_crop", "forest", "residential", "river", "sea_lake"]
    path = Path(settings.labels_path)

    if not path.exists():
        return fallback

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("%s is empty or invalid JSON - using fallback labels", path)
        return fallback

    if isinstance(raw, dict):            # {"0": "forest", "1": "river"}
        return [raw[k] for k in sorted(raw, key=int)]
    if isinstance(raw, list):            # ["forest", "river"]
        return list(raw)

    log.warning("Unexpected labels.json structure - using fallback labels")
    return fallback


def load_model() -> None:
    global _model, _labels, _model_loaded
    _labels = _load_labels()
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    _model = keras.models.load_model(model_path, compile=False)
    output_classes = int(_model.output_shape[-1])
    if output_classes != len(_labels):
        raise ValueError(
            f"Model outputs {output_classes} classes but labels.json contains {len(_labels)} labels"
        )
    _model_loaded = True
    log.info("Loaded Keras model %s with input shape %s", model_path, _model.input_shape)


def is_loaded() -> bool:
    return _model_loaded


def get_labels() -> list[str]:
    return _labels or _load_labels()


def run_inference(image_bytes: bytes) -> dict:
    if _model is None or not _model_loaded:
        raise RuntimeError("Model is not loaded")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                      
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    start = time.perf_counter()

    input_height = int(_model.input_shape[1])
    input_width = int(_model.input_shape[2])
    resized = img.resize((input_width, input_height), Image.Resampling.BILINEAR)
    batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
    probabilities = np.asarray(_model.predict(batch, verbose=0)[0], dtype=float)
    ranked = sorted(zip(get_labels(), probabilities.tolist()), key=lambda item: item[1], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000

    top = [{"class_name": c, "probability": round(p, 4)} for c, p in ranked[:5]]
    return {
        "predicted_class": top[0]["class_name"],
        "confidence": top[0]["probability"],
        "top_predictions": top,
        "inference_ms": round(elapsed_ms, 3),
        "model_version": settings.model_version,
    }
