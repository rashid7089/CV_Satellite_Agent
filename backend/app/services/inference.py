"""
Inference service.

The model is loaded ONCE at import time, not per request - that is instructor
review question #3. Right now this is a stub so the API can be built and tested
before the CV engineer delivers models/model.pt. Swap the marked section only.
"""
import io
import json
import random
import time
from pathlib import Path

from PIL import Image

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
    _model = None

    _model_loaded = True


def is_loaded() -> bool:
    return _model_loaded


def get_labels() -> list[str]:
    return _labels or _load_labels()


def run_inference(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                      
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    start = time.perf_counter()

    # ---- REPLACE THIS BLOCK WITH THE REAL FORWARD PASS -----------------
    labels = get_labels()
    scores = sorted((random.random() for _ in labels), reverse=True)
    total = sum(scores)
    probs = [s / total for s in scores]
    ranked = list(zip(labels, probs))
    # --------------------------------------------------------------------

    elapsed_ms = (time.perf_counter() - start) * 1000

    top = [{"class_name": c, "probability": round(p, 4)} for c, p in ranked[:5]]
    return {
        "predicted_class": top[0]["class_name"],
        "confidence": top[0]["probability"],
        "top_predictions": top,
        "inference_ms": round(elapsed_ms, 3),
        "model_version": settings.model_version,
    }