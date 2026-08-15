import hashlib

from sqlalchemy.orm import Session

from app.models import Prediction


def save_prediction(
    db: Session,
    *,
    image_name: str,
    image_bytes: bytes,
    result: dict,
    image_path: str | None = None,
) -> Prediction:
    """Persist one inference result. Commits and returns the stored row."""
    row = Prediction(
        image_name=image_name,
        image_path=image_path,
        image_hash=hashlib.sha256(image_bytes).hexdigest(),
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top_k_predictions=result["top_predictions"],
        inference_ms=result["inference_ms"],
        model_version=result["model_version"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def sanitize_filename(name: str | None) -> str:
    """Strip directory components and keep it short - spec 34."""
    if not name:
        return "upload"
    base = name.replace("\\", "/").split("/")[-1]
    safe = "".join(ch for ch in base if ch.isalnum() or ch in "._- ")
    return safe[:200] or "upload"