import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.schemas import PredictionOut
from app.services import inference
from app.services.prediction_service import sanitize_filename, save_prediction

router = APIRouter(tags=["predictions"])
log = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict", response_model=PredictionOut, status_code=201)
async def predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    raw = await image.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.max_upload_mb} MB limit.",
        )

    try:
        result = inference.run_inference(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed.")

    try:
        return save_prediction(
            db,
            image_name=sanitize_filename(image.filename),
            image_bytes=raw,
            result=result,
        )
    except SQLAlchemyError:
        db.rollback()
        log.exception("Database write failed")
        raise HTTPException(status_code=503, detail="Could not store the prediction.")