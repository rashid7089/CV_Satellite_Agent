import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.schemas import BatchPredictionItem, PredictionOut
from app.auth import current_user
from app.models import User
from app.services import inference
from app.services.prediction_service import sanitize_filename, save_prediction
from app.services.object_storage import upload_image

router = APIRouter(tags=["predictions"])
log = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict", response_model=PredictionOut, status_code=201)
async def predict(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
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
        safe_name = sanitize_filename(image.filename)
        image_path = upload_image(raw, safe_name, image.content_type or "application/octet-stream")
        return save_prediction(
            db,
            image_name=safe_name,
            image_bytes=raw,
            result=result,
            image_path=image_path,
            user_id=user.id,
        )
    except SQLAlchemyError:
        db.rollback()
        log.exception("Database write failed")
        raise HTTPException(status_code=503, detail="Could not store the prediction.")


@router.post("/predict/batch", response_model=list[BatchPredictionItem], status_code=207)
async def predict_batch(
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if not 1 <= len(images) <= 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 images.")
    output: list[BatchPredictionItem] = []
    for image in images:
        name = sanitize_filename(image.filename)
        try:
            if image.content_type not in ALLOWED_TYPES:
                raise ValueError("Unsupported file type")
            raw = await image.read()
            if not raw or len(raw) > settings.max_upload_bytes:
                raise ValueError("Image is empty or exceeds the upload limit")
            result = inference.run_inference(raw)
            path = upload_image(raw, name, image.content_type or "application/octet-stream")
            row = save_prediction(db, image_name=name, image_bytes=raw, result=result, image_path=path, user_id=user.id)
            output.append(BatchPredictionItem(filename=name, prediction=row))
        except Exception as exc:
            db.rollback()
            log.warning("Batch item failed: %s", name, exc_info=True)
            output.append(BatchPredictionItem(filename=name, error=str(exc)))
    return output
