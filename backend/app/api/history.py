from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction
from app.models import User
from app.auth import current_user
from app.schemas import PredictionOut

router = APIRouter(tags=["history"])


@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    predicted_class: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    stmt = select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.created_at.desc())
    if predicted_class:
        stmt = stmt.where(Prediction.predicted_class == predicted_class)
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.get(Prediction, prediction_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail=f"No prediction with id {prediction_id}.")
    return row
