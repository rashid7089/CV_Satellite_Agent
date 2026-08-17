from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction
from app.models import User
from app.auth import current_user
from app.schemas import StatsOut

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db), user: User = Depends(current_user)):
    owned = Prediction.user_id == user.id
    total = db.scalar(select(func.count()).select_from(Prediction).where(owned)) or 0

    rows = db.execute(
        select(Prediction.predicted_class, func.count())
        .where(owned).group_by(Prediction.predicted_class)
        .order_by(func.count().desc())
    ).all()

    avg_conf = db.scalar(select(func.avg(Prediction.confidence)).where(owned))
    avg_ms = db.scalar(select(func.avg(Prediction.inference_ms)).where(owned))

    return StatsOut(
        total_predictions=total,
        class_distribution={cls: n for cls, n in rows},
        average_confidence=round(float(avg_conf), 4) if avg_conf is not None else None,
        average_inference_ms=round(float(avg_ms), 3) if avg_ms is not None else None,
    )
