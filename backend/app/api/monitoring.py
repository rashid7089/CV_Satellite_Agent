from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction
from app.schemas import MonitoringOut
from app.auth import current_user
from app.models import User

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring", response_model=MonitoringOut)
def monitoring(db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    owned = Prediction.user_id == user.id
    total, recent, avg_conf, low, avg_ms = db.execute(select(
        func.count(Prediction.id),
        func.count(Prediction.id).filter(Prediction.created_at >= since),
        func.avg(Prediction.confidence),
        func.sum(case((Prediction.confidence < 0.6, 1), else_=0)),
        func.avg(Prediction.inference_ms),
    ).where(owned)).one()
    versions = db.execute(select(Prediction.model_version, func.count()).where(owned).group_by(Prediction.model_version)).all()
    return MonitoringOut(
        total_predictions=total or 0,
        predictions_last_24h=recent or 0,
        average_confidence=round(float(avg_conf), 4) if avg_conf is not None else None,
        low_confidence_rate=round(float(low or 0) / total, 4) if total else None,
        average_inference_ms=round(float(avg_ms), 3) if avg_ms is not None else None,
        model_versions={version: count for version, count in versions},
    )
