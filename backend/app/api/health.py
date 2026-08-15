from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import HealthOut
from app.services import inference

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database = "healthy"
    except Exception:
        database = "unhealthy"

    return HealthOut(
        api="healthy",
        database=database,
        model="loaded" if inference.is_loaded() else "not_loaded",
    )