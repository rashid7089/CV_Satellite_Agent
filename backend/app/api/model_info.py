import json
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.schemas import ModelInfoOut
from app.services import inference

router = APIRouter(tags=["model"])


@router.get("/model", response_model=ModelInfoOut)
def model_info():
    metrics = None
    metrics_path = Path("../reports/model_metrics.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = None

    return ModelInfoOut(
        model_name=settings.model_name,
        model_version=settings.model_version,
        classes=inference.get_labels(),
        input_size=[128, 128],
        metrics=metrics,
        status="deployed" if inference.is_loaded() else "not_loaded",
    )
