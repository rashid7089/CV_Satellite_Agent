import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    request_id: uuid.UUID
    image_name: str
    predicted_class: str
    confidence: float
    top_k_predictions: list[TopPrediction] | None = None
    inference_ms: float          # coerces the Decimal coming out of NUMERIC(10,3)
    model_version: str
    created_at: datetime


class InferenceResult(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: list[TopPrediction]
    inference_ms: float
    model_version: str


class StatsOut(BaseModel):
    total_predictions: int
    class_distribution: dict[str, int]
    average_confidence: float | None = None
    average_inference_ms: float | None = None


class ModelInfoOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    classes: list[str]
    input_size: list[int] = Field(default=[224, 224])
    metrics: dict | None = None
    status: str


class HealthOut(BaseModel):
    api: str
    database: str
    model: str