import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, history, model_info, predictions, stats
from app.config import settings
from app.database import engine
from app.models import Base
from app.services import inference

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Database schema verified")
    except Exception:
        log.exception("Could not reach the database at startup")

    inference.load_model()
    log.info("Model loaded once at startup")
    yield


app = FastAPI(
    title="CV Satellite Agent API",
    version=settings.model_version,
    description="Image classification service with PostgreSQL persistence.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /health sits at the root; everything else under /api/v1
app.include_router(health.router)
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")
app.include_router(model_info.router, prefix="/api/v1")