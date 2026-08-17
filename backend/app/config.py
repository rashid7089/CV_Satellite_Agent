from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),   # allows fields starting with model_
    )

    database_url: str = "postgresql+psycopg://postgres@localhost:5432/cvapp"

    model_path: str = "../models/model.onnx"
    labels_path: str = "../models/labels.json"
    model_version: str = "1.0.0"
    model_name: str = "ResNet50V2"

    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 10
    auth_secret: str = "change-this-auth-secret-in-production"
    access_token_hours: int = 24
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    redis_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    store_images: bool = False
    metrics_path: str = "reports/model_metrics.json"
    inference_cache_seconds: int = 3600

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
