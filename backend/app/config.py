from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),   # allows fields starting with model_
    )

    database_url: str = "postgresql+psycopg://postgres@localhost:5432/cvapp"

    model_path: str = "../models/ResNet_history.keras"
    labels_path: str = "../models/labels.json"
    model_version: str = "1.0.0"
    model_name: str = "resnet18"

    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 10

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
