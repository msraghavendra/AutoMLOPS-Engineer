import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./shipitml.db"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    API_KEY: str = "ship_it_ml_secret_api_key_2026"
    CORS_ORIGINS: str = "*"  # Comma-separated list in production, e.g., "http://localhost:3000"
    DEFAULT_DRIFT_THRESHOLD: float = 0.3  # Ratio of features drifted to trigger alert
    UPLOAD_DIR: str = "uploads"
    MODEL_DIR: str = "models"
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
