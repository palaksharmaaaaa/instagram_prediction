import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    PROJECT_NAME: str = "Instagram AI Prediction Engine"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Path resolution
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "instagram_profiles_posts.csv"
    LEGACY_DATA_PATH: Path = DATA_DIR / "top_200_instagrammers.csv"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Artifact Paths
    REACH_MODEL_PATH: Path = MODELS_DIR / "reach_pipeline.joblib"
    IMPRESSIONS_MODEL_PATH: Path = MODELS_DIR / "impressions_pipeline.joblib"
    MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

    # Instagram Platform Constraints & Guardrails
    MAX_INSTAGRAM_FOLLOWING: int = 7500
    MIN_ENGAGEMENT_RATE_WARN: float = 0.0005  # < 0.05% engagement triggers bot warning
    VIRALITY_DEV_THRESHOLD: float = 3.0       # 3-sigma deviation on share/like ratio
    MAX_PROMPT_LENGTH: int = 500

settings = Settings()
