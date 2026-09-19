from pathlib import Path
from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    PROJECT_NAME: str = "Reach Forecaster"
    VERSION: str = "3.0.0"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROFILES_PATH: Path = DATA_DIR / "creator_profiles.csv"   # observed profile-level facts
    POSTS_PATH: Path = DATA_DIR / "posts.csv"                 # user-supplied real post-level data
    MODELS_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = MODELS_DIR / "forecaster.joblib"
    MODEL_METADATA_PATH: Path = MODELS_DIR / "forecaster_metadata.json"

    # Minimum evidence required before a model may be trained or used.
    # Grouped 5-fold CV needs several creators per fold, and split-conformal
    # calibration needs enough residuals for a stable 90% quantile.
    MIN_TRAIN_POSTS: int = 100
    MIN_TRAIN_CREATORS: int = 30

    # A model must beat the follower-proportional baseline by this margin on
    # held-out creators, in a majority of folds, or the baseline is shipped.
    MIN_RELATIVE_IMPROVEMENT: float = 0.03
    MIN_FOLD_WIN_FRACTION: float = 0.60

    # Platform rule (Instagram hard limit on accounts followed)
    MAX_INSTAGRAM_FOLLOWING: int = 7500
    MAX_PROMPT_LENGTH: int = 500


settings = Settings()
