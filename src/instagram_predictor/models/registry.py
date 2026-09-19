import json
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
from ..config import settings, get_logger

logger = get_logger("model_registry")

_REGISTRY_CACHE: Dict[str, Any] = {
    "reach_pipeline": None,
    "impressions_pipeline": None,
    "metadata": None
}


def get_reach_pipeline():
    """
    Loads and caches the Reach prediction pipeline.
    """
    if _REGISTRY_CACHE["reach_pipeline"] is None:
        if not settings.REACH_MODEL_PATH.exists():
            from .trainer import train_and_persist_pipelines
            logger.info("Reach pipeline not found. Triggering automated model training...")
            train_and_persist_pipelines()
        _REGISTRY_CACHE["reach_pipeline"] = joblib.load(settings.REACH_MODEL_PATH)
    return _REGISTRY_CACHE["reach_pipeline"]


def get_impressions_pipeline():
    """
    Loads and caches the Impressions prediction pipeline.
    """
    if _REGISTRY_CACHE["impressions_pipeline"] is None:
        if not settings.IMPRESSIONS_MODEL_PATH.exists():
            from .trainer import train_and_persist_pipelines
            logger.info("Impressions pipeline not found. Triggering automated model training...")
            train_and_persist_pipelines()
        _REGISTRY_CACHE["impressions_pipeline"] = joblib.load(settings.IMPRESSIONS_MODEL_PATH)
    return _REGISTRY_CACHE["impressions_pipeline"]


def get_model_metadata() -> Dict[str, Any]:
    """
    Retrieves model metadata and performance metrics.
    """
    if _REGISTRY_CACHE["metadata"] is None:
        if not settings.MODEL_METADATA_PATH.exists():
            from .trainer import train_and_persist_pipelines
            train_and_persist_pipelines()
        with open(settings.MODEL_METADATA_PATH, "r") as f:
            _REGISTRY_CACHE["metadata"] = json.load(f)
    return _REGISTRY_CACHE["metadata"]
