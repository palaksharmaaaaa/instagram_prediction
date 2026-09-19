import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional
import joblib
from ..config import settings, get_logger

logger = get_logger("model_registry")

_REGISTRY_LOCK = threading.Lock()


class SecurityError(Exception):
    """Raised when artifact integrity verification or security check fails."""
    pass


_REGISTRY_CACHE: Dict[str, Any] = {
    "reach_pipeline": None,
    "impressions_pipeline": None,
    "metadata": None
}


def clear_registry_cache() -> None:
    """
    Clears the in-memory model registry cache in a thread-safe manner.
    """
    with _REGISTRY_LOCK:
        _REGISTRY_CACHE["reach_pipeline"] = None
        _REGISTRY_CACHE["impressions_pipeline"] = None
        _REGISTRY_CACHE["metadata"] = None


def compute_artifact_hash(path: Path) -> str:
    """
    Computes the SHA-256 digest of a model artifact file.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_artifact_integrity(artifact_path: Path, artifact_key: Optional[str] = None) -> bool:
    """
    Checks the SHA-256 digest of the artifact against an expected hash before calling joblib.load().
    If model_metadata.json exists and contains an artifact_hashes dictionary, verify the SHA-256 hash against it.
    If hashes don't match, raise a SecurityError("Artifact integrity verification failed").
    If artifact_hashes is not yet populated in metadata (e.g. during fresh training bootstrap),
    log a warning or compute it gracefully.
    """
    if not artifact_path.exists():
        return False

    actual_hash = compute_artifact_hash(artifact_path)

    if settings.MODEL_METADATA_PATH.exists():
        try:
            with open(settings.MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)

            artifact_hashes = metadata.get("artifact_hashes")
            if isinstance(artifact_hashes, dict):
                expected_hash = (
                    artifact_hashes.get(artifact_path.name)
                    or artifact_hashes.get(artifact_path.stem)
                    or (artifact_hashes.get(artifact_key) if artifact_key else None)
                )
                if expected_hash is not None:
                    if actual_hash.lower() != expected_hash.lower():
                        logger.error(
                            f"Artifact integrity verification failed for {artifact_path.name}: "
                            f"expected {expected_hash}, got {actual_hash}"
                        )
                        raise SecurityError("Artifact integrity verification failed")
                    logger.debug(f"Artifact integrity verified for {artifact_path.name}")
                    return True
                else:
                    logger.warning(
                        f"Artifact '{artifact_path.name}' not found in artifact_hashes; skipping verification."
                    )
            else:
                logger.warning(
                    "model_metadata.json exists but 'artifact_hashes' is not populated; skipping verification."
                )
        except SecurityError:
            raise
        except Exception as e:
            logger.warning(f"Could not verify artifact hash against metadata: {e}")
    else:
        logger.warning("model_metadata.json does not exist; skipping artifact verification.")

    return True


def get_reach_pipeline():
    """
    Loads and caches the Reach prediction pipeline after verifying artifact integrity.
    Thread-safe implementation using double-checked locking.
    """
    if _REGISTRY_CACHE["reach_pipeline"] is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_CACHE["reach_pipeline"] is None:
                if not settings.REACH_MODEL_PATH.exists():
                    from .trainer import train_and_persist_pipelines
                    logger.info("Reach pipeline not found. Triggering automated model training...")
                    train_and_persist_pipelines()
                verify_artifact_integrity(settings.REACH_MODEL_PATH, "reach_pipeline")
                _REGISTRY_CACHE["reach_pipeline"] = joblib.load(settings.REACH_MODEL_PATH)
    return _REGISTRY_CACHE["reach_pipeline"]


def get_impressions_pipeline():
    """
    Loads and caches the Impressions prediction pipeline after verifying artifact integrity.
    Thread-safe implementation using double-checked locking.
    """
    if _REGISTRY_CACHE["impressions_pipeline"] is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_CACHE["impressions_pipeline"] is None:
                if not settings.IMPRESSIONS_MODEL_PATH.exists():
                    from .trainer import train_and_persist_pipelines
                    logger.info("Impressions pipeline not found. Triggering automated model training...")
                    train_and_persist_pipelines()
                verify_artifact_integrity(settings.IMPRESSIONS_MODEL_PATH, "impressions_pipeline")
                _REGISTRY_CACHE["impressions_pipeline"] = joblib.load(settings.IMPRESSIONS_MODEL_PATH)
    return _REGISTRY_CACHE["impressions_pipeline"]


def get_model_metadata() -> Dict[str, Any]:
    """
    Retrieves model metadata and performance metrics.
    Thread-safe implementation using double-checked locking.
    """
    if _REGISTRY_CACHE["metadata"] is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_CACHE["metadata"] is None:
                if not settings.MODEL_METADATA_PATH.exists():
                    from .trainer import train_and_persist_pipelines
                    train_and_persist_pipelines()
                with open(settings.MODEL_METADATA_PATH, "r") as f:
                    _REGISTRY_CACHE["metadata"] = json.load(f)
    return _REGISTRY_CACHE["metadata"]
