"""
Model loading. Fails closed: a missing, tampered, or version-mismatched model is an error,
never a silent fallback and never an automatic retrain on data the user did not choose.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import sklearn

from ..config import settings, get_logger

logger = get_logger("model_registry")
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"key": None, "bundle": None, "metadata": None}

_NO_MODEL_MSG = (
    "No trained model. Provide real post-level data in data/posts.csv (see data/posts_template.csv) and train."
)


class NoModelError(RuntimeError):
    """No trained model exists (train one from real post-level data first)."""


class SecurityError(RuntimeError):
    """Model artifact failed integrity verification."""


class ModelVersionError(RuntimeError):
    """Model was trained under a different scikit-learn version and must be retrained."""


def clear_registry_cache() -> None:
    with _LOCK:
        _CACHE.update(key=None, bundle=None, metadata=None)


def compute_artifact_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def model_available() -> bool:
    return Path(settings.MODEL_PATH).exists() and Path(settings.MODEL_METADATA_PATH).exists()


def load_metadata() -> Dict[str, Any]:
    p = Path(settings.MODEL_METADATA_PATH)
    if not p.exists():
        raise NoModelError(_NO_MODEL_MSG)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (bundle, metadata) after verifying the artifact hash and library version."""
    model_path, meta_path = Path(settings.MODEL_PATH), Path(settings.MODEL_METADATA_PATH)
    if not (model_path.exists() and meta_path.exists()):
        raise NoModelError(_NO_MODEL_MSG)
    key = (str(model_path), model_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["bundle"], _CACHE["metadata"]
        metadata = load_metadata()
        expected = metadata.get("artifact_sha256")
        if not expected:
            raise SecurityError("Model metadata has no artifact_sha256; refusing to load an unverifiable model.")
        if compute_artifact_hash(model_path).lower() != str(expected).lower():
            raise SecurityError("Artifact integrity verification failed: model file does not match its recorded hash.")
        trained_with = metadata.get("software", {}).get("scikit_learn")
        if trained_with != sklearn.__version__:
            raise ModelVersionError(
                f"Model was trained with scikit-learn {trained_with}, installed is {sklearn.__version__}. Retrain the model."
            )
        bundle = joblib.load(model_path)
        _CACHE.update(key=key, bundle=bundle, metadata=metadata)
        return bundle, metadata
