"""
Test fixtures.

IMPORTANT: the frames built here exist only inside the test process to exercise code paths with a KNOWN
ground truth (e.g. "does the trainer pick a model when a real effect exists, and the baseline when none does?").
They are never written into data/, never shipped, and never read by the application.
"""

import io

import numpy as np
import pandas as pd
import pytest

from instagram_predictor.config import settings
from instagram_predictor.data import load_posts
from instagram_predictor.models import clear_registry_cache, save_model, train_forecaster


def make_posts_csv(seed=0, n_creators=60, per_creator=4, reel_effect=0.0, with_impressions=True) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_creators):
        followers = int(np.exp(rng.uniform(np.log(2e3), np.log(5e6))))
        creator_effect = rng.normal(0, 0.3)
        for j in range(per_creator):
            media = str(rng.choice(["Reel", "Carousel", "Static Image"]))
            cta = int(rng.random() < 0.5)
            log_reach = np.log(followers * 0.2) + creator_effect + (reel_effect if media == "Reel" else 0.0) + rng.normal(0, 0.25)
            reach = int(np.exp(log_reach))
            row = {
                "post_id": f"c{i}_{j}", "username": f"creator{i}", "media_type": media,
                "posted_day_of_week": str(rng.choice(["Monday", "Friday", "Sunday"])),
                "posted_hour_of_day": int(rng.integers(0, 24)), "total_followers": followers,
                "per_media_reach": reach, "has_call_to_action": cta, "hashtags_count": int(rng.integers(0, 20)),
            }
            if with_impressions:
                row["per_media_impressions"] = int(reach * rng.uniform(1.1, 1.6))
            rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


@pytest.fixture
def posts_effect():
    posts, _ = load_posts(io.StringIO(make_posts_csv(seed=1, reel_effect=0.6)))
    return posts


@pytest.fixture
def posts_noise():
    posts, _ = load_posts(io.StringIO(make_posts_csv(seed=2, reel_effect=0.0)))
    return posts


@pytest.fixture
def trained_model(tmp_path, monkeypatch, posts_effect):
    """Trains on the effect fixture and points the registry at a temp directory."""
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "forecaster.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "forecaster_metadata.json")
    bundle, meta = train_forecaster(posts_effect, coverage_repeats=8)
    save_model(bundle, meta)
    clear_registry_cache()
    yield bundle, meta
    clear_registry_cache()
