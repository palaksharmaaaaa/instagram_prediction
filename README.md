# Reach Forecaster

Forecast the reach (and, when observed, impressions) of a social post **from real data you supply** - and refuse
when the data cannot support an answer.

**The rule of this project:** every number shown is either *observed* (in a file you provide, or returned by the
Meta Graph API) or *computed from observed data by a model whose measured accuracy is shown next to it*.
Nothing is generated, defaulted, imputed silently, or rewritten.

## What it does and does not do

| It does | It does not |
|---|---|
| Trains only on real post-level rows in `data/posts.csv` | Ship any training data or pre-trained model |
| Refuses to train below 100 posts / 30 creators | Fill gaps with "typical" values |
| Compares every model with a follower-proportional baseline on **held-out creators** and ships the baseline unless a model clearly wins | Claim a model is better than a simple rule without evidence |
| Gives 80% / 90% conformal intervals and reports their **measured** coverage on held-out creators | Present an interval whose coverage it has not measured |
| Refuses forecasts outside the trained follower range or for unseen platforms/formats | Extrapolate or guess |
| Lists every optional input you left blank (`imputed_fields`) | Hide what the model filled in |
| Reports search filters it could not apply | Silently return unfiltered results |
| Records only fields Meta actually returns | Guess country, category, CTA, video length, or impressions |

Forecasts describe **association in your data**, not the causal effect of a creative choice. Accuracy is only
known for creators similar to those you trained on.

## Data

* `data/creator_profiles.csv` - 200 creators, profile-level facts *as supplied* (unverified snapshot). See
  [`data/PROVENANCE.md`](data/PROVENANCE.md) for what it is and which fabricated columns/rows were removed.
* `data/posts.csv` - **you provide this**: real per-post observations. Template: `data/posts_template.csv`.
  Required: `username, media_type, posted_day_of_week, posted_hour_of_day, total_followers, per_media_reach`.
  Optional (used only if >= 80% of rows have them): `caption_length_chars, hashtags_count, mentions_count,
  has_call_to_action, video_duration_seconds, carousel_slide_count, per_media_impressions`.
  Invalid rows (e.g. impressions < reach) are rejected and listed - never repaired.

Bring posts from your own analytics export, or use **Data -> Import from Meta Graph API**, which appends only
observed insights (hours are UTC; `total_followers` is the *current* count; `impressions` is unavailable in current
API versions).

## Run

```bash
uv sync --extra dev
uv run streamlit run app.py
```

Train from the UI (Data tab) or:

```bash
uv run python -m instagram_predictor.models.trainer
```

## How training works (`src/instagram_predictor/models/trainer.py`)

1. Load and validate `posts.csv` (`data/loader.py`).
2. Choose features from the data itself (an optional column needs >= 80% coverage and variation).
3. Score three candidates by **repeated creator-grouped CV**: a follower-proportional baseline, a ridge model, and
   gradient boosting. All learn deviations from `log(followers)`, so predictions scale with follower count.
4. Ship a non-baseline model only if it lowers held-out log-MAE by >= 3% **and** wins in >= 60% of folds.
5. Calibrate intervals with cross-conformal residuals (out-of-fold, grouped by creator; follower-tier calibration
   only where there is enough evidence), then **measure** their coverage on creators never used for fitting or
   calibration.
6. Save the model with its SHA-256, library versions, and a fingerprint of the training data. Loading fails closed on
   a hash or scikit-learn version mismatch.

## Tests

```bash
uv run pytest
```

`tests/truthful/` covers loaders (rejects, never repairs), trainer gating and baseline selection, interval
coverage, forecast refusal and disclosure, artifact integrity, the parser's warnings, the Meta connector's
"only what Meta returned" rule, and the UI. Fixtures build tiny frames with known ground truth **inside the tests
only**; they are never written to `data/` or read by the app.

## License

MIT - see [LICENSE](LICENSE).
