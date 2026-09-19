# ⚡ Instagram AI Prediction & Analytics Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Environment: uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Framework: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20LightGBM-orange.svg)](https://scikit-learn.org/)
[![Validation: Pydantic v2](https://img.shields.io/badge/validation-Pydantic%20v2-green.svg)](https://docs.pydantic.dev/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests: 92 Passing](https://img.shields.io/badge/tests-92%2F92%20passing-brightgreen.svg)](https://pytest.org/)

> **Enterprise-grade machine learning forecasting, Mondrian conformal uncertainty quantification, high-dimensional feature engineering, defensive guardrails, and what-if post simulation engine for Instagram creators, influencer marketing agencies, and social media brands.**

---

## 📑 Table of Contents
1. [Executive Summary & Core Principles](#-executive-summary--core-principles)
2. [End-to-End System Architecture](#-end-to-end-system-architecture)
3. [Project Directory & File Manifest](#-project-directory--file-manifest)
4. [Granular 47-Column Dataset & Data Architecture](#-granular-47-column-dataset--data-architecture)
5. [Hierarchical Categorization Architecture](#-hierarchical-categorization-architecture)
6. [Pydantic v2 Defensive Schema & Validation](#-pydantic-v2-defensive-schema--validation)
7. [High-Dimensional Feature Engineering (51 Features)](#-high-dimensional-feature-engineering-51-features)
8. [Machine Learning Models & Algorithms](#-machine-learning-models--algorithms)
9. [Mondrian Conformal Prediction & Epistemic Uncertainty](#-mondrian-conformal-prediction--epistemic-uncertainty)
10. [Defensive Guardrails & Anomaly Detection](#-defensive-guardrails--anomaly-detection)
11. [NLP Natural Language Query Engine](#-nlp-natural-language-query-engine)
12. [Streamlit Enterprise Dashboard (`app.py`)](#-streamlit-enterprise-dashboard-apppy)
13. [CLI Scripts & Execution Guide](#-cli-scripts--execution-guide)
14. [Comprehensive Test Suite (92 Tests)](#-comprehensive-test-suite-92-tests)
15. [Installation & Build Configuration](#-installation--build-configuration)

---

## 🌟 Executive Summary & Core Principles

The **Instagram AI Prediction & Analytics Engine** is designed to solve one of the most difficult challenges in social media analytics: **forecasting organic reach and impressions before publishing a post**, while quantifying predictive uncertainty with finite-sample mathematical guarantees.

### Architectural Pillars

1. **Defensive Separation of Concerns**: Strict boundary isolation across domain schemas, defensive guardrails, feature engineering transformations, gradient-boosted decision trees, natural language query parsers, and presentation dashboards.
2. **Zero-Leakage Creator Validation**: Strict GroupKFold cross-validation grouped by creator `username` ensures models generalize across distinct creators rather than memorizing individual account biases.
3. **Mondrian Conformal Calibration**: Quantifies heteroscedastic prediction uncertainty across distinct creator tiers (Nano, Micro, Macro, Mega) with provable finite-sample coverage guarantees ($1 - \alpha$).
4. **Cryptographic Artifact Integrity**: Model weights and transformers are verified via SHA-256 HMAC digests prior to deserialization, closing arbitrary code execution (RCE) vectors.
5. **Multi-Threaded Concurrency & Safety**: Double-checked locking with `threading.Lock()` and atomic file replacement guarantee thread safety in multi-user Streamlit deployments.
6. **Epistemic Out-Of-Distribution (OOD) Safety**: Evaluates feature-space distances from the training distribution, alerting users when simulations represent extrapolation beyond observed data.
7. **Deterministic Invariant Enforcement**: Mathematical domain invariants are strictly guaranteed ($\text{Reach} \le \text{Impressions}$, $\text{Likes} \le \text{Impressions}$, $\text{Shares} \le \text{Impressions}$, $\text{Following} \le 7,500$).

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        GEN[Data Generator<br/>generator.py] -->|47 Columns| CSV[(instagram_profiles_posts.csv)]
        CSV --> LOADER[Cached Data Loader<br/>loader.py]
    end

    subgraph Validation & Guardrails
        LOADER --> SCHEMAS[Pydantic v2 Schemas<br/>ProfileInput / PostInput]
        SCHEMAS --> GUARDS[Defensive Guardrails<br/>Sanity / Bot / Viral / Prompt / CSV DDE Defense]
    end

    subgraph Feature Engineering
        GUARDS --> FE[Feature Engineering Pipeline<br/>feature_engineering.py]
        FE -->|44 Numeric + 7 Categorical| MAT[51-Dimensional Feature Matrix]
    end

    subgraph Training & Modeling
        MAT --> GKF[GroupKFold Split<br/>Grouped by Creator]
        GKF --> LGBM[LightGBM / HistGBM<br/>Log1p Target Transform]
        LGBM --> HPO[Optuna Bayesian HPO<br/>Log-Scale MAE Loss]
        LGBM --> MONDRIAN[Mondrian Conformal Calibration<br/>Nano / Micro / Macro / Mega Tiers]
        MONDRIAN --> ARTIFACTS[Serialized Artifacts<br/>models/*.joblib & model_metadata.json]
    end

    subgraph Service & Interface
        ARTIFACTS --> SEC_CHECK[SHA-256 Cryptographic Check<br/>registry.py]
        SEC_CHECK --> ENGINE[Prediction Engine<br/>engine.py]
        NLP[NLP Query Parser<br/>nlp/parser.py] --> SERVICE[Analytics & Simulator Services]
        ENGINE --> SERVICE
        SERVICE --> APP[Streamlit Enterprise App<br/>app.py (Cached Resources)]
    end
```

---

## 📂 Project Directory & File Manifest

```
project-instagram-prediction/
├── pyproject.toml                                # Dependency specs, build system, and pytest config
├── README.md                                     # Comprehensive system documentation
├── data/
│   ├── raw/
│   │   └── instagram_profiles_posts.csv          # 47-column enterprise dataset (750 posts across 4 tiers)
│   └── top_200_instagrammers.csv                 # Legacy benchmark creator dataset
├── models/
│   ├── reach_pipeline.joblib                     # Serialized Reach ML Pipeline (Preprocessor + GBDT)
│   ├── impressions_pipeline.joblib               # Serialized Impressions ML Pipeline (Preprocessor + GBDT)
│   └── model_metadata.json                       # Versioning, metrics, SHA-256 hashes & Mondrian quantiles
├── src/
│   ├── instagram_predictor/                      # Core production package
│   │   ├── __init__.py                           # Package exports
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py                        # Structured JSON/console logging setup
│   │   │   └── settings.py                       # Project paths, constants, and global configurations
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── profile.py                        # ProfileInput, PostInput, ContentCategory, ContentStyle
│   │   │   ├── prediction.py                     # PredictionInput, PredictionOutput, ConfidenceInterval
│   │   │   └── query.py                          # QueryIntent, FilterCriteria, ParsedQuery
│   │   ├── guardrails/
│   │   │   ├── __init__.py
│   │   │   ├── anomaly_detector.py               # Bot detection, zero-comment anomaly, viral spikes
│   │   │   ├── input_validator.py                # Type & boundary enforcement
│   │   │   ├── safety.py                         # Recursive HTML neutralization, prompt defense, CSV DDE
│   │   │   └── sanity_rules.py                   # Platform bounds & mathematical invariants
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py                      # Stratified log-normal synthetic data generator (Nano to Mega)
│   │   │   ├── loader.py                         # Thread-safe cached CSV loader with robust list parsing
│   │   │   └── feature_engineering.py            # 51-feature matrix transformer & interaction signals
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py                         # Inference engine, post simulation, and Mondrian bounds
│   │   │   ├── hpo.py                            # Optuna Bayesian HPO with log-scale error evaluation
│   │   │   ├── registry.py                       # Thread-safe pipeline loading with SHA-256 verification
│   │   │   └── trainer.py                        # GroupKFold training, exact conformal calibration
│   │   ├── nlp/
│   │   │   ├── __init__.py
│   │   │   ├── auditor.py                        # Query auditing & telemetry
│   │   │   ├── intent_analyzer.py                # Intent classification (search, compare, simulate)
│   │   │   └── parser.py                         # Two-tiered NLP regex parser with decimal multipliers
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── analytics_service.py              # Profile discovery, regex-safe filters, KPIs
│   │   │   └── simulator_service.py              # Post simulation orchestration & creator recommendations
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── formatting.py                     # Humanized number, percentage, and currency formatters
│   ├── pipeline.py                               # Legacy prototype pipeline (deprecated, points to v2)
│   ├── predictor.py                              # Legacy predictor interface (deprecated, regex-hardened)
│   ├── preprocessing.py                          # Legacy preprocessing wrapper (deprecated)
│   ├── prompt_parser.py                          # Legacy prompt parser wrapper (deprecated)
│   ├── synthetic_targets.py                      # Legacy target generator (deprecated)
│   └── train.py                                  # Legacy training runner (deprecated)
├── app.py                                        # Enterprise Streamlit Web Dashboard (Cached)
└── tests/                                        # Comprehensive pytest suite (92 tests)
    ├── test_audit_hardening.py                   # 17 consolidated audit edge-case and invariant tests
    ├── test_concurrency.py                       # Multi-threaded stress tests & deadlock verification
    ├── test_conformal_mondrian.py                # Exact finite-sample conformal quantiles & Nano tier tests
    ├── test_end_to_end.py                        # Legacy end-to-end scenarios (10 tests)
    ├── test_end_to_end_service.py                # Analytics & simulator service tests
    ├── test_feature_engineering.py               # Feature generation, list parsing, and Nano tier tests
    ├── test_guardrails.py                        # Sanity invariants, bot detection, zero-comment anomalies
    ├── test_models.py                            # Pipeline loading, creative sensitivity, overflow guards
    ├── test_query_parser.py                      # NLP parser unit tests & decimal multipliers
    ├── test_schemas.py                           # Pydantic v2 schema boundary & gender sum tests
    └── test_security.py                          # SHA-256 tampering, regex DoS, HTML evasion, CSV DDE
```

---

## 📊 Granular 47-Column Dataset & Data Architecture

The dataset ([`instagram_profiles_posts.csv`](file:///d:/Work/Projects/project-instagram-prediction/data/raw/instagram_profiles_posts.csv)) consists of **750 post records** across 250 creators, stratified across all 4 creator tiers:
- **Nano Tier (<10k)**: 177 posts (59 profiles, min 503 followers)
- **Micro Tier (10k-100k)**: 177 posts (59 profiles)
- **Macro Tier (100k-1M)**: 177 posts (59 profiles)
- **Mega Tier (>=1M)**: 219 posts (73 profiles, including 14 seed celebrities)

### Exhaustive 47-Column Schema Specification

| # | Column Name | Data Type | Valid Range / Allowed Values | Description & Algorithmic Significance |
|---|---|---|---|---|
| **1** | `username` | `string` | Unique handle (a-z, 0-9, `_`, `.`) | Creator identifier used for GroupKFold validation grouping |
| **2** | `full_name` | `string` | UTF-8 String | Creator display name |
| **3** | `country` | `string` | ISO 2-letter country code | Creator home country |
| **4** | `total_followers` | `int64` | $500 \le x \le 1,000,000,000$ | Total follower count (stratified log-normal) |
| **5** | `total_following` | `int64` | $0 \le x \le 7,500$ | Instagram platform limit capped at 7,500 |
| **6** | `total_media_posts`| `int64` | $1 \le x \le 100,000$ | Total lifetime posts published |
| **7** | `is_verified` | `bool` | `True`, `False` | Instagram Blue Badge verification status |
| **8** | `account_category` | `string` | 9 Content Categories | Primary profile niche / industry |
| **9** | `account_age_years` | `float64` | $0.5 \le x \le 15.0$ | Account age in years (authority signal) |
| **10** | `posting_frequency_per_week` | `float64` | $0.5 \le x \le 28.0$ | Average posts published per week |
| **11** | `follower_growth_rate_30d` | `float64` | $-0.20 \le x \le +2.00$ | Net 30-day percentage follower growth |
| **12** | `account_bio_has_link` | `bool` | `True`, `False` | Presence of external link in creator bio |
| **13** | `post_id` | `string` | Unique string (e.g. `user_p1`) | Unique post identifier |
| **14** | `media_type` | `string` | `Reel`, `Carousel`, `Static Image`, `Story` | Instagram media format |
| **15** | `category` | `string` | 9 Content Categories | Single primary post category |
| **16** | `categorization` | `string` | Comma-separated string | Primary style or comma-delimited styles |
| **17** | `categorizations` | `list[str]` | 1 to 3 distinct Content Styles | Multi-label array of content styles |
| **18** | `caption_length_chars` | `int64` | $10 \le x \le 2,200$ | Character count of the caption (enforced $\le 2200$) |
| **19** | `hashtags_count` | `int64` | $0 \le x \le 30$ | Number of hashtags (platform limit 30) |
| **20** | `mentions_count` | `int64` | $0 \le x \le 20$ | Number of tagged `@handles` |
| **21** | `has_call_to_action` | `bool` | `True`, `False` | Presence of explicit Save/Share/Comment CTA |
| **22** | `video_duration_seconds` | `float64` | $0.0 \le x \le 90.0$ | Duration in seconds (0 for Static/Carousel) |
| **23** | `carousel_slide_count` | `int64` | $1 \le x \le 10$ | Number of slides (1 for Reel/Static) |
| **24** | `posted_day_of_week` | `string` | `Monday` – `Sunday` | Day of publication |
| **25** | `posted_hour_of_day` | `int64` | $0 \le x \le 23$ | Hour of publication (local time) |
| **26** | `is_weekend` | `int64` | `0` or `1` | Dynamically derived weekend indicator |
| **27** | `is_peak_posting_hour` | `int64` | `0` or `1` | Dynamically derived peak hour indicator |
| **28** | `top_country` | `string` | ISO 2-letter code | Primary audience country |
| **29** | `secondary_country` | `string` | ISO 2-letter code | Secondary audience country |
| **30** | `primary_age_group` | `string` | `13-17`, `18-24`, `25-34`, `35-44`, `45-54`, `55+` | Predominant demographic age bracket |
| **31** | `gender_female_pct` | `float64` | $0.0 \le x \le 1.0$ | Percentage of female audience |
| **32** | `gender_male_pct` | `float64` | $0.0 \le x \le 1.0$ | Percentage of male audience ($\text{female} + \text{male} = 1.0$) |
| **33** | `audience_activity_score` | `float64` | $0.1 \le x \le 1.0$ | Audience daily active engagement score |
| **34** | `per_media_likes` | `int64` | $0 \le x \le \text{Impressions}$ | Observed likes count |
| **35** | `per_media_comments` | `int64` | $0 \le x \le \text{Impressions}$ | Observed comments count (enforced $\le \text{Impressions}$) |
| **36** | `per_media_shares` | `int64` | $0 \le x \le \text{Impressions}$ | Observed shares (enforced $\le \text{Impressions}$) |
| **37** | `per_media_saves` | `int64` | $0 \le x \le \text{Impressions}$ | Observed saves (enforced $\le \text{Impressions}$) |
| **38** | `per_media_video_views` | `int64` | $0 \le x \le \text{Impressions}$ | Video plays (for Reels / Videos) |
| **39** | `per_media_completion_rate` | `float64` | $0.0 \le x \le 1.0$ | Video watch-through rate |
| **40** | `reach_from_home_pct` | `float64` | $0.0 \le x \le 1.0$ | Percentage of reach from follower feed |
| **41** | `reach_from_explore_pct`| `float64` | $0.0 \le x \le 1.0$ | Percentage of reach from Explore page |
| **42** | `reach_from_hashtags_pct`| `float64`| $0.0 \le x \le 1.0$ | Percentage of reach from search/hashtags |
| **43** | `reach_from_other_pct` | `float64` | $0.0 \le x \le 1.0$ | Percentage of reach from DMs/external |
| **44** | `per_media_reach` | `int64` | $100 \le x \le \text{Impressions}$ | **Target 1**: Unique accounts reached |
| **45** | `per_media_impressions` | `int64` | $x \ge \text{Reach}$ | **Target 2**: Total view occurrences |
| **46** | `account_categories` | `list[str]` | List of Content Categories | Profile-wide distinct category array |
| **47** | `account_categories_str` | `string` | Comma-delimited categories | Serialized profile distinct categories |

---

## 🏷️ Hierarchical Categorization Architecture

The system enforces a dual-level categorization architecture:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             HIERARCHICAL CATEGORIZATION ARCHITECTURE                             │
├──────────────────────┬───────────────────────────────┬───────────────────────────────────────────┤
│ Hierarchy Level      │ Fields                        │ Values / Taxonomies                       │
├──────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ **Profile Level**    │ `account_category`            │ Single primary niche                      │
│                      │ `account_categories`          │ Distinct array of all post categories     │
│                      │                               │ published by this creator                 │
│                      │ `account_categories_str`      │ Comma-separated export string             │
├──────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ **Post Level**       │ `category`                    │ Single primary category of the post       │
│                      │ `categorizations`             │ Array of 1 to 3 distinct Content Styles   │
│                      │ `categorization`              │ Comma-separated or legacy single style    │
└──────────────────────┴───────────────────────────────┴───────────────────────────────────────────┘
```

---

## 🛡️ Pydantic v2 Defensive Schema & Validation

All incoming data—whether loaded from CSV, submitted via API, or provided through Streamlit—is rigorously validated using **Pydantic v2**:

### Key Schemas ([`src/instagram_predictor/schemas/`](file:///d:/Work/Projects/project-instagram-prediction/src/instagram_predictor/schemas/))

- **`PostInput`**:
  - Enforces platform boundaries: `caption_length_chars` ($0 \le x \le 2200$), `video_duration_seconds` ($0 \le x \le 90$), `carousel_slide_count` ($1 \le x \le 10$), `hashtags_count` ($0 \le x \le 30$).
  - Dual-direction `@model_validator(mode="before")` synchronizes `categorization` and `categorizations`.
  - Exposes property `primary_categorization` to guarantee scalar access when required.
- **`ProfileInput`**:
  - Enforces platform constraints: `total_following` capped at 7,500.
  - Demographic bounds: `gender_female_pct + gender_male_pct == 1.0` ($\pm 0.05$).
  - Pre-validator `sync_gender_pct`: Automatically populates the opposite gender percentage if only one is specified, preserving backward compatibility.
  - Synchronizes `account_category` with `account_categories`.
- **`ConfidenceInterval`**:
  - Enforces monotonic confidence bounds: `lower_bound` $\le$ `point_estimate` $\le$ `upper_bound`.
  - Contains `level: Optional[str]` reporting the exact calibrated coverage applied (e.g. `"80% Mondrian Conformal Coverage"`).
- **`ParsedQuery`**:
  - Exposes ergonomic helper properties `username`, `min_followers`, and `max_followers` with decimal multiplier resolution.

---

## 🔬 High-Dimensional Feature Engineering (51 Features)

The feature engineering pipeline ([`feature_engineering.py`](file:///d:/Work/Projects/project-instagram-prediction/src/instagram_predictor/data/feature_engineering.py)) transforms raw inputs into a **51-dimensional machine learning feature matrix** (44 Numeric + 7 Categorical).

### 1. Multi-Hot Binary Categorization Encodings

Post content styles are decomposed into 5 multi-hot binary indicator features:
- `is_educational = 1` if `"Educational / How-To"` $\in \text{categorizations}$, else `0`.
- `is_entertaining = 1` if `"Entertaining / Trend"` $\in \text{categorizations}$, else `0`.
- `is_promotional = 1` if `"Promotional / Sponsored"` $\in \text{categorizations}$, else `0`.
- `is_behind_scenes = 1` if `"Behind-the-Scenes / Personal"` $\in \text{categorizations}$, else `0`.
- `is_inspirational = 1` if `"Inspirational / Storytelling"` $\in \text{categorizations}$, else `0`.

### 2. Dynamic Temporal Features
- `is_weekend`: Dynamically derived as `1.0` if `posted_day_of_week` is Saturday or Sunday, else `0.0`.
- `is_peak_posting_hour`: Dynamically derived as `1.0` if `posted_hour_of_day` $\in [11, 12, 13, 18, 19, 20, 21]$, else `0.0`.

### 3. Domain Interaction Terms & Mathematical Formulas

#### 1. Creator Scale Engagement (Replaces Collinear Reach Potential)
$$\text{Creator Scale Engagement} = \ln(1 + \text{Followers}) \times \left(\frac{\text{Total Engagement}}{\text{Followers} + 1}\right)$$

#### 2. Save Efficiency
$$\text{Save Efficiency} = \left(\frac{\text{Saves}}{\text{Likes} + 1}\right) \times (1 + \text{IsCarousel} + 0.5 \times \text{IsEducational})$$

#### 3. Virality Momentum
$$\text{Virality Momentum} = \frac{2.0 \times \text{Shares} + 1.5 \times \text{Saves}}{\text{Likes} + \text{Comments} + 1}$$

#### 4. Explore Discovery Potential
$$\text{Explore Discovery Potential} = \text{Virality Momentum} \times (1 + \text{Explore Reach Pct})$$

#### 5. Watch Efficiency
$$\text{Watch Efficiency} = \text{Completion Rate} \times \text{IsReel}$$

#### 6. Call-to-Action Boost
$$\text{Call-to-Action Boost} = \text{HasCTA} \times (1 + 0.5 \times \text{IsCarousel})$$

#### 7. Hashtag Density
$$\text{Hashtag Density} = \frac{\text{Hashtags Count}}{\ln(1 + \text{Caption Length Chars}) + 1}$$

---

## 🤖 Machine Learning Models & Algorithms

### 1. Model Selection & Architecture
- **LightGBM Regressors** (`LGBMRegressor`) / **HistGradientBoostingRegressor** with histogram-based binning.
- **Target Transformation**: Trained under a $\log(1 + y)$ (`log1p`) transformation with inverse exponential mapping (`expm1`).
- **Numerical Overflow Protection**: Log bounds are clipped to `[0.0, 30.0]` before exponentiation via `np.expm1`, preventing infinite bounds or overflow errors.
- **Cryptographic Checksum Verification**: Model pipelines are verified via SHA-256 digests recorded in `model_metadata.json` before `joblib.load()` executes.

### 2. Validation Strategy & HPO
- **5-Fold GroupKFold cross-validation** partitioned strictly on `username`.
- **Log-Scale HPO Objective**: Optuna Bayesian optimization minimizes log-space MAE:
  $$\text{Loss} = \text{MAE}(\ln(1 + y), \ln(1 + \hat{y}))$$
  This prevents celebrity accounts (~70M reach) from dominating loss over smaller creators.

---

## 📐 Mondrian Conformal Prediction & Epistemic Uncertainty

### 1. Finite-Sample Conformal Quantile Formulation
For significance level $\alpha \in (0, 1)$ over $n$ calibration samples, the empirical quantile is computed using the exact inductive conformal prediction formula:
$$p = \min\left(\frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}, 1.0\right) \quad (\text{with } \text{method} = \text{"higher"})$$

### 2. Calibrated Mondrian Strata Across All 4 Tiers

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MONDRIAN CONFORMAL STRATA                                 │
├──────────────┬────────────────────────┬─────────────┬─────────────────┬────────────────┤
│ Tier         │ Follower Range         │ Samples ($n$)│ Reach Quantiles │ Imp. Quantiles │
├──────────────┼────────────────────────┼─────────────┼─────────────────┼────────────────┤
│ **Nano**     │ $500 - 9,999$          │ $n=36$      │ $q_{80}=0.4987$ │ $q_{80}=0.4858$│
│              │                        │             │ $q_{90}=0.6177$ │ $q_{90}=0.6069$│
├──────────────┼────────────────────────┼─────────────┼─────────────────┼────────────────┤
│ **Micro**    │ $10,000 - 99,999$      │ $n=42$      │ $q_{80}=0.3340$ │ $q_{80}=0.3950$│
│              │                        │             │ $q_{90}=0.5050$ │ $q_{90}=0.5820$│
├──────────────┼────────────────────────┼─────────────┼─────────────────┼────────────────┤
│ **Macro**    │ $100,000 - 999,999$    │ $n=42$      │ $q_{80}=0.2650$ │ $q_{80}=0.2780$│
│              │                        │             │ $q_{90}=0.3680$ │ $q_{90}=0.4200$│
├──────────────┼────────────────────────┼─────────────┼─────────────────┼────────────────┤
│ **Mega**     │ $\ge 1,000,000$        │ $n=30$      │ $q_{80}=0.3120$ │ $q_{80}=0.3750$│
│              │                        │             │ $q_{90}=0.3650$ │ $q_{90}=0.4700$│
└──────────────┴────────────────────────┴─────────────┴─────────────────┴────────────────┘
```

---

## 🛡️ Defensive Guardrails & Anomaly Detection

Located in [`src/instagram_predictor/guardrails/`](file:///d:/Work/Projects/project-instagram-prediction/src/instagram_predictor/guardrails/):

1. **`PlatformSanityGuardrail`**:
   - Following Limit: $\text{Following} \le 7,500$.
   - Bounds Check: Non-negative metrics across all counts.
   - Demographics Check: `gender_female_pct` + `gender_male_pct` $\approx 1.0$.
2. **`MathematicalInvariantGuardrail`**:
   - $\text{Reach} \le \text{Impressions}$.
   - $\text{Likes} \le \text{Impressions}$, $\text{Shares} \le \text{Impressions}$, $\text{Comments} \le \text{Impressions}$.
3. **`BotAnomalyDetector`**:
   - Flags accounts with engagement rates $< 0.05\%$.
   - Flags accounts with `avg_comments == 0 and avg_likes > 100` (purchased/bot likes).
   - Flags like-to-comment ratio spikes $> 500:1$.
4. **`ViralAnomalyDetector`**:
   - Detects viral breakout anomalies where $\text{Reach} > 10 \times \text{Followers}$.
5. **`AdversarialPromptSanitizer`**:
   - Recursive HTML tag stripping loop neutralizes nested bypasses (`<<script>script>`).
   - Zero-width character stripping (`\u200b`, etc.).
   - Prompt injection blacklist (`"disregard previous instructions"`, `"system directive"`).
6. **`CSVFormulaSanitizer`**:
   - Neutralizes CSV formula injection (DDE) by escaping cells starting with `=`, `+`, `-`, or `@`.

---

## 🔍 NLP Natural Language Query Engine

The NLP engine ([`src/instagram_predictor/nlp/parser.py`](file:///d:/Work/Projects/project-instagram-prediction/src/instagram_predictor/nlp/parser.py)):

### Two-Tiered Handle Extraction
- **Explicit Handles**: Always matches `@([a-zA-Z0-9_\.]{3,30})\b`.
- **Structured Prefixes**: Matches `account of`, `handle of`, `creator named`.
- **Preposition Immunity**: Conversational phrases like `"Show accounts for marketing campaigns"` or `"Predict reach for new product"` never falsely extract `"marketing"` or `"new"` as usernames.

### Decimal Follower Multipliers
- `0.5m` / `0.5M` $\rightarrow$ `500,000`
- `1.5m` / `1.5M` $\rightarrow$ `1,500,000`
- `2.5k` / `2.5K` $\rightarrow$ `2,500`
- `1.2b` / `1.2B` $\rightarrow$ `1,200,000,000`

---

## 💻 Streamlit Enterprise Dashboard (`app.py`)

Run the dashboard via:
```powershell
uv run streamlit run app.py
```

### Performance & Caching Architecture
- `@st.cache_data`: Caches the feature-engineered dataset and metadata.
- `@st.cache_resource`: Caches model pipelines across user sessions, preventing redundant disk reads.
- **What-If Post Performance Simulator (Tab 2)**:
  - Creative sliders (caption length, hashtags, CTA, media format, styles, schedule) dynamically modulate projected engagement and reach.
  - Generates Mondrian conformal confidence intervals ($80\%$ and $90\%$).

---

## 🛠️ CLI Scripts & Execution Guide

### 1. Synthetic Data Generation
Generates the 47-column dataset:
```powershell
uv run python -m instagram_predictor.data.generator
```

### 2. Model Retraining & Conformal Calibration
Executes GroupKFold cross-validation, trains LightGBM regressors, calculates Mondrian quantiles, and serializes artifacts to `models/`:
```powershell
uv run python -m instagram_predictor.models.trainer
```

### 3. Hyperparameter Optimization
Runs Optuna Bayesian optimization:
```powershell
uv run python -m instagram_predictor.models.hpo
```

### 4. Run the Streamlit Application
```powershell
uv run streamlit run app.py
```

---

## 🧪 Comprehensive Test Suite (92 Tests)

The test suite consists of **92 automated tests** across 11 test modules:

```powershell
uv run --with pytest python -m pytest tests/ -v
```

### Test Suite Manifest

1. **`test_audit_hardening.py` (17 tests)**:
   - Negative & malformed inputs (empty DataFrames, missing columns, all-NaN rows).
   - Invariant enforcement (`reach <= impressions`, bounds).
   - Epistemic OOD & boundary handling (1B followers, 0 followers, negative values).
   - Security hardening (SHA-256 tampering, regex DoS, nested HTML evasion, CSV DDE).
   - Concurrency & thread-safety under multi-threaded contention.
   - Conformal calibration exact quantiles and Mondrian tier coverage.
   - NLP preposition immunity and decimal multipliers.
   - Simulation creative controls sensitivity.
2. **`test_concurrency.py` (7 tests)**:
   - Multi-threaded stress tests hitting registry and loader caches simultaneously.
3. **`test_conformal_mondrian.py` (10 tests)**:
   - Exact finite-sample quantile formula verification, Nano tier calibration, HPO log-loss objective.
4. **`test_end_to_end.py` (10 tests)**:
   - Legacy end-to-end scenarios, follower unit scaling, and comparison operators.
5. **`test_end_to_end_service.py` (2 tests)**:
   - Analytics service and post simulation service workflows.
6. **`test_feature_engineering.py` (6 tests)**:
   - Feature generation, Nano tier coverage, and robust multi-label list parsing.
7. **`test_guardrails.py` (11 tests)**:
   - Platform sanity checks, invariant enforcement on shares/comments, zero-comment anomaly detection.
8. **`test_models.py` (6 tests)**:
   - Pipeline deserialization, creative parameter sensitivity, extreme numerical bounds.
9. **`test_query_parser.py` (6 tests)**:
   - Handle extraction disambiguation, decimal multipliers, and operator parsing.
10. **`test_schemas.py` (7 tests)**:
    - Pydantic v2 validation boundaries, platform limits, caption length, and gender sum validation.
11. **`test_security.py` (10 tests)**:
    - SHA-256 artifact integrity, regex metacharacter safety, prompt injection defense, CSV formula escaping.

---

## ⚙️ Installation & Build Configuration

### Prerequisites
- Python $\ge 3.10$ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/palaksharmaaaaa/instagram_prediction.git
   cd project-instagram-prediction
   ```

2. **Set Up Virtual Environment & Dependencies using `uv`**:
   ```powershell
   uv venv
   uv pip install -e ".[dev]"
   ```

3. **Verify Installation**:
   ```powershell
   uv run --with pytest python -m pytest tests/ -v
   ```

4. **Launch Application**:
   ```powershell
   uv run streamlit run app.py
   ```

---

## 📄 License & Maintainers

- **License**: MIT Enterprise Open Source License
- **Maintainers**: Instagram AI Prediction Engine Team
