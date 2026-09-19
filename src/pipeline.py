"""
Legacy Pipeline Module (v1 Prototype Architecture)
==================================================
DEPRECATION NOTICE:
This module represents the v1 prototype pipeline and is maintained solely for
backward compatibility with legacy test suites (e.g., tests/test_end_to_end.py).

Production systems should use the enterprise-grade `instagram_predictor` package:
- Analytics pipeline: `instagram_predictor.services.run_analytics_pipeline`
- Post simulation: `instagram_predictor.services.run_post_simulation`
- Model inference: `instagram_predictor.models.predict_batch`
"""

import os
import pandas as pd

try:
    from preprocessing import load_data
    from prompt_parser import parse_prompt
    from predictor import apply_filter, predict
except ModuleNotFoundError:
    from src.preprocessing import load_data
    from src.prompt_parser import parse_prompt
    from src.predictor import apply_filter, predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "top_200_instagrammers.csv")


def run_pipeline(prompt, dataset_path=None):
    """
    Executes the end-to-end natural language analytics and prediction pipeline:
    1. Loads dataset
    2. Parses user prompt into structured requirements
    3. Applies filters (category, country, handle, numeric thresholds, top-n)
    4. Runs selective prediction models if requested
    """
    path = dataset_path or DATA_PATH
    df = load_data(path)

    requirements = parse_prompt(prompt)
    filters = requirements.get("filters", {})
    predict_reach = requirements.get("predict_reach", False)
    predict_impressions = requirements.get("predict_impressions", False)

    filtered_df = apply_filter(df, filters)

    if predict_reach or predict_impressions:
        if not filtered_df.empty:
            filtered_df = predict(
                filtered_df,
                predict_reach=predict_reach,
                predict_impressions=predict_impressions
            )

    return requirements, filtered_df