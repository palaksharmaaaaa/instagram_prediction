from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    lower: int = Field(ge=0)
    point_estimate: int = Field(ge=0)
    upper: int = Field(ge=0)
    confidence_level: float
    calibration: str = Field(description="How the interval width was calibrated (e.g. 'follower tier: micro')")


class Forecast(BaseModel):
    """
    A model forecast plus everything needed to judge it. Every field is either
    computed from the trained model / its held-out evaluation or copied from the input.
    """
    reach_80: ConfidenceInterval
    reach_90: ConfidenceInterval
    impressions_80: Optional[ConfidenceInterval] = None
    impressions_90: Optional[ConfidenceInterval] = None

    model_name: str
    model_is_baseline: bool
    n_train_posts: int
    n_train_creators: int
    heldout_median_abs_pct_error: float = Field(description="Median |error| of this model on held-out creators, as a fraction")
    empirical_coverage_80: Optional[float] = Field(default=None, description="Measured coverage of the 80% interval on held-out creators")
    empirical_coverage_90: Optional[float] = None

    imputed_fields: List[str] = Field(default_factory=list, description="Model inputs that were not provided")
    extrapolated_fields: List[str] = Field(default_factory=list, description="Inputs outside the range seen in training")
    notes: List[str] = Field(default_factory=list)


class MetricCard(BaseModel):
    title: str
    value: str
    delta: Optional[str] = None
    help_text: Optional[str] = None
