from typing import List, Optional
from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    lower: int = Field(ge=0)
    point_estimate: int = Field(ge=0)
    upper: int = Field(ge=0)
    confidence_level: float = Field(default=0.80)


class SimulationPrediction(BaseModel):
    projected_reach: ConfidenceInterval
    projected_impressions: ConfidenceInterval
    projected_engagement_rate: float = Field(description="Projected engagement rate percentage")
    projected_save_rate: float = Field(description="Projected save rate percentage")
    projected_share_rate: float = Field(description="Projected share rate percentage")
    virality_score: float = Field(description="Share to like virality index")
    virality_tier: str = Field(description="Viral tier classification")
    calibration_tier: str = Field(default="Standard", description="Mondrian conformal calibration follower tier")
    uncertainty_rating: str = Field(default="Calibrated (High Confidence)", description="Epistemic uncertainty and in-distribution assessment")
    prediction_interval_coverage: str = Field(default="80% Conformal Coverage Guarantee", description="Mathematical finite-sample coverage guarantee")
    optimization_tips: List[str] = Field(default_factory=list)


class MetricCard(BaseModel):
    title: str
    value: str
    delta: Optional[str] = None
    help_text: Optional[str] = None
