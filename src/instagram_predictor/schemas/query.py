from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class NumericFilter(BaseModel):
    operator: Literal[">", ">=", "<", "<=", "==", "!=", "between"]
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class TopNSpec(BaseModel):
    limit: int = Field(default=10, ge=1, le=200)
    sort_by: str = Field(default="total_followers")
    ascending: bool = Field(default=False)


class QueryIntent(BaseModel):
    primary_goal: str = Field(default="Exploratory / Discovery", description="Primary strategic intent of the query")
    creator_tier: str = Field(default="Any", description="Detected creator scale (Nano, Micro, Mid, Macro, Mega)")
    audience_focus: str = Field(default="Broad", description="Target demographic or demographic segment")
    tone: str = Field(default="Informational", description="Query tone (Analytical, Commercial, Casual)")


class AuditReport(BaseModel):
    clarity_score: int = Field(default=80, ge=0, le=100, description="Clarity score from 0 to 100")
    detected_intents: List[str] = Field(default_factory=list)
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    refinement_suggestions: List[str] = Field(default_factory=list)
    audit_feedback: str = Field(default="", description="Executive feedback on prompt specificity and quality")


class ParsedQuery(BaseModel):
    filters: Dict[str, Any] = Field(default_factory=dict)
    predict_reach: bool = False
    predict_impressions: bool = False
    original_prompt: str = ""
    sanitized_prompt: str = ""
    safety_flags: List[str] = Field(default_factory=list)
    intent: Optional[QueryIntent] = None
    audit_report: Optional[AuditReport] = None

    @property
    def username(self) -> Optional[str]:
        return self.filters.get("username")

    @property
    def min_followers(self) -> Optional[float]:
        tf = self.filters.get("total_followers")
        if not tf or not isinstance(tf, dict):
            return None
        if tf.get("operator") == "between":
            v = tf.get("min")
            return int(v) if v is not None and isinstance(v, (int, float)) and float(v).is_integer() else v
        if tf.get("operator") in (">", ">="):
            v = tf.get("value")
            return int(v) if v is not None and isinstance(v, (int, float)) and float(v).is_integer() else v
        return None

    @property
    def max_followers(self) -> Optional[float]:
        tf = self.filters.get("total_followers")
        if not tf or not isinstance(tf, dict):
            return None
        if tf.get("operator") == "between":
            v = tf.get("max")
            return int(v) if v is not None and isinstance(v, (int, float)) and float(v).is_integer() else v
        if tf.get("operator") in ("<", "<="):
            v = tf.get("value")
            return int(v) if v is not None and isinstance(v, (int, float)) and float(v).is_integer() else v
        return None
