from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    """
    Result of parsing a natural-language search. `understood` lists exactly what was turned
    into a filter; `warnings` lists everything that looked like a constraint but was NOT applied,
    so a partially understood query never silently returns unfiltered results.
    """
    filters: Dict[str, Any] = Field(default_factory=dict)
    understood: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    original_prompt: str = ""
    safety_flags: List[str] = Field(default_factory=list)
