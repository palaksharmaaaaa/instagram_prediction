import html
import re
from typing import Any, List, Tuple
import pandas as pd
from ..config import settings


INJECTION_PATTERNS = [
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff]",
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:previous|all|prior|above)\s+instructions?",
    r"system\s+(?:prompt|directive|message|role)",
    r"javascript\s*:",
    r"drop\s+table",
    r"delete\s+from",
    r"exec\s*\(",
    r"eval\s*\(",
    r"union\s+select"
]


def sanitize_prompt(raw_prompt: str) -> Tuple[str, List[str]]:
    """
    Sanitizes natural language prompt inputs against prompt injection,
    malicious scripts, nested tags, zero-width evasion, and length overflow.
    """
    if not raw_prompt:
        return "", []

    flags: List[str] = []
    text = raw_prompt.strip()

    # 1. Length guardrail
    if len(text) > settings.MAX_PROMPT_LENGTH:
        text = text[: settings.MAX_PROMPT_LENGTH]
        flags.append("PROMPT_TRUNCATED_MAX_LENGTH")

    # 2. HTML / Script tag detection and recursive stripping
    if re.search(r"<[^>]+>", text):
        flags.append("HTML_TAGS_STRIPPED")
        prev = None
        while prev != text and re.search(r"<[^>]+>", text):
            prev = text
            text = re.sub(r"<[^>]+>", "", text)

    # Neutralize residual HTML characters
    text = html.escape(text)

    # 3. Injection pattern checks
    # Check for zero-width characters first to prevent evasion
    if re.search(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", text):
        flags.append("POTENTIAL_INJECTION_DETECTED")
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("POTENTIAL_INJECTION_DETECTED")
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)

    # Deduplicate flags
    unique_flags = list(dict.fromkeys(flags))
    return text.strip(), unique_flags


# Alias for query input sanitization
sanitize_query_input = sanitize_prompt


def sanitize_csv_cell(val: Any) -> Any:
    """
    Sanitizes a cell value against CSV / Spreadsheet Formula Injection (DDE).
    If the value is a string starting with '=', '+', '-', or '@',
    prepends a single quote "'" to neutralize formula execution in Excel/Sheets.
    """
    if isinstance(val, str) and val.startswith(("=", "+", "-", "@")):
        return f"'{val}"
    return val


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the DataFrame with all text cells starting with '=', '+', '-', or '@'
    sanitized by prepending a single quote "'" to prevent spreadsheet formula execution.
    Numeric columns and non-string values are preserved.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object or isinstance(df_clean[col].dtype, pd.StringDtype):
            df_clean[col] = df_clean[col].apply(sanitize_csv_cell)
    return df_clean
