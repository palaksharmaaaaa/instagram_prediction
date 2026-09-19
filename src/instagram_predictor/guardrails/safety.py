import re
from typing import List, Tuple
from ..config import settings


INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|all)\s+instructions?",
    r"system\s+prompt",
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
    malicious scripts, and length overflow.
    """
    if not raw_prompt:
        return "", []

    flags: List[str] = []
    text = raw_prompt.strip()

    # 1. Length guardrail
    if len(text) > settings.MAX_PROMPT_LENGTH:
        text = text[: settings.MAX_PROMPT_LENGTH]
        flags.append("PROMPT_TRUNCATED_MAX_LENGTH")

    # 2. HTML / Script tag detection and stripping
    if re.search(r"<[^>]+>", text):
        flags.append("HTML_TAGS_STRIPPED")
        text = re.sub(r"<[^>]+>", "", text)

    # 3. Injection pattern checks
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("POTENTIAL_INJECTION_DETECTED")
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)

    # Deduplicate flags
    unique_flags = list(dict.fromkeys(flags))
    return text.strip(), unique_flags
