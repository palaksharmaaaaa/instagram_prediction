from typing import Union


def format_number(val: Union[int, float]) -> str:
    """
    Formats raw numbers into human-readable compact representations (e.g. 1.2M, 450.5K).
    """
    if val is None or (isinstance(val, float) and (val != val)):  # NaN check
        return "N/A"
    
    abs_val = abs(val)
    sign = "-" if val < 0 else ""

    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.1f}K"
    return f"{sign}{val:,.0f}" if isinstance(val, int) else f"{sign}{val:.2f}"


def format_percentage(val: Union[int, float], is_ratio: bool = True) -> str:
    """
    Formats ratios or percentages (e.g. 0.015 -> 1.50%).
    """
    if val is None or (isinstance(val, float) and (val != val)):
        return "N/A"
    num = val * 100.0 if is_ratio else val
    return f"{num:.2f}%"
