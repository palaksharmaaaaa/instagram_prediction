"""
Strict, deterministic query parser.

It turns explicit constraints into filters and reports everything else. If a phrase looks like a
constraint but cannot be applied (or refers to data that does not exist), a warning is emitted, so a
partially understood query never silently returns unfiltered results. No intent is guessed.

Supported (all applied conjunctively):
  followers:        "over 1m followers", "followers between 10k and 50k", "at most 500k followers", "5m+ followers"
  engagement rate:  "engagement above 2%", "engagement rate between 1% and 3%"
  average likes:    "avg likes over 1m", "average likes at least 500k"
  handle:           "@username"
  category text:    "category sports" / "in the music category"   (substring match on the reported category)
  ranking:          "top 10", "top 5 by engagement", "top 20 by average likes"
"""

import html
import re
from typing import Any, Dict, List, Tuple

from ..config import settings
from ..guardrails.safety import sanitize_prompt
from ..schemas import ParsedQuery

_NUM = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_UNIT = r"(k|thousand|m|mil|million|b|bil|billion)?"
_OPS = [
    (r"more than|greater than|higher than|above|over|>", ">"),
    (r"at least|minimum|min|>=", ">="),
    (r"less than|fewer than|lower than|below|under|<", "<"),
    (r"at most|maximum|max|up to|<=", "<="),
]
_OP_RE = "(" + "|".join(p for p, _ in _OPS) + ")"
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bil": 1e9, "billion": 1e9}

# Concepts users ask for that this dataset does not contain. Reported, never guessed.
_UNSUPPORTED = ["country", "countries", "gender", "female", "male", "age group", "audience", "morning", "evening",
                "afternoon", "night", "weekend", "weekday", "last month", "this year", "reach", "impressions",
                "saves", "shares", "virality", "viral"]


_TOPICS = ("sports|music|fashion|beauty|gaming|comedy|entertainment|movies|fitness|food|travel|tech|technology|"
           "education|business|finance|dance|animals|cars|art|science|news|kids")

_COUNTRIES = (
    "india|usa|america|united states|united kingdom|uk|england|britain|canada|australia|germany|france|spain|italy|"
    "brazil|mexico|japan|china|korea|russia|turkey|indonesia|nigeria|egypt|argentina|colombia|pakistan|bangladesh|"
    "philippines|vietnam|thailand|saudi arabia|uae|dubai|netherlands|sweden|norway|poland|ukraine|south africa|"
    "new zealand|ireland|portugal|chile|peru|kenya|morocco|iran|israel|greece|belgium|switzerland|austria|denmark|finland"
)


def _op(word: str) -> str:
    w = html.unescape(word.strip().lower())
    for pat, sym in _OPS:
        if re.fullmatch(pat, w):
            return sym
    raise ValueError(word)


def _num(txt: str, unit: str) -> float:
    return float(txt.replace(",", "")) * _MULT.get((unit or "").lower(), 1.0)


def _blank(text: str, span: Tuple[int, int]) -> str:
    return text[:span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def parse_query(prompt: str) -> ParsedQuery:
    _, safety_flags = sanitize_prompt(prompt)          # flags only; parsing uses the user's own text
    text = html.unescape((prompt or "").strip())[: settings.MAX_PROMPT_LENGTH].lower()
    residual = text
    filters: Dict[str, Any] = {}
    understood: List[str] = []
    warnings: List[str] = []

    def add_cond(key: str, cond: Dict[str, Any], label: str) -> None:
        filters.setdefault(key, []).append(cond)
        understood.append(label)

    # ---- followers: between
    for m in list(re.finditer(rf"followers?\s*(?:between|from)\s*{_NUM}\s*{_UNIT}\s*(?:and|to)\s*{_NUM}\s*{_UNIT}"
                              rf"|(?:between|from)\s*{_NUM}\s*{_UNIT}\s*(?:and|to)\s*{_NUM}\s*{_UNIT}\s*followers?", residual)):
        g = m.groups()
        a, ua, b, ub = (g[0], g[1], g[2], g[3]) if g[0] else (g[4], g[5], g[6], g[7])
        lo, hi = _num(a, ua or ub), _num(b, ub)
        lo, hi = min(lo, hi), max(lo, hi)
        add_cond("total_followers", {"operator": ">=", "value": lo}, f"followers >= {lo:,.0f}")
        add_cond("total_followers", {"operator": "<=", "value": hi}, f"followers <= {hi:,.0f}")
        residual = _blank(residual, m.span())

    # ---- followers: "(op) N unit followers" and "followers (op) N unit"
    for pat in (rf"{_OP_RE}\s*{_NUM}\s*{_UNIT}\s*followers?", rf"followers?\s*{_OP_RE}\s*{_NUM}\s*{_UNIT}"):
        for m in list(re.finditer(pat, residual)):
            op, n, u = m.group(1), m.group(2), m.group(3)
            v = _num(n, u)
            add_cond("total_followers", {"operator": _op(op), "value": v}, f"followers {_op(op)} {v:,.0f}")
            residual = _blank(residual, m.span())

    # ---- followers: "5m+ followers", "500k or more followers", "500 followers or more/less"
    for m in list(re.finditer(rf"{_NUM}\s*(k|m|mil|million|thousand|b|bil|billion)?\s*(\+|or more|or above|or higher)\s*followers?"
                              rf"|{_NUM}\s*(k|m|mil|million|thousand|b|bil|billion)?\s*followers?\s*(or more|or above|or higher|\+)", residual)):
        g = m.groups()
        n, u = (g[0], g[1]) if g[0] else (g[3], g[4])
        v = _num(n, u)
        add_cond("total_followers", {"operator": ">=", "value": v}, f"followers >= {v:,.0f}")
        residual = _blank(residual, m.span())
    for m in list(re.finditer(rf"{_NUM}\s*(k|m|mil|million|thousand|b|bil|billion)?\s*followers?\s*(or fewer|or less|or below|or lower)", residual)):
        v = _num(m.group(1), m.group(2))
        add_cond("total_followers", {"operator": "<=", "value": v}, f"followers <= {v:,.0f}")
        residual = _blank(residual, m.span())

    # ---- engagement rate (percent or decimal)
    for m in list(re.finditer(rf"(?:engagement rate|engagement|\ber\b)\s*(?:between|from)\s*{_NUM}\s*%?\s*(?:and|to)\s*{_NUM}\s*%?", residual)):
        a, b = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        if a > 1 or b > 1 or "%" in m.group(0):
            a, b = a / 100, b / 100
        lo, hi = min(a, b), max(a, b)
        add_cond("engagement_rate", {"operator": ">=", "value": lo}, f"engagement rate >= {lo:.2%}")
        add_cond("engagement_rate", {"operator": "<=", "value": hi}, f"engagement rate <= {hi:.2%}")
        residual = _blank(residual, m.span())
    for m in list(re.finditer(rf"(?:engagement rate|engagement|\ber\b)\s*{_OP_RE}\s*{_NUM}\s*(%)?", residual)):
        v = float(m.group(2).replace(",", ""))
        v = v / 100 if (m.group(3) or v > 1) else v
        add_cond("engagement_rate", {"operator": _op(m.group(1)), "value": v}, f"engagement rate {_op(m.group(1))} {v:.2%}")
        residual = _blank(residual, m.span())

    # ---- average likes
    for m in list(re.finditer(rf"(?:avg\.?|average)\s*likes?\s*{_OP_RE}\s*{_NUM}\s*{_UNIT}", residual)):
        v = _num(m.group(2), m.group(3))
        add_cond("avg_likes", {"operator": _op(m.group(1)), "value": v}, f"average likes {_op(m.group(1))} {v:,.0f}")
        residual = _blank(residual, m.span())

    # ---- handle
    m = re.search(r"@([a-z0-9_\.]{2,30})", residual)
    if m:
        filters["username"] = m.group(1).rstrip(".")
        understood.append(f"handle contains '{filters['username']}'")
        residual = _blank(residual, m.span())

    # ---- category text
    m = re.search(r"(?:category|categories)\s+([a-z&][a-z &]{1,30}?)(?=\s+(?:with|and|over|under|above|below|at)\b|$)"
                  r"|(?:in|of)\s+(?:the\s+)?([a-z&][a-z &]{1,30}?)\s+categor(?:y|ies)", residual)
    if m:
        cat = (m.group(1) or m.group(2)).strip()
        filters["category_contains"] = cat
        understood.append(f"category contains '{cat}'")
        residual = _blank(residual, m.span())

    # ---- "<topic> creators": an explicit topic word next to creators/accounts/influencers
    if "category_contains" not in filters:
        m = re.search(rf"\b({_TOPICS})\s+(?:creators?|accounts?|influencers?|pages?)\b", residual)
        if m:
            filters["category_contains"] = m.group(1)
            understood.append(f"category contains '{m.group(1)}'")
            residual = _blank(residual, m.span())

    # ---- ranking
    m = re.search(r"\btop\s+(\d{1,3})\b(?:\s+by\s+(followers|engagement(?: rate)?|avg\.? likes|average likes))?", residual)
    if m:
        lim = int(m.group(1))
        by = (m.group(2) or "followers")
        sort_by = {"followers": "total_followers", "average likes": "avg_likes", "avg likes": "avg_likes", "avg. likes": "avg_likes"}.get(by, "engagement_rate" if by.startswith("engagement") else "total_followers")
        filters["_top_n"] = {"limit": max(1, lim), "sort_by": sort_by, "ascending": False}
        understood.append(f"top {lim} by {sort_by}")
        residual = _blank(residual, m.span())

    # ---- contradictions among numeric conditions
    for key, label in (("total_followers", "followers"), ("engagement_rate", "engagement rate"), ("avg_likes", "average likes")):
        conds = filters.get(key, [])
        lo = max((c["value"] for c in conds if c["operator"] in (">", ">=")), default=None)
        hi = min((c["value"] for c in conds if c["operator"] in ("<", "<=")), default=None)
        if lo is not None and hi is not None and lo > hi:
            warnings.append(f"Contradictory {label} constraints (lower bound {lo:g} > upper bound {hi:g}); no rows can match.")

    # ---- anything that still looks like a constraint is reported, not dropped silently
    leftover = re.sub(r"\s+", " ", residual).strip()
    frag = (r"(?:(?:more than|greater than|higher than|above|over|at least|minimum|less than|fewer than|lower than|below|"
            r"under|at most|maximum|up to|>=|<=|>|<)\s*)?[^\s]*\d[^\s]*(?:\s*(?:million|thousand|billion|mil|bil|k|m|b|%|followers?))?")
    for m in re.finditer(frag, leftover):
        warnings.append(f"Could not interpret '{m.group(0).strip()}' - not applied. Use e.g. 'over 1m followers' or 'engagement above 2%'.")
    for m in re.finditer(rf"\b(?:in|from|based in)\s+({_COUNTRIES})\b|\b({_COUNTRIES})\s+(?:creators?|accounts?|influencers?)\b", leftover):
        place = m.group(1) or m.group(2)
        warnings.append(f"'{place}' cannot be filtered: country is not a field in the creator data. Not applied.")
    for word in _UNSUPPORTED:
        if re.search(rf"\b{re.escape(word)}\b", leftover):
            warnings.append(f"'{word}' cannot be filtered: it is not a field in the creator data. Not applied.")

    return ParsedQuery(filters=filters, understood=understood, warnings=warnings,
                       original_prompt=prompt or "", safety_flags=safety_flags)
