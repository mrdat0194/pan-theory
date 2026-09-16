"""
Template router: maps a chemistry query to the appropriate fallback module.

The matching is keyword-based and case-insensitive. It is only invoked
after all LLM self-correction retries are exhausted.
"""

from typing import Optional
from dataclasses import dataclass

from app.generator.templates import ph_scale, covalent_bonding, ionic_vs_covalent


@dataclass
class FallbackScene:
    topic: str
    title: str
    narration_script: str
    animation_code: str


# Ordered from most-specific to least-specific to avoid false positives
_REGISTRY: list[tuple[list[str], object]] = [
    (["ionic", "difference", "ionic and covalent"], ionic_vs_covalent),
    (["covalent", "why do atoms", "atoms form covalent"], covalent_bonding),
    (["ph scale", "ph", "acid", "base", "acidic", "basic"], ph_scale),
]


def resolve_fallback(query: str) -> Optional[FallbackScene]:
    """Return the best-matching fallback scene for *query*, or None.

    Matching is keyword-based and case-insensitive. The registry is evaluated
    in order; the first match wins.
    """
    q = query.lower()
    for keywords, module in _REGISTRY:
        if any(kw in q for kw in keywords):
            return FallbackScene(
                topic=module.TOPIC,
                title=module.TITLE,
                narration_script=module.NARRATION_SCRIPT,
                animation_code=module.ANIMATION_CODE,
            )
    return None
