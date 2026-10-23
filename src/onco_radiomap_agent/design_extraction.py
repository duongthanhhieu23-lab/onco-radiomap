"""Shared high-specificity study-design cues used by reference and prediction facts."""

from __future__ import annotations

import re


def explicit_controlled_design(text: str) -> bool:
    """Return true only for an explicit study-level control design."""

    return bool(
        re.search(
            r"\b(?:randomi[sz]ed\s+controlled|controlled\s+(?:clinical\s+)?"
            r"(?:trial|study)|placebo[- ]controlled)\b|"
            r"\b(?:control|comparator)\s+(?:arm|group|cohort)\b|"
            r"\b(?:arm|group|cohort)\s+(?:served\s+as\s+)?(?:the\s+)?control\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def credible_treatment_comparator(text: str) -> bool:
    """Detect an intervention comparator, not a prognostic subgroup contrast."""

    randomized = bool(re.search(r"\brandomi[sz](?:ed|ation)\b", text, re.IGNORECASE))
    if randomized or explicit_controlled_design(text):
        return True
    treatment_term = (
        r"(?:therapy|treatment|monotherapy|combination|regimen|prrt|rlt|"
        r"radioligand\s+therapy|radionuclide\s+therapy)"
    )
    return bool(
        re.search(
            rf"\b{treatment_term}\b.{{0,100}}"
            rf"\b(?:versus|vs\.?|compared\s+with)\b.{{0,100}}"
            rf"\b{treatment_term}\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        or re.search(
            rf"\b(?:patients?|participants?)\s+(?:receiving|treated\s+with)\s+"
            rf"[^.;:]{{1,100}}\b(?:versus|vs\.?|compared\s+with)\b"
            rf"[^.;:]{{1,100}}\b{treatment_term}\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def single_arm_defined_cohort(text: str) -> bool:
    """Detect an explicit single arm or a non-comparative early-phase cohort."""

    if re.search(r"\bsingle[- ]arm\b", text, re.IGNORECASE):
        return True
    early_phase = bool(
        re.search(r"\bphase\s+(?:i|1|ii|2)\b", text, re.IGNORECASE)
    )
    return bool(
        early_phase
        and not credible_treatment_comparator(text)
        and not explicit_controlled_design(text)
        and not re.search(r"\brandomi[sz](?:ed|ation)\b", text, re.IGNORECASE)
    )
