"""Independent full-system rules over reference-free prediction facts.

This module intentionally does not import ``reference_labels`` and never reads
``cases.jsonl`` answer fields. It applies the public Phase-3 rules only to
prediction-side facts extracted from raw source text and shared candidates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from typing import Any

_RANK = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "T1": 1, "T2": 2, "T3": 3, "T4": 4}


def population_scope_compatible(fact_scope: object, query_scope: str) -> bool:
    """Apply the frozen adult-primary, pediatric-isolated population policy."""

    fact_value = str(fact_scope or "not_reported")
    if query_scope == "not_reported":
        # ``not_reported`` is an unstratified query, not a pediatric label.
        # Adult-indexed evidence remains in the main analysis, while pediatric
        # and mixed evidence stay separately evaluated.
        return fact_value in {"not_reported", "adult"}
    return fact_value == query_scope


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    parts = str(value).split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (TypeError, ValueError):
        return None


def _eligible(fact: dict[str, Any]) -> bool:
    source_signature = f"{fact.get('source_type', '')} {fact.get('record_locator', '')}".casefold()
    return bool(
        fact.get("relation_decision_status") == "clear_eligible"
        and fact.get("source_role") == "primary_human_result"
        and "clinicaltrials.gov" not in source_signature
        and "clinicaltrials_registration" not in source_signature
        and fact.get("human_subjects")
        and fact.get("oncology_context")
        and fact.get("result_reported")
        and fact.get("clinical_use_alignment")
        and not fact.get("retracted")
    )


def query_pair_match_basis(
    fact: dict[str, Any],
    *,
    tumor_subtype_id: str,
    radiopharmaceutical_id: str,
) -> str | None:
    """Return a direct-aggregation basis without forming a Cartesian pair."""

    resolved_tumor = fact.get("tumor_subtype_id")
    resolved_radio = fact.get("radiopharmaceutical_id")
    if resolved_tumor is not None and resolved_tumor != tumor_subtype_id:
        return None
    if resolved_radio is not None and resolved_radio != radiopharmaceutical_id:
        return None
    tumor_candidates = set(fact.get("tumor_candidate_ids", []))
    radio_candidates = set(fact.get("radiopharmaceutical_candidate_ids", []))
    if resolved_tumor is not None and resolved_radio is not None:
        return (
            "resolved_pair"
            if resolved_tumor == tumor_subtype_id
            and resolved_radio == radiopharmaceutical_id
            else None
        )
    if (
        len(tumor_candidates) == 1
        and len(radio_candidates) == 1
        and tumor_subtype_id in tumor_candidates
        and radiopharmaceutical_id in radio_candidates
    ):
        return "unique_candidate_pair"
    for binding in fact.get("candidate_pair_bindings", []):
        if (
            binding.get("tumor_subtype_id") == tumor_subtype_id
            and binding.get("radiopharmaceutical_id") == radiopharmaceutical_id
        ):
            return str(binding.get("pair_match_basis") or "unique_same_segment_pair")
    return None


def query_pair_candidate_status(
    fact: dict[str, Any],
    *,
    tumor_subtype_id: str,
    radiopharmaceutical_id: str,
) -> str | None:
    """Classify a query pair as direct, ambiguous-for-LLM, or absent."""

    direct = query_pair_match_basis(
        fact,
        tumor_subtype_id=tumor_subtype_id,
        radiopharmaceutical_id=radiopharmaceutical_id,
    )
    if direct is not None:
        return direct
    resolved_tumor = fact.get("tumor_subtype_id")
    resolved_radio = fact.get("radiopharmaceutical_id")
    if resolved_tumor is not None and resolved_tumor != tumor_subtype_id:
        return None
    if resolved_radio is not None and resolved_radio != radiopharmaceutical_id:
        return None
    tumor_candidates = set(fact.get("tumor_candidate_ids", []))
    radio_candidates = set(fact.get("radiopharmaceutical_candidate_ids", []))
    if (
        tumor_subtype_id in tumor_candidates
        and radiopharmaceutical_id in radio_candidates
    ):
        return "ambiguous_candidate_pair_for_llm"
    return None


def _atomic_conclusion(fact: dict[str, Any]) -> str:
    result = fact.get("result", {})
    if not result.get("adequately_interpretable") or result.get(
        "materially_imprecise_or_internally_mixed"
    ):
        return "inconclusive"
    objective = result.get("primary_objective_status")
    if objective == "met":
        return "supportive"
    if objective == "not_met":
        return "negative"
    direction = result.get("reported_effect_direction")
    if direction == "favorable":
        return "supportive"
    if direction == "unfavorable":
        return "negative"
    return "inconclusive"


def _maturity(fact: dict[str, Any]) -> str:
    design = fact.get("design", {})
    if fact.get("clinical_use") == "diagnostic":
        performance = any(
            design.get(field)
            for field in (
                "prespecified_diagnostic_performance",
                "gold_standard_comparison",
                "head_to_head_tracer_comparison",
                "management_impact_evaluated",
            )
        )
        if design.get("prospective") and design.get("multicenter") and (
            design.get("management_impact_evaluated")
            or design.get("patient_relevant_diagnostic_utility_endpoint")
        ):
            return "D4"
        if design.get("prospective") and design.get("multicenter") and performance:
            return "D3"
        return "D2" if performance else "D1"
    if (
        design.get("phase_iii")
        and design.get("controlled")
        and design.get("patient_relevant_treatment_endpoint")
    ):
        return "T4"
    if (
        design.get("randomized")
        or design.get("credible_comparator")
        or design.get("controlled")
    ) and design.get("prespecified_objective_efficacy"):
        return "T3"
    if (
        design.get("prespecified_objective_efficacy")
        and (design.get("prospective") or design.get("single_arm_defined_cohort"))
        and (
            design.get("safety_reported")
            or design.get("biodistribution_or_dosimetry_reported")
        )
    ):
        return "T2"
    return "T1"


def predict_atomic(fact: dict[str, Any]) -> dict[str, Any]:
    relation_status = str(fact.get("relation_decision_status", "needs_arbiter"))
    eligible: bool | None
    if relation_status == "clear_eligible":
        eligible = _eligible(fact)
    elif relation_status == "clear_ineligible":
        eligible = False
    else:
        eligible = None
    missing_fields = list(fact.get("missing_or_ambiguous_fields", []))
    classification_missing = [
        field
        for field in missing_fields
        if field not in {"tumor_subtype_id", "radiopharmaceutical_id"}
    ]
    return {
        "predicted_tumor_subtype_id": fact.get("tumor_subtype_id"),
        "predicted_radiopharmaceutical_id": fact.get("radiopharmaceutical_id"),
        "predicted_relation_eligible": eligible,
        "predicted_atomic_conclusion": _atomic_conclusion(fact) if eligible is True else None,
        "maturity_code": _maturity(fact) if eligible is True else None,
        "rule_decision_status": (
            "needs_arbiter_missing_fields"
            if relation_status == "needs_arbiter"
            else "clear"
        ),
        "rule_reason_codes": (
            [f"missing:{field}" for field in classification_missing]
            if relation_status == "needs_arbiter"
            else (
                list(fact.get("explicit_ineligibility_reasons", []))
                if relation_status == "clear_ineligible"
                else ["reference_free_prediction_fact_rule"]
            )
        ),
        "supporting_fact_ids": [str(fact.get("fact_id"))],
        "supporting_source_ids": [str(fact.get("source_id"))],
        "supporting_locators": [str(fact.get("record_locator", ""))],
    }


def _merge_design(members: list[dict[str, Any]]) -> dict[str, bool]:
    fields = {field for member in members for field in member.get("design", {})}
    return {
        field: any(member.get("design", {}).get(field) for member in members)
        for field in fields
    }


def _family_rows(facts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        groups[str(fact["study_family_id"])].append(fact)
    families = []
    for family_id, members in sorted(groups.items()):
        conclusions = {_atomic_conclusion(member) for member in members}
        conclusion = conclusions.pop() if len(conclusions) == 1 else "inconclusive"
        maturity = max((_maturity(member) for member in members), key=_RANK.__getitem__)
        families.append(
            {
                "study_family_id": family_id,
                "conclusion": conclusion,
                "maturity": maturity,
                "design": _merge_design(members),
            }
        )
    return families


def _snapshot_maturity(clinical_use: str, families: list[dict[str, Any]]) -> str:
    maturity = max((family["maturity"] for family in families), key=_RANK.__getitem__)
    if clinical_use == "diagnostic" and _RANK[maturity] < 3:
        performance_families = sum(
            any(
                family["design"].get(field)
                for field in (
                    "prespecified_diagnostic_performance",
                    "gold_standard_comparison",
                    "head_to_head_tracer_comparison",
                    "management_impact_evaluated",
                )
            )
            for family in families
        )
        if performance_families >= 2:
            return "D3"
    if clinical_use == "therapeutic":
        if _RANK[maturity] < 4:
            controlled = sum(
                bool(
                    family["design"].get("controlled")
                    and family["design"].get("patient_relevant_treatment_endpoint")
                )
                for family in families
            )
            if controlled >= 2:
                return "T4"
        if _RANK[maturity] < 3:
            prospective = sum(
                bool(
                    family["design"].get("prospective")
                    and family["design"].get("prespecified_objective_efficacy")
                )
                for family in families
            )
            if prospective >= 2:
                return "T3"
    return maturity


def _snapshot_conclusion(families: list[dict[str, Any]]) -> str:
    directional = [
        family
        for family in families
        if family["conclusion"] in {"supportive", "negative"}
    ]
    if not directional:
        return "inconclusive"
    supportive = [family for family in directional if family["conclusion"] == "supportive"]
    negative = [family for family in directional if family["conclusion"] == "negative"]
    if supportive and negative:
        comparable = any(
            abs(_RANK[left["maturity"]] - _RANK[right["maturity"]]) <= 1
            and left["study_family_id"] != right["study_family_id"]
            for left in supportive
            for right in negative
        )
        if comparable:
            return "conflicting"
        highest_rank = max(_RANK[family["maturity"]] for family in directional)
        highest = {
            family["conclusion"]
            for family in directional
            if _RANK[family["maturity"]] == highest_rank
        }
        return "conflicting" if len(highest) > 1 else next(iter(highest))
    return "supportive" if supportive else "negative"


def predict_snapshot(
    facts: Iterable[dict[str, Any]],
    *,
    allowed_fact_ids: set[str],
    dataset_id: str,
    tumor_subtype_id: str,
    radiopharmaceutical_id: str,
    clinical_use: str,
    population_scope: str,
    cutoff_date: str,
) -> dict[str, Any]:
    cutoff = _parse_date(cutoff_date)
    selected: list[dict[str, Any]] = []
    match_basis_by_fact_id: dict[str, str] = {}
    for fact in facts:
        match_basis = query_pair_match_basis(
            fact,
            tumor_subtype_id=tumor_subtype_id,
            radiopharmaceutical_id=radiopharmaceutical_id,
        )
        if not (
            fact.get("fact_id") in allowed_fact_ids
            and fact.get("origin_dataset_id") == dataset_id
            and match_basis is not None
            and fact.get("clinical_use") == clinical_use
            and population_scope_compatible(
                fact.get("applicability", {}).get("population_scope"),
                population_scope,
            )
            and _eligible(fact)
            and cutoff is not None
            and (fact_date := _parse_date(fact.get("evidence_public_date")))
            is not None
            and fact_date <= cutoff
        ):
            continue
        selected.append(fact)
        match_basis_by_fact_id[str(fact["fact_id"])] = match_basis
    if not selected:
        return {
            "predicted_tumor_subtype_id": tumor_subtype_id,
            "predicted_radiopharmaceutical_id": radiopharmaceutical_id,
            "predicted_maturity_code": None,
            "predicted_evidence_conclusion": None,
            "supporting_fact_count": 0,
            "supporting_family_count": 0,
            "supporting_fact_ids": [],
            "supporting_source_ids": [],
            "supporting_locators": [],
            "supporting_study_family_ids": [],
            "supporting_source_family_pairs": [],
            "supporting_pair_match_basis": {},
            "rule_decision_status": "abstain_no_eligible_evidence",
            "rule_reason_codes": ["no_eligible_evidence_before_cutoff"],
        }
    families = _family_rows(selected)
    conclusion = _snapshot_conclusion(families)
    return {
        "predicted_tumor_subtype_id": tumor_subtype_id,
        "predicted_radiopharmaceutical_id": radiopharmaceutical_id,
        "predicted_maturity_code": _snapshot_maturity(clinical_use, families),
        "predicted_evidence_conclusion": conclusion,
        "supporting_fact_count": len(selected),
        "supporting_family_count": len(families),
        "supporting_fact_ids": sorted(str(fact["fact_id"]) for fact in selected),
        "supporting_source_ids": sorted(
            {str(fact.get("source_id")) for fact in selected if fact.get("source_id")}
        ),
        "supporting_locators": sorted(
            {
                str(fact.get("record_locator"))
                for fact in selected
                if fact.get("record_locator")
            }
        ),
        "supporting_study_family_ids": sorted(
            {str(fact["study_family_id"]) for fact in selected}
        ),
        "supporting_source_family_pairs": sorted(
            (
                {
                    "source_id": str(fact["source_id"]),
                    "study_family_id": str(fact["study_family_id"]),
                    "record_locator": str(fact.get("record_locator", "")),
                }
                for fact in selected
                if fact.get("source_id") and fact.get("study_family_id")
            ),
            key=lambda row: (
                row["study_family_id"],
                row["source_id"],
                row["record_locator"],
            ),
        ),
        "supporting_pair_match_basis": dict(sorted(match_basis_by_fact_id.items())),
        "rule_decision_status": (
            "needs_arbiter_independent_study_conflict"
            if conclusion == "conflicting"
            else "clear"
        ),
        "rule_reason_codes": (
            ["independent_study_conflict"]
            if conclusion == "conflicting"
            else ["eligible_study_family_aggregation"]
        ),
    }
