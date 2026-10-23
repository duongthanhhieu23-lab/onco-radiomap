from __future__ import annotations

from typing import Any

def _eligible(fact: dict[str, Any]) -> bool:
    source_signature = f"{fact.get('source_type', '')} {fact.get('record_locator', '')}".casefold()
    if fact.get('hard_exclusion'):
        return False
    if fact.get('atomic_review', {}).get('audited'):
        return fact['atomic_review'].get('relation_eligible') is True
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
    if fact.get('attribution_status') == 'needs_review':
        return None
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
    # Resolved IDs are extraction proposals, not permission to discard a
    # different candidate before an arbiter has inspected the complete source.
    if fact.get('atomic_review', {}).get('audited'):
        return None
    tumor_candidates = set(fact.get("tumor_candidate_ids", []))
    radio_candidates = set(fact.get("radiopharmaceutical_candidate_ids", []))
    if (
        tumor_subtype_id in tumor_candidates
        and radiopharmaceutical_id in radio_candidates
    ):
        return "ambiguous_candidate_pair_for_llm"
    return None
