"""Field-level control semantics for the V1.2 rules-first full system."""

from __future__ import annotations

from typing import Any

ENTITY_OVERRIDE_FIELDS = frozenset(
    {
        "predicted_tumor_subtype_id",
        "predicted_radiopharmaceutical_id",
    }
)
OVERRIDE_REASON_CODES = frozenset(
    {
        "entity_conflict",
        "rule_evidence_mismatch",
        "missing_field_resolved",
        "independent_study_conflict",
    }
)


def arbiter_trigger_status(proposal: dict[str, Any]) -> str:
    """Describe why an LLM arbiter is required without changing rule status."""

    allowed = {str(value) for value in proposal.get("allowed_override_fields", [])}
    if not allowed:
        return "not_required"

    entity_fields = allowed.intersection(ENTITY_OVERRIDE_FIELDS)
    classification_fields = allowed.difference(ENTITY_OVERRIDE_FIELDS)
    rule_status = str(proposal.get("rule_decision_status", ""))

    if not classification_fields:
        return "required_entity_resolution"

    if rule_status == "needs_arbiter_independent_study_conflict":
        base = "required_independent_study_conflict"
    elif rule_status in {
        "needs_arbiter_missing_fields",
        "abstain_no_eligible_evidence",
    }:
        base = "required_missing_fields"
    else:
        base = "required_other_classification_review"

    if entity_fields:
        return f"{base}_and_entity_resolution"
    return base


def arbiter_required(proposal: dict[str, Any]) -> bool:
    """Return whether any explicitly permitted field needs arbitration."""

    return arbiter_trigger_status(proposal) != "not_required"


def expected_override_reason(
    proposal: dict[str, Any],
    overridden_fields: set[str],
) -> str:
    """Map one declared arbiter trigger to its only valid reason code."""

    classification_fields = overridden_fields.difference(ENTITY_OVERRIDE_FIELDS)
    rule_status = str(proposal.get("rule_decision_status", ""))
    if overridden_fields and not classification_fields:
        return "entity_conflict"
    if rule_status == "needs_arbiter_independent_study_conflict":
        return "independent_study_conflict"
    if rule_status in {
        "needs_arbiter_missing_fields",
        "abstain_no_eligible_evidence",
    }:
        return "missing_field_resolved"
    return "rule_evidence_mismatch"
