from __future__ import annotations

import os

import re

from collections import defaultdict

from typing import Any

from onco_radiomap_agent.entity_candidates import candidate_ids

from onco_radiomap_agent.full_system_control import ENTITY_OVERRIDE_FIELDS, arbiter_trigger_status, expected_override_reason

PROMPT_VERSION = "onco-radiomap-baseline-prompt-v1.5"

FULL_SYSTEM_PROMPT_VERSION = "onco-radiomap-full-arbiter-prompt-v1.3-r2-field-audit"

ATOMIC_CONCLUSIONS = {"supportive", "negative", "inconclusive"}

SNAPSHOT_CONCLUSIONS = {"supportive", "negative", "inconclusive", "conflicting"}

def _prompt_version(method: str) -> str:
    return FULL_SYSTEM_PROMPT_VERSION if method == "full_system" else PROMPT_VERSION

def _normalize_llm_prediction(
    case: dict[str, Any],
    raw: dict[str, Any] | None,
    *,
    method: str,
    run_id: int,
    parse_status: str,
    tumor_ids: set[str],
    radio_ids: set[str],
    context_chars: int,
    rule_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw or {}
    parsed_ok = "parsed_attempt_" in parse_status
    requested_status = raw.get("prediction_status")
    if requested_status not in {None, "valid", "abstain"}:
        parsed_ok = False
    requested_status = "valid" if requested_status is None else requested_status
    common = {
        "case_id": case["case_id"],
        "dataset_id": case["dataset_id"],
        "case_type": case["case_type"],
        "clinical_use": case["clinical_use"],
        "method_id": method,
        "run_id": run_id,
        "prediction_provenance": {
            "prompt_version": _prompt_version(method),
            "model": (
                os.getenv("HUANYU_BULK_MODEL")
                or os.getenv("LLM_CHAT_MODEL")
                or os.getenv("OPENAI_CHAT_MODEL")
                or "deepseek-v4-flash"
            ),
            "thinking": "disabled_for_structured_output",
            "parse_status": parse_status,
            "answer_fields_accessed": False,
            "context_chars": context_chars,
            "rule_proposal_used": rule_proposal is not None,
            "answer_free_candidate_interface": True,
        },
    }
    if case["case_type"] == "atomic_source_record":
        supported_tumors = candidate_ids(case.get("candidate_tumors", []), "tumor_subtype_id")
        supported_radios = candidate_ids(
            case.get("candidate_radiopharmaceuticals", []),
            "radiopharmaceutical_id",
        )
        tumor = raw.get("tumor_subtype_id")
        radio = raw.get("radiopharmaceutical_id")
        fallback_tumor = (rule_proposal or {}).get("predicted_tumor_subtype_id")
        fallback_radio = (rule_proposal or {}).get("predicted_radiopharmaceutical_id")
        tumor = None if raw.get('prediction_status') == 'abstain' or ('tumor_subtype_id' in raw and tumor is None) else (
            tumor
            if tumor in tumor_ids and tumor in supported_tumors
            else fallback_tumor
            if fallback_tumor in tumor_ids and fallback_tumor in supported_tumors
            else None
        )
        radio = None if raw.get('prediction_status') == 'abstain' or ('radiopharmaceutical_id' in raw and radio is None) else (
            radio
            if radio in radio_ids and radio in supported_radios
            else fallback_radio
            if fallback_radio in radio_ids and fallback_radio in supported_radios
            else None
        )
        raw_eligible = raw.get("relation_eligible")
        fallback_eligible = (rule_proposal or {}).get("predicted_relation_eligible")
        eligible = raw_eligible if isinstance(raw_eligible, bool) else fallback_eligible
        conclusion = raw.get("atomic_conclusion")
        if conclusion not in ATOMIC_CONCLUSIONS:
            conclusion = (rule_proposal or {}).get("predicted_atomic_conclusion")
        rule_can_survive_arbiter_failure = bool(
            method == "full_system"
            and isinstance(fallback_eligible, bool)
            and (not fallback_eligible or conclusion in ATOMIC_CONCLUSIONS)
        )
        if requested_status == "abstain":
            prediction_status = "abstain"
        elif (
            (not parsed_ok and not rule_can_survive_arbiter_failure)
            or not isinstance(eligible, bool)
            or (eligible and conclusion not in ATOMIC_CONCLUSIONS)
        ):
            prediction_status = "invalid"
        else:
            prediction_status = "valid"
        if prediction_status != "valid":
            eligible = None
            conclusion = None
        entity_status = _entity_resolution_status(tumor, radio)
        return {
            **common,
            "predicted_tumor_subtype_id": tumor,
            "predicted_radiopharmaceutical_id": radio,
            "predicted_relation_eligible": eligible,
            "predicted_atomic_conclusion": conclusion if eligible is True else None,
            "entity_resolution_status": entity_status,
            "graph_insertable": bool(
                prediction_status == "valid" and eligible is True and entity_status == "resolved_both"
            ),
            "prediction_status": prediction_status,
        }
    maturity = raw.get("maturity_code")
    allowed_maturity = (
        {"D1", "D2", "D3", "D4"}
        if case["clinical_use"] == "diagnostic"
        else {
            "T1",
            "T2",
            "T3",
            "T4",
        }
    )
    if maturity not in allowed_maturity:
        maturity = (rule_proposal or {}).get("predicted_maturity_code")
    conclusion = raw.get("evidence_conclusion")
    if conclusion not in SNAPSHOT_CONCLUSIONS:
        conclusion = (rule_proposal or {}).get("predicted_evidence_conclusion")
    rule_can_survive_arbiter_failure = bool(
        method == "full_system" and maturity in allowed_maturity and conclusion in SNAPSHOT_CONCLUSIONS
    )
    if requested_status == "abstain":
        prediction_status = "abstain"
    elif (
        (not parsed_ok and not rule_can_survive_arbiter_failure)
        or maturity not in allowed_maturity
        or conclusion not in SNAPSHOT_CONCLUSIONS
    ):
        prediction_status = "invalid"
    else:
        prediction_status = "valid"
    if prediction_status != "valid":
        maturity = None
        conclusion = None
    return {
        **common,
        "predicted_tumor_subtype_id": case["tumor_subtype_id"],
        "predicted_radiopharmaceutical_id": case["radiopharmaceutical_id"],
        "predicted_maturity_code": maturity,
        "predicted_evidence_conclusion": conclusion,
        "entity_resolution_status": "resolved_both",
        "graph_insertable": prediction_status == "valid",
        "prediction_status": prediction_status,
    }

def _entity_resolution_status(tumor_id: Any, radiopharmaceutical_id: Any) -> str:
    tumor_resolved = tumor_id is not None
    radio_resolved = radiopharmaceutical_id is not None
    if tumor_resolved and radio_resolved:
        return "resolved_both"
    if tumor_resolved:
        return "tumor_only"
    if radio_resolved:
        return "radiopharmaceutical_only"
    return "unresolved"

def _full_system_proposals(
    cases: list[dict[str, Any]],
    *,
    rule_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build override metadata from the preparation stage's existing rules."""
    rows = rule_rows
    case_index = {(str(case["dataset_id"]), str(case["case_id"])): case for case in cases}
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        proposal = {key: value for key, value in row.items() if key.startswith("predicted_")}
        proposal.update(row.get("prediction_provenance", {}))
        proposal["prediction_status"] = row["prediction_status"]
        proposal["allowed_override_fields"] = _allowed_override_fields(
            case=case_index[(str(row["dataset_id"]), str(row["case_id"]))],
            proposal=proposal,
        )
        allowed = set(proposal["allowed_override_fields"])
        classification_fields = allowed.difference(ENTITY_OVERRIDE_FIELDS)
        proposal["override_reason_contract"] = {
            "classification_override_reason_code": (
                expected_override_reason(proposal, classification_fields) if classification_fields else None
            ),
            "entity_only_override_reason_code": (
                "entity_conflict" if allowed.intersection(ENTITY_OVERRIDE_FIELDS) else None
            ),
        }
        proposals[(str(row["dataset_id"]), str(row["case_id"]))] = proposal
    return proposals

def _allowed_override_fields(
    *,
    case: dict[str, Any],
    proposal: dict[str, Any],
) -> list[str]:
    """Return the exact task fields an arbiter may change for this trigger."""

    rule_status = str(proposal.get("rule_decision_status", ""))
    allowed: set[str] = set()
    if case["case_type"] == "atomic_source_record":
        tumor_candidates = candidate_ids(case.get("candidate_tumors", []), "tumor_subtype_id")
        radio_candidates = candidate_ids(
            case.get("candidate_radiopharmaceuticals", []),
            "radiopharmaceutical_id",
        )
        if tumor_candidates and (
            len(tumor_candidates) > 1 or proposal.get("predicted_tumor_subtype_id") not in tumor_candidates
        ):
            allowed.add("predicted_tumor_subtype_id")
        if radio_candidates and (
            len(radio_candidates) > 1 or proposal.get("predicted_radiopharmaceutical_id") not in radio_candidates
        ):
            allowed.add("predicted_radiopharmaceutical_id")
        if rule_status == "needs_arbiter_missing_fields":
            allowed.update({"predicted_relation_eligible", "predicted_atomic_conclusion"})
        if rule_status == "needs_arbiter_rule_evidence_mismatch":
            allowed.update({"predicted_relation_eligible", "predicted_atomic_conclusion"})
        if proposal.get('semantic_audit_contract') == 'v1.3_verbatim_quote' and not proposal.get('hard_exclusion'):
            allowed.update({'predicted_relation_eligible', 'predicted_atomic_conclusion',
                            'predicted_tumor_subtype_id', 'predicted_radiopharmaceutical_id'})
    else:
        if rule_status == "needs_arbiter_independent_study_conflict":
            allowed.add("predicted_evidence_conclusion")
        if rule_status == "abstain_no_eligible_evidence":
            allowed.update({"predicted_maturity_code", "predicted_evidence_conclusion"})
        if rule_status == "needs_arbiter_rule_evidence_mismatch":
            allowed.update({"predicted_maturity_code", "predicted_evidence_conclusion"})
    return sorted(allowed)

def _task_field_values(row: dict[str, Any]) -> dict[str, Any]:
    if row["case_type"] == "atomic_source_record":
        fields = (
            "predicted_tumor_subtype_id",
            "predicted_radiopharmaceutical_id",
            "predicted_relation_eligible",
            "predicted_atomic_conclusion",
        )
    else:
        fields = ("predicted_maturity_code", "predicted_evidence_conclusion")
    return {field: row.get(field) for field in fields}

def _string_list(value: Any, singular: Any = None) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(singular, str) and singular.strip():
        return [singular.strip()]
    return []

def _source_context_block(context: str, source_id: str) -> str | None:
    expected = source_id.casefold()
    patterns = (
        re.compile(r"(?:^|[;\s])source_id=([^;\]\s]+)", re.IGNORECASE),
        re.compile(r"(?:^|[;\s\[])source\s+SELF:([^;\]\s]+)", re.IGNORECASE),
        re.compile(r"^Source\s+([^\s]+)", re.IGNORECASE),
    )
    for block in re.split(r'\n\s*\n(?=\[?(?:LightRAG|Source|NaiveRAG)\b)', context):
        header = block.splitlines()[0] if block.splitlines() else ""
        for pattern in patterns:
            match = pattern.search(header)
            if match and match.group(1).rstrip(".,").casefold() == expected:
                return block
    return None

def _override_contract_metadata_complete(raw: dict[str, Any] | None) -> bool:
    raw = raw or {}
    reason = str(raw.get("override_reason_code", ""))
    source_ids = _string_list(raw.get("source_ids"), raw.get("source_id"))
    locators = _string_list(raw.get("evidence_locators"), raw.get("evidence_locator"))
    quotes = _string_list(raw.get("evidence_quotes"), raw.get("evidence_quote"))
    return bool(reason and source_ids and (locators or quotes))

def _complete_deterministic_override_reason(
    raw: dict[str, Any] | None,
    *,
    proposal: dict[str, Any],
    overridden_fields: set[str],
) -> tuple[dict[str, Any], bool]:
    """Fill only a mechanically implied reason code when evidence metadata exists."""

    completed = dict(raw or {})
    atomic_semantic = bool(proposal.get('atomic_source_id') and
                           proposal.get('semantic_audit_contract') == 'v1.3_verbatim_quote')
    if str(completed.get("override_reason_code", "")).strip() and not atomic_semantic:
        return completed, False
    source_ids = _string_list(completed.get("source_ids"), completed.get("source_id"))
    locators = _string_list(
        completed.get("evidence_locators"),
        completed.get("evidence_locator"),
    )
    quotes = _string_list(
        completed.get("evidence_quotes"),
        completed.get("evidence_quote"),
    )
    expected_reason = expected_override_reason(proposal, overridden_fields)
    if expected_reason and source_ids and (locators or quotes):
        changed = completed.get('override_reason_code') != expected_reason
        completed["override_reason_code"] = expected_reason
        return completed, changed
    return completed, False

def _override_audit(
    raw: dict[str, Any] | None,
    context: str,
    *,
    proposal: dict[str, Any],
    overridden_fields: set[str],
) -> tuple[bool, str, dict[str, Any]]:
    raw = raw or {}
    reason_code = raw.get("override_reason_code")
    expected_reason = expected_override_reason(proposal, overridden_fields)
    if reason_code != expected_reason:
        return False, "override_reason_does_not_match_trigger", {}

    source_ids = _string_list(raw.get("source_ids"), raw.get("source_id"))
    locators = _string_list(raw.get("evidence_locators"), raw.get("evidence_locator"))
    quotes = _string_list(raw.get("evidence_quotes"), raw.get("evidence_quote"))
    minimum_sources = 2 if reason_code == "independent_study_conflict" else 1
    semantic_contract = proposal.get('semantic_audit_contract') == 'v1.3_verbatim_quote'
    if semantic_contract and proposal.get('atomic_source_id') and set(source_ids) != {proposal['atomic_source_id']}:
        return False, 'atomic_override_must_cite_own_source', {}
    if semantic_contract and proposal.get('atomic_source_id') and (
        'predicted_relation_eligible' in overridden_fields and raw.get('relation_eligible') is False
    ):
        reasons = {'nonhuman_only', 'review_only', 'protocol_only', 'registry_only',
                   'wrong_clinical_use', 'nononcology', 'no_reported_results', 'retracted'}
        if raw.get('eligibility_reason_code') not in reasons:
            return False, 'ineligibility_requires_source_reason_not_entity_uncertainty', {}
    if len(set(source_ids)) < minimum_sources:
        return False, "insufficient_override_sources", {}

    if reason_code == "independent_study_conflict":
        family_by_source: dict[str, set[str]] = defaultdict(set)
        for item in proposal.get("supporting_source_family_pairs", []):
            family_by_source[str(item.get("source_id", ""))].add(str(item.get("study_family_id", "")))
        supported_families = {
            family for source_id in source_ids for family in family_by_source.get(source_id, set()) if family
        }
        if len(supported_families) < 2:
            return False, "conflict_sources_not_independent", {}

    verified_sources: list[str] = []
    for index, source_id in enumerate(source_ids):
        block = _source_context_block(context, source_id)
        if block is None:
            return False, "source_id_not_in_retrieved_context", {}
        locator = locators[index] if index < len(locators) else ""
        quote = quotes[index] if index < len(quotes) else ""
        normalized_block = " ".join(block.casefold().split())
        locator_supported = bool(locator and locator.casefold() in block.casefold())
        normalized_quote = " ".join(quote.casefold().split())
        quote_supported = bool(len(" ".join(quote.split())) >= 12 and normalized_quote in normalized_block)
        if semantic_contract and not quote_supported:
            return False, 'semantic_override_requires_verbatim_quote', {}
        if not locator_supported and not quote_supported:
            return False, "evidence_not_located_within_source_block", {}
        verified_sources.append(source_id)
    return (
        True,
        "override_audit_pass",
        {
            "override_reason_code": reason_code,
            "override_source_ids": verified_sources,
            "override_record_locators": locators,
            "override_evidence_quotes": quotes,
            "overridden_fields": sorted(overridden_fields),
        },
    )

def _recompute_prediction_state(row: dict[str, Any]) -> None:
    entity_status = _entity_resolution_status(
        row.get("predicted_tumor_subtype_id"),
        row.get("predicted_radiopharmaceutical_id"),
    )
    row["entity_resolution_status"] = entity_status
    if row["case_type"] == "atomic_source_record":
        row["graph_insertable"] = bool(
            row["prediction_status"] == "valid"
            and row.get("predicted_relation_eligible") is True
            and row.get("predicted_atomic_conclusion") in ATOMIC_CONCLUSIONS
            and entity_status == "resolved_both"
        )
    else:
        row["graph_insertable"] = bool(row["prediction_status"] == "valid" and entity_status == "resolved_both")

def _merge_full_system_arbiter(
    *,
    rule_row: dict[str, Any],
    normalized_row: dict[str, Any],
    raw: dict[str, Any] | None,
    context: str,
    parse_status: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    rule_copy = {
        **rule_row,
        "prediction_provenance": dict(rule_row["prediction_provenance"]),
    }
    allowed_fields = set(proposal.get("allowed_override_fields", []))
    control_provenance = {
        "rule_decision_status": proposal.get("rule_decision_status"),
        "rule_reason_codes": proposal.get("rule_reason_codes", []),
        "allowed_override_fields": sorted(allowed_fields),
        "arbiter_trigger_status": arbiter_trigger_status(proposal),
        "override_reason_contract": proposal.get("override_reason_contract", {}),
    }
    rule_copy["prediction_provenance"].update(control_provenance)

    if "parsed_attempt_" not in parse_status:
        rule_copy["prediction_provenance"].update(
            {
                "llm_called": True,
                "arbiter_parse_status": parse_status,
                "decision_source": "rule_retained_after_arbiter_failure",
                "override_applied": False,
                "arbiter_outcome": "technical_failure",
            }
        )
        if proposal.get('semantic_audit_contract') == 'v1.3_verbatim_quote':
            rule_copy['prediction_status'] = 'invalid'
            for field in ('predicted_relation_eligible', 'predicted_atomic_conclusion',
                          'predicted_maturity_code', 'predicted_evidence_conclusion'):
                if field in rule_copy:
                    rule_copy[field] = None
            _recompute_prediction_state(rule_copy)
        return rule_copy

    if normalized_row["prediction_status"] != "valid":
        rule_copy["prediction_provenance"].update(
            {
                "llm_called": True,
                "arbiter_parse_status": parse_status,
                "decision_source": ('arbiter_abstained' if normalized_row['prediction_status'] == 'abstain'
                                    else 'rule_retained_after_arbiter_failure'),
                "arbiter_outcome": ('abstain' if normalized_row['prediction_status'] == 'abstain'
                                    else 'technical_failure'),
                "override_applied": False,
            }
        )
        if proposal.get('semantic_audit_contract') == 'v1.3_verbatim_quote':
            for field in ('predicted_relation_eligible', 'predicted_atomic_conclusion',
                          'predicted_maturity_code', 'predicted_evidence_conclusion'):
                if field in rule_copy:
                    rule_copy[field] = None
            rule_copy['prediction_status'] = normalized_row['prediction_status']
            _recompute_prediction_state(rule_copy)
        return rule_copy

    rule_values = _task_field_values(rule_row)
    normalized_values = _task_field_values(normalized_row)
    attempted_changes = {field for field, value in normalized_values.items() if value != rule_values[field]}
    locked_attempts = attempted_changes.difference(allowed_fields)
    candidate = {
        **normalized_row,
        "prediction_provenance": {**rule_row['prediction_provenance'], **normalized_row["prediction_provenance"]},
    }
    for field in locked_attempts:
        candidate[field] = rule_values[field]
    _recompute_prediction_state(candidate)
    candidate_values = _task_field_values(candidate)
    overridden_fields = {field for field in allowed_fields if candidate_values.get(field) != rule_values.get(field)}

    if not overridden_fields:
        rule_copy["prediction_provenance"].update(
            {
                "llm_called": True,
                "arbiter_parse_status": parse_status,
                "decision_source": ("rule_retained_locked_fields" if locked_attempts else "rule_confirmed_by_arbiter"),
                "override_applied": False,
                "locked_field_attempts": sorted(locked_attempts),
            }
        )
        return rule_copy

    audit_raw, reason_code_completed = _complete_deterministic_override_reason(
        raw,
        proposal=proposal,
        overridden_fields=overridden_fields,
    )
    audit_ok, audit_status, audit_payload = _override_audit(
        audit_raw,
        context,
        proposal=proposal,
        overridden_fields=overridden_fields,
    )
    if not audit_ok:
        raw_payload = raw or {}
        rule_copy["prediction_provenance"].update(
            {
                "llm_called": True,
                "arbiter_parse_status": parse_status,
                "decision_source": "rule_retained_after_rejected_override",
                "override_applied": False,
                "override_rejection_code": audit_status,
                "arbiter_outcome": "rejected_override",
                "attempted_override_fields": sorted(overridden_fields),
                "override_contract_metadata_complete": (_override_contract_metadata_complete(audit_raw)),
                "override_received_reason_code": raw_payload.get("override_reason_code"),
                "override_reason_code_deterministically_completed": reason_code_completed,
                "override_completed_reason_code": audit_raw.get("override_reason_code"),
                "override_received_source_ids": _string_list(
                    raw_payload.get("source_ids"),
                    raw_payload.get("source_id"),
                ),
                "override_received_evidence_locators": _string_list(
                    raw_payload.get("evidence_locators"),
                    raw_payload.get("evidence_locator"),
                ),
                "override_received_evidence_quotes": _string_list(
                    raw_payload.get("evidence_quotes"),
                    raw_payload.get("evidence_quote"),
                ),
                "locked_field_attempts": sorted(locked_attempts),
            }
        )
        return rule_copy

    candidate["prediction_provenance"].update(
        {
            **control_provenance,
            "llm_called": True,
            "decision_source": "audited_llm_override",
            "override_applied": True,
            "override_audit_status": audit_status,
            "attempted_override_fields": sorted(overridden_fields),
            "override_contract_metadata_complete": True,
            "override_reason_code_deterministically_completed": reason_code_completed,
            "override_received_reason_code": (raw or {}).get('override_reason_code'),
            "eligibility_reason_code": (raw or {}).get('eligibility_reason_code'),
            "locked_field_attempts": sorted(locked_attempts),
            "override_changes": {
                field: {
                    "rule_value": rule_values[field],
                    "override_value": candidate_values[field],
                }
                for field in sorted(overridden_fields)
            },
            **audit_payload,
        }
    )
    return candidate
