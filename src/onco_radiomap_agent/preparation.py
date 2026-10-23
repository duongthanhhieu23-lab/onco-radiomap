from __future__ import annotations

from typing import Any

from onco_radiomap_agent.entity_candidates import candidate_ids

from onco_radiomap_agent.full_system_control import ENTITY_OVERRIDE_FIELDS, arbiter_required, expected_override_reason

from onco_radiomap_agent.prediction_rules import predict_atomic, predict_snapshot

METHODS = ("pure_llm", "naive_rag", "lightrag", "full_system")

PAIR_SNAPSHOT_GUIDANCE = """Pair snapshots: the pair IDs and names are query inputs. Preserve the query IDs and evaluate only results attributable to that exact tumor-radiopharmaceutical pair, clinical use, population, and cutoff. Return maturity independently from conclusion. Deduplicate records with the same study_family; multiple versions of one study are not independent. A registry, protocol, review, or discovery record never raises maturity and is not a reported outcome. More sources alone never raise maturity.
Diagnostic maturity uses explicit design evidence: D1 is human feasibility, uptake, localization, or visualization without a D2-D4 criterion. D2 requires at least one study family with prespecified diagnostic performance (for example sensitivity, specificity, accuracy, or AUC), comparison with a reference/gold standard, head-to-head tracer comparison, or evaluated management impact. D3 requires either a prospective multicenter family meeting a D2 performance criterion, or at least two independent study families each meeting a D2 performance criterion. D4 requires a prospective multicenter family with evaluated management impact or a patient-relevant diagnostic-utility endpoint.
Therapeutic maturity also uses explicit combinations: T1 is human therapeutic dosimetry, safety, or feasibility without a T2-T4 combination. T2 requires prespecified objective efficacy in a prospective or defined single-arm cohort plus reported safety or biodistribution/dosimetry. T3 requires either randomized/controlled/credible-comparator evidence plus prespecified objective efficacy in one family, or at least two independent prospective families with prespecified objective efficacy. T4 requires phase III plus controlled design plus a patient-relevant treatment endpoint in one family, or at least two independent controlled families with patient-relevant treatment endpoints.
Conclusion is about an interpretable main result for the exact query pair: supportive when its objective was met or the effect was clearly favorable, negative when its objective was not met or the effect was clearly unfavorable, and inconclusive when the main objective/result is absent, unclear, mixed, imprecise, only feasibility/uptake/dosimetry is shown, or cannot be attributed to the query pair. A favorable secondary observation does not by itself make the snapshot supportive. Conflicting requires directly attributable opposing interpretable results from at least two independent study_family values of comparable maturity; disagreement within one family is inconclusive.
When retrieved_context is supplied, use only source blocks within it. Treat raw_text_design_signals as cues extracted from that same source text; absence means not explicitly reported, not proven false. A block with rule_aggregation_eligible=false is an ambiguous candidate: use it only when its excerpt directly attributes the reported result to the exact query pair; otherwise ignore that block. If no eligible directly attributable family remains, abstain."""

SYSTEM_PROMPT = (
    """You classify evidence for a two-entity oncology evidence graph.
Return one JSON object with a `predictions` array and no prose. Preserve each integer `i`.
There are exactly two entities: a standardized tumor subtype and an active radiopharmaceutical construct.
Atomic source records require two separate decisions. First decide relation eligibility from evidence type: a source-locatable HUMAN oncology result for the requested use is eligible even when its finding is negative or inconclusive. Registry-only, review-only, protocol-only, discovery, cell, animal, or preclinical evidence is ineligible. Human therapeutic dosimetry or safety for an administered therapeutic construct is eligible T1 evidence; diagnostic imaging/dosimetry, trial registration, or a review is not human therapeutic-outcome evidence. Second, only when eligible, assign atomic_conclusion: supportive for an interpretable favorable main result or met objective, negative for an interpretable unfavorable main result or failed objective, and inconclusive when the main result/objective is absent, unclear, mixed, imprecise, or not interpretable. Mere target uptake, feasibility, safety, or a favorable secondary observation does not prove that the main objective was met. `conflicting` is forbidden at atomic level. Return relation_eligible independently from entity resolution; an unresolved entity must never flip the relation decision to false.
Resolve a radiopharmaceutical by isotope plus carrier/ligand plus active construct. Select canonical IDs only from candidate_tumors and candidate_radiopharmaceuticals. If multiple constructs occur and the reported result cannot be attributed to one candidate, return null rather than guessing.
"""
    + PAIR_SNAPSHOT_GUIDANCE
    + """
Do not turn stage, target expression, age, prior therapy, or disease status into tumor-subtype entities. Do not extrapolate adult evidence to children. Never treat a trial registration as a reported result.
Use prediction_status="abstain" with null task labels when supplied evidence cannot support a valid answer. Never fill a failed or uncertain answer with false, D1, or T1.
Use exactly these object contracts inside `predictions`:
- atomic: {"i":0,"prediction_status":"valid","tumor_subtype_id":null,"radiopharmaceutical_id":null,"relation_eligible":false,"atomic_conclusion":null}
- snapshot: {"i":0,"prediction_status":"valid","tumor_subtype_id":"...","radiopharmaceutical_id":"...","maturity_code":"D1","evidence_conclusion":"inconclusive"}
"""
)

FULL_SYSTEM_PROMPT = (
    """You are the evidence arbiter inside the Onco-RadioMap Full System.
There is exactly one input item. Return exactly one JSON object shaped as {"predictions":[{...}]} and no prose. The predictions array must contain exactly one object with i=0. Never use `inputs` as the top-level output key and never return the prediction object without the `predictions` envelope.

Each input contains a reference-free rule_proposal, allowed_override_fields, and retrieved_context. The proposal is not a reference answer. Fields outside allowed_override_fields are locked. Keep the rule value unless complete source-locatable evidence supports changing an allowed field. Never infer a relation from entity resolution alone. Never use stage, target expression, age, prior therapy, or disease status as a tumor subtype. Never treat a registry record as a reported result, and never extrapolate adult evidence to children.

Atomic source records: decide relation eligibility independently from entity resolution. Eligible means a source-locatable HUMAN oncology result for the requested clinical use. Registry-only, protocol-only, review-only, discovery, cell, animal, and preclinical evidence are ineligible. For an eligible atomic record, atomic_conclusion is supportive, negative, or inconclusive. `conflicting` is forbidden at atomic level.

"""
    + PAIR_SNAPSHOT_GUIDANCE
    + """

Use prediction_status="abstain" and null task labels when evidence is insufficient. Never replace failure or uncertainty with false, D1, or T1.

Conditional audit contract:
1. If you do not change any allowed field, return the ordinary task object; audit fields are optional.
2. If you change any allowed field, the SAME prediction object must include override_reason_code, source_ids, and exactly one of evidence_locators or evidence_quotes. Arrays must correspond by source position. Use at most two source_ids. Prefer locators. If a quote is necessary, return exactly one verbatim quote of 12-240 characters.
3. Allowed reason codes are entity_conflict, rule_evidence_mismatch, missing_field_resolved, and independent_study_conflict. The rule_proposal supplies override_reason_contract. If any classification field changes, use classification_override_reason_code; if only entity fields change, use entity_only_override_reason_code. independent_study_conflict requires at least two source_ids from two distinct study_family values.
4. Return one compact prediction object for i=0. Never repeat or summarize retrieved_context, rule_proposal, or source prose outside the single permitted quote.

Ordinary object shapes:
- atomic: {"i":0,"prediction_status":"valid","tumor_subtype_id":null,"radiopharmaceutical_id":null,"relation_eligible":false,"atomic_conclusion":null}
- snapshot: {"i":0,"prediction_status":"valid","tumor_subtype_id":"...","radiopharmaceutical_id":"...","maturity_code":"D1","evidence_conclusion":"inconclusive"}

Final output envelope (mandatory): {"predictions":[<the one compact atomic or snapshot object above>]}
"""
)

def _llm_item(
    case: dict[str, Any],
    context: str,
    index: int,
    rule_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if case["case_type"] == "atomic_source_record":
        item = {
            "i": index,
            "case_type": "atomic_source_record",
            "clinical_use": case["clinical_use"],
            "source_id": case["source_id"],
            "record_locator": case["record_locator"],
            "source_text": str(case["source_text"]),
            "publication_types": case.get("publication_types", []),
            "population_index_terms": case.get("population_index_terms", []),
            "human_indexed": case.get("human_indexed"),
            "candidate_tumors": case.get("candidate_tumors", []),
            "candidate_radiopharmaceuticals": case.get("candidate_radiopharmaceuticals", []),
        }
    else:
        item = {
            "i": index,
            "case_type": "pair_snapshot",
            "clinical_use": case["clinical_use"],
            "cutoff_date": case["cutoff_date"],
            "population_scope": case["population_scope"],
            "tumor_subtype_id": case["tumor_subtype_id"],
            "tumor_subtype_name": case["tumor_subtype_name"],
            "radiopharmaceutical_id": case["radiopharmaceutical_id"],
            "radiopharmaceutical_name": case["radiopharmaceutical_name"],
        }
    if context:
        item["retrieved_context"] = context
    if rule_proposal is not None:
        item["rule_proposal"] = rule_proposal
    return item

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

def _allowed_fact_ids_by_dataset(
    cases: list[dict[str, Any]],
    fact_id_by_case: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    dataset_ids = sorted({str(case["dataset_id"]) for case in cases})
    fact_id_by_case = fact_id_by_case or {}
    return {
        dataset_id: {
            str(
                fact_id_by_case.get(
                    str(case.get("case_id", "")),
                    str(case.get("source_fact_id", "")),
                )
            )
            for case in cases
            if case["dataset_id"] == dataset_id
            and case["case_type"] == "atomic_source_record"
            and (str(case.get("case_id", "")) in fact_id_by_case or bool(case.get("source_fact_id")))
        }
        for dataset_id in dataset_ids
    }

def _full_system_rows(
    cases: list[dict[str, Any]],
    *,
    facts: list[dict[str, Any]],
    evidence_pool_cases: list[dict[str, Any]] | None = None,
    run_id: int,
) -> list[dict[str, Any]]:
    evidence_pool_cases = evidence_pool_cases or cases
    dataset_facts = {
        dataset_id: [fact for fact in facts if str(fact.get("origin_dataset_id")) == dataset_id]
        for dataset_id in sorted({str(case["dataset_id"]) for case in cases})
    }
    facts_by_origin_case = {str(fact["origin_case_id"]): fact for fact in facts}
    fact_id_by_case = {str(fact["origin_case_id"]): str(fact["fact_id"]) for fact in facts}
    # The manifest selects scored cases only.  The evidence allowlist is built
    # from every atomic case in the selected dataset, never from the subset.
    allowed_by_dataset = _allowed_fact_ids_by_dataset(
        evidence_pool_cases,
        fact_id_by_case,
    )
    rows = []
    for case in cases:
        dataset_id = str(case["dataset_id"])
        common = {
            "case_id": case["case_id"],
            "dataset_id": dataset_id,
            "case_type": case["case_type"],
            "clinical_use": case["clinical_use"],
            "method_id": "full_system",
            "run_id": run_id,
        }
        if case["case_type"] == "atomic_source_record":
            fact = facts_by_origin_case[str(case["case_id"])]
            predicted = predict_atomic(fact)
            tumor_candidates = candidate_ids(case.get("candidate_tumors", []), "tumor_subtype_id")
            radio_candidates = candidate_ids(
                case.get("candidate_radiopharmaceuticals", []),
                "radiopharmaceutical_id",
            )
            tumor_id = predicted["predicted_tumor_subtype_id"]
            radio_id = predicted["predicted_radiopharmaceutical_id"]
            tumor_id = tumor_id if tumor_id in tumor_candidates else None
            radio_id = radio_id if radio_id in radio_candidates else None
            entity_status = _entity_resolution_status(tumor_id, radio_id)
            prediction_status = "abstain" if predicted["predicted_relation_eligible"] is None else "valid"
            row = {
                **common,
                "predicted_tumor_subtype_id": tumor_id,
                "predicted_radiopharmaceutical_id": radio_id,
                "predicted_relation_eligible": predicted["predicted_relation_eligible"],
                "predicted_atomic_conclusion": predicted["predicted_atomic_conclusion"],
                "entity_resolution_status": entity_status,
                "graph_insertable": bool(
                    prediction_status == "valid"
                    and predicted["predicted_relation_eligible"]
                    and entity_status == "resolved_both"
                ),
                "prediction_status": prediction_status,
                "prediction_provenance": {
                    "engine": "reference_free_rules_first_engine_v1.2",
                    "decision_source": "rules_only",
                    "llm_called": False,
                    "override_applied": False,
                    "prediction_fact_id": fact["fact_id"],
                    "prediction_fact_schema_version": fact["prediction_fact_schema_version"],
                    "prediction_fact_source_text_sha256": fact["prediction_provenance"]["source_text_sha256"],
                    "rule_decision_status": predicted["rule_decision_status"],
                    "rule_reason_codes": predicted["rule_reason_codes"],
                    "supporting_source_ids": predicted["supporting_source_ids"],
                    "supporting_locators": predicted["supporting_locators"],
                    "answer_fields_accessed": False,
                    "deterministic_repeat": True,
                },
            }
        else:
            predicted = predict_snapshot(
                dataset_facts[dataset_id],
                allowed_fact_ids=allowed_by_dataset[dataset_id],
                dataset_id=dataset_id,
                tumor_subtype_id=str(case["tumor_subtype_id"]),
                radiopharmaceutical_id=str(case["radiopharmaceutical_id"]),
                clinical_use=str(case["clinical_use"]),
                population_scope=str(case["population_scope"]),
                cutoff_date=str(case["cutoff_date"]),
            )
            prediction_status = "abstain" if predicted["rule_decision_status"].startswith("abstain_") else "valid"
            row = {
                **common,
                "predicted_tumor_subtype_id": predicted["predicted_tumor_subtype_id"],
                "predicted_radiopharmaceutical_id": predicted["predicted_radiopharmaceutical_id"],
                "predicted_maturity_code": predicted["predicted_maturity_code"],
                "predicted_evidence_conclusion": predicted["predicted_evidence_conclusion"],
                "entity_resolution_status": "resolved_both",
                "graph_insertable": prediction_status == "valid",
                "prediction_status": prediction_status,
                "prediction_provenance": {
                    "engine": "reference_free_rules_first_engine_v1.2",
                    "decision_source": "rules_only",
                    "llm_called": False,
                    "override_applied": False,
                    "supporting_fact_count": predicted["supporting_fact_count"],
                    "supporting_family_count": predicted["supporting_family_count"],
                    "supporting_fact_ids": predicted["supporting_fact_ids"],
                    "supporting_source_ids": predicted["supporting_source_ids"],
                    "supporting_locators": predicted["supporting_locators"],
                    "supporting_study_family_ids": predicted["supporting_study_family_ids"],
                    "supporting_source_family_pairs": predicted["supporting_source_family_pairs"],
                    "supporting_pair_match_basis": predicted["supporting_pair_match_basis"],
                    "rule_decision_status": predicted["rule_decision_status"],
                    "rule_reason_codes": predicted["rule_reason_codes"],
                    "answer_fields_accessed": False,
                    "deterministic_repeat": True,
                },
            }
        rows.append(row)
    return rows

def _full_system_proposals(
    cases: list[dict[str, Any]],
    *,
    rule_rows: list[dict[str, Any]] | None = None,
    facts: list[dict[str, Any]] | None = None,
    evidence_pool_cases: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = rule_rows
    if rows is None:
        if facts is None:
            raise ValueError("facts are required when rule_rows are not supplied")
        rows = _full_system_rows(
            cases,
            facts=facts,
            evidence_pool_cases=evidence_pool_cases,
            run_id=1,
        )
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
    else:
        if rule_status == "needs_arbiter_independent_study_conflict":
            allowed.add("predicted_evidence_conclusion")
        if rule_status == "abstain_no_eligible_evidence":
            allowed.update({"predicted_maturity_code", "predicted_evidence_conclusion"})
        if rule_status == "needs_arbiter_rule_evidence_mismatch":
            allowed.update({"predicted_maturity_code", "predicted_evidence_conclusion"})
    return sorted(allowed)

def _requires_full_system_arbiter(
    case: dict[str, Any],
    proposal: dict[str, Any],
) -> bool:
    del case
    return arbiter_required(proposal)
