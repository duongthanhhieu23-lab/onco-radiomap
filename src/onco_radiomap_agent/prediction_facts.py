"""Build reference-free structured facts for the V1.2 full system.

The frozen ``SourceFact`` records are inputs to the automatic reference-label
program.  Reusing those records in the predictor would therefore measure rule
consistency rather than independent prediction.  This module instead creates a
small, auditable prediction-side representation from the same raw title and
abstract exposed to every baseline plus the shared alias candidate interface.

No ``reference_*`` field, canonical entity ID outside the candidate interface,
or structured label-generator field is accepted or emitted here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .design_extraction import (
    credible_treatment_comparator,
    explicit_controlled_design,
    single_arm_defined_cohort,
)
from .entity_candidates import candidate_ids

PREDICTION_FACT_SCHEMA_VERSION = "onco-radiomap-prediction-fact-v1.2.1"

_HUMAN_RE = re.compile(
    r"\b(?:patients?|participants?|subjects?|volunteers?|study\s+in\s+humans?|"
    r"human\s+(?:participants?|subjects?|study|studies|cohort|trial)|"
    r"pilot\s+clinical|clinical\s+(?:pet|spect|imaging)[^.;]{0,80}"
    r"(?:demonstrated|showed|revealed))\b",
    re.IGNORECASE,
)
_NONHUMAN_RE = re.compile(
    r"\b(?:mice|mouse|murine|rats?|xenografts?|cell\s+lines?|in\s+vitro|"
    r"animal\s+models?|preclinical|porcine|canine|rabbit|hamster)\b",
    re.IGNORECASE,
)
_STRONG_PRECLINICAL_TITLE_RE = re.compile(
    r"\b(?:preclinical|nonclinical|in\s+vitro|cell\s+lines?|mice|mouse|murine|"
    r"xenografts?|animal\s+models?)\b",
    re.IGNORECASE,
)
_STRONG_ANIMAL_EXPERIMENT_RE = re.compile(
    r"\b(?:tumou?r[- ]bearing\s+(?:mice|rats?)|(?:mice|rats?)\s+bearing\s+"
    r"tumou?rs?|mouse\s+model|murine\s+model|animal\s+model|"
    r"human\s+tumou?r\s+xenografts?|xenograft\s+models?)\b",
    re.IGNORECASE,
)
_ONCOLOGY_RE = re.compile(
    r"\b(?:cancers?|tumou?rs?|carcinoma|sarcoma|lymphoma|leukemia|myeloma|"
    r"melanoma|glioma|neoplasms?|metastases?|metastatic|malignan)\w*\b",
    re.IGNORECASE,
)
_RESULT_EVIDENCE_CUES = (
    (
        "results_section",
        re.compile(r"\bresults?\b", re.IGNORECASE),
    ),
    (
        "conclusion_section",
        re.compile(r"\bconclusions?\b", re.IGNORECASE),
    ),
    (
        "patient_population",
        re.compile(r"\bpatients?\b", re.IGNORECASE),
    ),
    (
        "participant_population",
        re.compile(r"\bparticipants?\b", re.IGNORECASE),
    ),
    (
        "subject_population",
        re.compile(r"\bsubjects?\b", re.IGNORECASE),
    ),
    (
        "observed_or_treated_cohort",
        re.compile(
            r"\b(?:n\s*=|enrolled|included|underwent|received|treated)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clinical_endpoint",
        re.compile(
            r"\b(?:sensitivity|specificity|response\s+rate|survival|dosimetry|"
            r"adverse\s+events?)\b",
            re.IGNORECASE,
        ),
    ),
)
_ABSTRACT_RE = re.compile(r"(?is)\bAbstract:\s*(.*?)(?:\nDOI:|$)")
_DIAGNOSTIC_USE_RE = re.compile(
    r"\b(?:pet(?:/ct)?|spect(?:/ct)?|scintigraph|imaging|diagnos|detect|"
    r"locali[sz]|staging|sensitivity|specificity|accuracy|visuali[sz]|"
    r"biodistribution)\w*\b",
    re.IGNORECASE,
)
_THERAPEUTIC_USE_RE = re.compile(
    r"\b(?:radionuclide\s+therap|radioligand\s+therap|targeted\s+radionuclide|"
    r"radiopharmaceutical\s+therap|treatment|treated|therapeutic|therapy|"
    r"efficacy|response|survival|progression[- ]free|toxicity|adverse\s+events?|"
    r"dosimetry|absorbed\s+dose|maximum\s+tolerated)\w*\b",
    re.IGNORECASE,
)
_THERAPEUTIC_ACTION_RE = re.compile(
    r"\b(?:radionuclide\s+therap|radioligand\s+therap|targeted\s+radionuclide|"
    r"radiopharmaceutical\s+therap|treatment|treated|therapeutic|therapy|"
    r"efficacy|objective\s+response|overall\s+survival|progression[- ]free|"
    r"toxicity|adverse\s+events?|maximum\s+tolerated)\w*\b",
    re.IGNORECASE,
)
_THERAPEUTIC_DOSIMETRY_RE = re.compile(
    r"\b(?:dosimetry|absorbed\s+dose)\b.*\b(?:177\s*lu|225\s*ac|131\s*i|"
    r"90\s*y|186\s*re|188\s*re|223\s*ra|212\s*pb)\b|"
    r"\b(?:177\s*lu|225\s*ac|131\s*i|90\s*y|186\s*re|188\s*re|223\s*ra|"
    r"212\s*pb)\b.*\b(?:dosimetry|absorbed\s+dose)\b",
    re.IGNORECASE | re.DOTALL,
)
_REVIEW_TITLE_RE = re.compile(
    r"\b(?:systematic\s+review|meta-analysis|review|overview|state[- ]of[- ]the[- ]art|"
    r"best\s+practices|current\s+status|prospects?)\b",
    re.IGNORECASE,
)
_PROTOCOL_TITLE_RE = re.compile(
    r"\b(?:study\s+protocol|trial\s+protocol|protocol\s+for|rationale\s+and\s+design)\b",
    re.IGNORECASE,
)
_REGISTRY_SOURCE_RE = re.compile(
    r"(?:clinicaltrials\.gov|clinicaltrials_registration|trial[_ -]registration)",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:did\s+not\s+meet|failed\s+to|no\s+(?:significant\s+)?(?:benefit|"
    r"response|association|difference|tumou?r\s+locali[sz]ation)|not\s+effective|"
    r"ineffective|inferior|unfavourable|unfavorable)\b",
    re.IGNORECASE,
)
_SUPPORTIVE_RE = re.compile(
    r"\b(?:met\s+(?:the\s+)?primary\s+(?:end\s*point|objective)|significantly\s+"
    r"(?:improved|higher|reduced)|high(?:ly)?\s+(?:accurate|sensitive|specific)|"
    r"effective|efficacious|promising|feasible|safe|well[- ]tolerated|"
    r"demonstrated\s+(?:benefit|accuracy|uptake)|accurate)\b",
    re.IGNORECASE,
)
_MIXED_RE = re.compile(
    r"\b(?:mixed\s+results?|discordant|heterogeneous\s+(?:results?|response)|"
    r"variable\s+(?:results?|response)|wide\s+confidence\s+interval)\b",
    re.IGNORECASE,
)
_DOI_LINE_RE = re.compile(r"(?i)\bDOI:\s*([^\r\n]+)")
_VALID_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
_EVIDENCE_SEGMENT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _contains(pattern: re.Pattern[str], text: str) -> bool:
    return pattern.search(text) is not None


def _result_evidence_cues(source_text: str) -> list[str]:
    """Return independent, source-locatable cues for a reported human result.

    A single percentage, toxicity word, or background mention is not enough to
    turn a synthesis, review, protocol, or incidental report into an eligible
    clinical result.  The supplied abstract must be non-empty and at least two
    distinct cue classes must occur.  This is a closed-world decision over the
    complete title/abstract exposed to every method, not an entity-resolution
    shortcut.
    """

    abstract_match = _ABSTRACT_RE.search(source_text)
    if abstract_match is None or not abstract_match.group(1).strip():
        return []
    return [
        name
        for name, pattern in _RESULT_EVIDENCE_CUES
        if _contains(pattern, source_text)
    ]


def _single_candidate(
    candidates: Iterable[Mapping[str, Any]],
    id_field: str,
) -> str | None:
    rows = list(candidates)
    identifiers = candidate_ids(rows, id_field)
    if len(identifiers) == 1:
        return next(iter(identifiers))
    specificity_by_id: dict[str, int] = {}
    for candidate in rows:
        candidate_id = str(candidate.get(id_field, "")).strip()
        if not candidate_id:
            continue
        specificity_by_id[candidate_id] = max(
            (
                len(
                    " ".join(
                        "".join(
                            character if character.isalnum() else " "
                            for character in str(alias).casefold()
                        ).split()
                    )
                )
                for alias in candidate.get("matched_aliases", [])
            ),
            default=0,
        )
    if not specificity_by_id:
        return None
    best_specificity = max(specificity_by_id.values())
    best_ids = {
        candidate_id
        for candidate_id, specificity in specificity_by_id.items()
        if specificity == best_specificity
    }
    return next(iter(best_ids)) if best_specificity > 0 and len(best_ids) == 1 else None


def _title_from_source_text(text: str) -> str:
    match = re.search(r"(?im)^Title:\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip()
    return text.split("Abstract:", 1)[0].strip()


def _normalized_phrase(value: object) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in str(value).casefold()).split()
    )


def _candidate_ids_in_segment(
    segment: str,
    candidates: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
) -> set[str]:
    normalized_segment = f" {_normalized_phrase(segment)} "
    matched: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get(id_field, "")).strip()
        aliases = [str(alias) for alias in candidate.get("matched_aliases", [])]
        if candidate_id and any(
            (normalized := _normalized_phrase(alias))
            and f" {normalized} " in normalized_segment
            for alias in aliases
        ):
            matched.add(candidate_id)
    return matched


def _candidate_pair_bindings(
    source_text: str,
    tumor_candidates: Iterable[Mapping[str, Any]],
    radio_candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return only pair assignments that are unique within one evidence segment."""

    tumor_rows = list(tumor_candidates)
    radio_rows = list(radio_candidates)
    tumor_ids = candidate_ids(tumor_rows, "tumor_subtype_id")
    radio_ids = candidate_ids(radio_rows, "radiopharmaceutical_id")
    bindings: dict[tuple[str, str], dict[str, str]] = {}
    if len(tumor_ids) == 1 and len(radio_ids) == 1:
        tumor_id = next(iter(tumor_ids))
        radio_id = next(iter(radio_ids))
        bindings[(tumor_id, radio_id)] = {
            "tumor_subtype_id": tumor_id,
            "radiopharmaceutical_id": radio_id,
            "pair_match_basis": "unique_candidate_pair",
        }
    for segment in _EVIDENCE_SEGMENT_RE.split(source_text):
        segment = segment.strip()
        if not segment:
            continue
        segment_tumors = _candidate_ids_in_segment(
            segment,
            tumor_rows,
            id_field="tumor_subtype_id",
        )
        segment_radios = _candidate_ids_in_segment(
            segment,
            radio_rows,
            id_field="radiopharmaceutical_id",
        )
        if len(segment_tumors) != 1 or len(segment_radios) != 1:
            continue
        tumor_id = next(iter(segment_tumors))
        radio_id = next(iter(segment_radios))
        bindings[(tumor_id, radio_id)] = {
            "tumor_subtype_id": tumor_id,
            "radiopharmaceutical_id": radio_id,
            "pair_match_basis": "unique_same_segment_pair",
        }
    return [bindings[key] for key in sorted(bindings)]


def _population_scope(text: str, index_terms: Iterable[str] = ()) -> str:
    """Apply the frozen population contract without broad age-term expansion."""

    indexed = {str(term) for term in index_terms}
    pediatric = bool(
        re.search(
            r"\b(?:paediatric|pediatric|children|childhood|adolescents?)\b",
            text,
            re.IGNORECASE,
        )
    ) or bool(indexed.intersection({"Infant", "Child", "Adolescent"}))
    adult = bool(re.search(r"\badults?\b", text, re.IGNORECASE)) or "Adult" in indexed
    if pediatric and adult:
        return "mixed"
    if pediatric:
        return "pediatric"
    if adult:
        return "adult"
    return "not_reported"


def _clinical_use_alignment(text: str, clinical_use: str) -> bool | None:
    if clinical_use == "diagnostic":
        return True if _contains(_DIAGNOSTIC_USE_RE, text) else None
    if clinical_use == "therapeutic":
        if _contains(_THERAPEUTIC_ACTION_RE, text):
            return True
        if _contains(_THERAPEUTIC_DOSIMETRY_RE, text):
            return True
        return None
    return None


def _source_role(
    *,
    text: str,
    source_id: str,
    human_subjects: bool | None,
    oncology_context: bool | None,
    result_reported: bool | None,
    clinical_use_alignment: bool | None,
    source_type: str = "",
    record_locator: str = "",
    publication_types: Iterable[str] = (),
) -> str:
    source_signature = f"{source_id} {source_type} {record_locator}"
    publication_type_set = {str(value).casefold() for value in publication_types}
    if source_id.startswith("CTG:") or _contains(
        _REGISTRY_SOURCE_RE,
        source_signature,
    ):
        return "clinical_trial_registration"
    if _contains(_REVIEW_TITLE_RE, _title_from_source_text(text)) or any(
        "review" in value or "meta-analysis" in value
        for value in publication_type_set
    ):
        return "review_or_discovery"
    if _contains(_PROTOCOL_TITLE_RE, _title_from_source_text(text)) or any(
        "protocol" in value for value in publication_type_set
    ):
        return "discovery_or_bibliographic"
    if (
        human_subjects is True
        and oncology_context is True
        and result_reported is True
        and clinical_use_alignment is True
    ):
        return "primary_human_result"
    return "discovery_or_bibliographic"


def _design_flags(text: str, clinical_use: str) -> dict[str, bool]:
    flag_patterns = {
        "prospective": r"\bprospective\b",
        "multicenter": r"\bmulti[- ]?cent(?:er|re)\b",
        "randomized": r"\brandomi[sz]ed\b",
        "phase_iii": r"\bphase\s+(?:iii|3)\b",
        "first_in_human": r"\bfirst[- ]in[- ]human\b",
        "case_report_or_small_series": r"\b(?:case\s+report|case\s+series|[1-9]\s+patients?)\b",
        "safety_reported": r"\b(?:safety|toxicity|adverse\s+events?|well[- ]tolerated)\b",
        "biodistribution_or_dosimetry_reported": r"\b(?:biodistribution|dosimetry|absorbed\s+dose)\b",
        "prespecified_diagnostic_performance": r"\b(?:sensitivity|specificity|accuracy|auc|receiver\s+operating)\b",
        "gold_standard_comparison": r"\b(?:reference\s+standard|histopatholog|pathologic\s+confirmation|biopsy)\b",
        "head_to_head_tracer_comparison": r"\b(?:head[- ]to[- ]head|tracer\s+comparison|compared\s+with\s+.+(?:pet|spect))\b",
        "management_impact_evaluated": r"\b(?:management\s+impact|changed\s+(?:clinical\s+)?management|treatment\s+decision)\b",
        "patient_relevant_diagnostic_utility_endpoint": r"\b(?:management\s+impact|patient\s+outcome|clinical\s+utility)\b",
        "prespecified_objective_efficacy": r"\b(?:primary\s+(?:end\s*point|objective)|objective\s+response|response\s+rate|progression[- ]free|overall\s+survival)\b",
        "patient_relevant_treatment_endpoint": r"\b(?:overall\s+survival|progression[- ]free|quality\s+of\s+life|symptom\s+control)\b",
        "target_visualization_reported": r"\b(?:tumou?r\s+uptake|lesion\s+detection|visuali[sz]|locali[sz])\w*\b",
    }
    flags = {
        key: re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
        for key, pattern in flag_patterns.items()
    }
    flags["controlled"] = explicit_controlled_design(text)
    flags["credible_comparator"] = credible_treatment_comparator(text)
    flags["single_arm_defined_cohort"] = single_arm_defined_cohort(text)
    if clinical_use != "diagnostic":
        flags["prespecified_diagnostic_performance"] = False
        flags["gold_standard_comparison"] = False
        flags["head_to_head_tracer_comparison"] = False
        flags["management_impact_evaluated"] = False
        flags["patient_relevant_diagnostic_utility_endpoint"] = False
    if clinical_use != "therapeutic":
        flags["prespecified_objective_efficacy"] = False
        flags["patient_relevant_treatment_endpoint"] = False
    return flags


def _result_fields(text: str, *, result_reported: bool | None) -> dict[str, Any]:
    negative = _contains(_NEGATIVE_RE, text)
    # Remove explicit negative phrases before looking for positive keywords;
    # otherwise "not effective" would spuriously match "effective" as well.
    supportive = _contains(_SUPPORTIVE_RE, _NEGATIVE_RE.sub(" ", text))
    materially_mixed = _contains(_MIXED_RE, text) or (negative and supportive)
    interpretable = bool(result_reported is True and (negative or supportive))
    if materially_mixed:
        objective_status = "not_reported"
        direction = "not_reported"
    elif negative:
        objective_status = "not_met"
        direction = "unfavorable"
    elif supportive:
        objective_status = "met"
        direction = "favorable"
    else:
        objective_status = "not_reported"
        direction = "not_reported"
    return {
        "adequately_interpretable": interpretable,
        "materially_imprecise_or_internally_mixed": materially_mixed,
        "primary_objective_status": objective_status,
        "reported_effect_direction": direction,
    }


def _valid_doi(source_text: str) -> str | None:
    match = _DOI_LINE_RE.search(source_text)
    if not match:
        return None
    candidate = match.group(1).strip().strip("<>[](){}.,;")
    candidate = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", candidate, flags=re.IGNORECASE)
    return candidate.casefold() if _VALID_DOI_RE.fullmatch(candidate) else None


def _study_family_id(source_id: str, source_text: str) -> str:
    nct_ids = {
        match.group(0).upper()
        for match in _NCT_RE.finditer(f"{source_id} {source_text}")
    }
    if len(nct_ids) == 1:
        return f"STUDY:{next(iter(nct_ids))}"
    doi = _valid_doi(source_text)
    if doi:
        return f"STUDY:DOI:{doi}"
    if source_id.startswith("PMID:"):
        return f"STUDY:PMID:{source_id.removeprefix('PMID:')}"
    return f"STUDY:SOURCE:{source_id}"


def _evidence_excerpt(source_text: str, *, max_chars: int = 900) -> str:
    normalized = " ".join(source_text.replace("\r", " ").split())
    markers = ("RESULTS:", "Results:", "CONCLUSION:", "Conclusion:")
    starts = [normalized.find(marker) for marker in markers if normalized.find(marker) >= 0]
    excerpt = normalized[min(starts) :] if starts else normalized
    if len(excerpt) <= max_chars:
        return excerpt
    boundary = max(excerpt.rfind(". ", 0, max_chars), excerpt.rfind("。", 0, max_chars))
    return excerpt[: boundary + 1] if boundary >= max_chars // 2 else excerpt[:max_chars].rstrip()


def prediction_fact_from_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Create one structured prediction fact from an atomic prediction input."""

    if case.get("case_type") != "atomic_source_record":
        raise ValueError("prediction facts can only be built from atomic source records")
    if any(str(key).startswith("reference_") for key in case):
        raise ValueError("reference fields are forbidden in prediction-fact inputs")

    source_text = str(case.get("source_text", ""))
    source_id = str(case.get("source_id", ""))
    clinical_use = str(case.get("clinical_use", ""))
    tumor_candidates = list(case.get("candidate_tumors", []))
    radio_candidates = list(case.get("candidate_radiopharmaceuticals", []))
    publication_types = [str(value) for value in case.get("publication_types", [])]
    population_index_terms = [
        str(value) for value in case.get("population_index_terms", [])
    ]
    tumor_id = _single_candidate(tumor_candidates, "tumor_subtype_id")
    radio_id = _single_candidate(
        radio_candidates,
        "radiopharmaceutical_id",
    )

    human_match = _contains(_HUMAN_RE, source_text)
    nonhuman_match = _contains(_NONHUMAN_RE, source_text)
    strong_nonhuman_experiment = bool(
        _contains(_STRONG_PRECLINICAL_TITLE_RE, _title_from_source_text(source_text))
        or _contains(_STRONG_ANIMAL_EXPERIMENT_RE, source_text)
    )
    human_subjects: bool | None
    if strong_nonhuman_experiment and not human_match:
        human_subjects = False
    elif human_match:
        human_subjects = True
    elif nonhuman_match:
        human_subjects = False
    elif case.get("human_indexed") is True:
        human_subjects = True
    else:
        human_subjects = None
    oncology_context = True if _contains(_ONCOLOGY_RE, source_text) else None
    result_evidence_cues = _result_evidence_cues(source_text)
    result_reported = len(result_evidence_cues) >= 2
    use_alignment = _clinical_use_alignment(source_text, clinical_use)
    role = _source_role(
        text=source_text,
        source_id=source_id,
        human_subjects=human_subjects,
        oncology_context=oncology_context,
        result_reported=result_reported,
        clinical_use_alignment=use_alignment,
        source_type=str(case.get("source_type", "")),
        record_locator=str(case.get("record_locator", "")),
        publication_types=publication_types,
    )
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    fact_hash = hashlib.sha256(
        "|".join(
            (
                str(case.get("dataset_id", "")),
                str(case.get("case_id", "")),
                source_id,
                clinical_use,
                source_hash,
            )
        ).encode("utf-8")
    ).hexdigest()[:24]
    missing = [
        field
        for field, value in (
            ("human_subjects", human_subjects),
            ("oncology_context", oncology_context),
            ("result_reported", result_reported),
            ("clinical_use_alignment", use_alignment),
        )
        if value is None
    ]
    explicit_ineligibility_reasons: list[str] = []
    if role != "primary_human_result":
        explicit_ineligibility_reasons.append(f"source_role:{role}")
    if human_subjects is not True:
        explicit_ineligibility_reasons.append(
            "nonhuman_source"
            if human_subjects is False
            else "no_explicit_human_subject_evidence"
        )
    if oncology_context is not True:
        explicit_ineligibility_reasons.append("no_explicit_oncology_context")
    if result_reported is not True:
        explicit_ineligibility_reasons.append("insufficient_result_evidence_cues")
    if use_alignment is not True:
        explicit_ineligibility_reasons.append(
            "no_explicit_requested_clinical_use_alignment"
        )
    if bool(re.search(r"\b(?:retracted|retraction)\b", source_text, re.IGNORECASE)):
        explicit_ineligibility_reasons.append("retracted_source")
    if explicit_ineligibility_reasons:
        relation_decision_status = "clear_ineligible"
    elif not missing:
        relation_decision_status = "clear_eligible"
    else:
        relation_decision_status = "needs_arbiter"
    if tumor_id is None:
        missing.append("tumor_subtype_id")
    if radio_id is None:
        missing.append("radiopharmaceutical_id")

    return {
        "prediction_fact_schema_version": PREDICTION_FACT_SCHEMA_VERSION,
        "fact_id": f"PFACT:{fact_hash}",
        "origin_case_id": str(case["case_id"]),
        "origin_dataset_id": str(case["dataset_id"]),
        "source_id": source_id,
        "source_type": str(case.get("source_type", "not_reported")),
        "source_role": role,
        "record_locator": str(case.get("record_locator", "not_reported")),
        "evidence_public_date": str(case.get("evidence_public_date", "")),
        "study_family_id": _study_family_id(source_id, source_text),
        "clinical_use": clinical_use,
        "tumor_subtype_id": tumor_id,
        "radiopharmaceutical_id": radio_id,
        "tumor_candidate_ids": sorted(
            candidate_ids(tumor_candidates, "tumor_subtype_id")
        ),
        "radiopharmaceutical_candidate_ids": sorted(
            candidate_ids(radio_candidates, "radiopharmaceutical_id")
        ),
        "candidate_pair_bindings": _candidate_pair_bindings(
            source_text,
            tumor_candidates,
            radio_candidates,
        ),
        "human_subjects": human_subjects,
        "oncology_context": oncology_context,
        "result_reported": result_reported,
        "result_evidence_cues": result_evidence_cues,
        "result_evidence_cue_count": len(result_evidence_cues),
        "clinical_use_alignment": use_alignment,
        "retracted": bool(
            re.search(r"\b(?:retracted|retraction)\b", source_text, re.IGNORECASE)
        ),
        "design": _design_flags(source_text, clinical_use),
        "result": _result_fields(source_text, result_reported=result_reported),
        "applicability": {
            "population_scope": _population_scope(
                source_text,
                population_index_terms,
            )
        },
        "evidence_excerpt": _evidence_excerpt(source_text),
        "relation_decision_status": relation_decision_status,
        "explicit_ineligibility_reasons": explicit_ineligibility_reasons,
        "extraction_status": "complete" if not missing else "needs_arbiter",
        "missing_or_ambiguous_fields": sorted(missing),
        "prediction_provenance": {
            "extractor": "deterministic_raw_text_decision_table_v1.2",
            "source_text_sha256": source_hash,
            "candidate_interface_only": True,
            "structured_source_fact_read": False,
            "frozen_label_fields_read": False,
            "input_fields": sorted(
                {
                    "case_id",
                    "dataset_id",
                    "clinical_use",
                    "source_id",
                    "source_type",
                    "publication_types",
                    "population_index_terms",
                    "human_indexed",
                    "record_locator",
                    "evidence_public_date",
                    "source_text",
                    "candidate_tumors",
                    "candidate_radiopharmaceuticals",
                }
            ),
        },
    }


def build_prediction_facts(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    facts = [
        prediction_fact_from_case(case)
        for case in cases
        if case.get("case_type") == "atomic_source_record"
    ]
    facts.sort(key=lambda row: (row["origin_dataset_id"], row["origin_case_id"]))
    identifiers = [str(fact["fact_id"]) for fact in facts]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("prediction fact IDs are not unique")
    forbidden = [
        key
        for fact in facts
        for key in _nested_keys(fact)
        if key.startswith("reference_")
    ]
    if forbidden:
        raise RuntimeError(f"prediction facts leaked reference fields: {forbidden[:5]}")
    return facts


def prediction_fact_fingerprint(facts: Iterable[Mapping[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(dict(fact), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for fact in facts
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_prediction_safe_corpus(
    root: Path,
    *,
    source_details_path: Path | None = None,
    evidence_chunks_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return retrieval records whose text contains only title/abstract/DOI.

    The frozen evidence chunks append resolved entity IDs and source-role
    labels to their text.  Those lines are useful for graph construction but
    are not prediction-safe.  Retrieval baselines therefore receive a strict
    metadata whitelist and rebuild content from the raw PubMed detail record.
    """

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    source_details_path = source_details_path or (
        root / "evidence" / "executor_phase2" / "source_fact_details.jsonl"
    )
    evidence_chunks_path = evidence_chunks_path or (
        root / "data" / "seed" / "evidence_chunks.jsonl"
    )
    details = read_jsonl(source_details_path)
    detail_by_source: dict[str, dict[str, Any]] = {}
    for detail in details:
        source_id = str(detail.get("source_id", ""))
        if source_id and source_id not in detail_by_source:
            detail_by_source[source_id] = detail

    safe: list[dict[str, Any]] = []
    for record in read_jsonl(evidence_chunks_path):
        source_id = str(record["source_id"])
        detail = detail_by_source.get(source_id, {})
        text = (
            f"Source {source_id}\n"
            f"Title: {detail.get('title', '')}\n"
            f"Abstract: {detail.get('abstract', '')}\n"
            f"DOI: {detail.get('doi') or 'not reported'}"
        )
        safe.append(
            {
                "chunk_id": str(record["chunk_id"]),
                "dataset_membership": [
                    str(value) for value in record.get("dataset_membership", [])
                ],
                "evidence_public_date": str(
                    record.get("evidence_public_date", "")
                ),
                "record_locator": str(record.get("record_locator", "")),
                "source_id": source_id,
                "text": text,
            }
        )
    forbidden_markers = (
        "Resolved tumor subtype:",
        "Resolved radiopharmaceutical:",
        "Source role:",
        "Source level:",
    )
    if any(
        marker in str(record["text"])
        for record in safe
        for marker in forbidden_markers
    ):
        raise RuntimeError("prediction-safe corpus contains structured answer text")
    return safe


def _nested_keys(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_keys(nested)
