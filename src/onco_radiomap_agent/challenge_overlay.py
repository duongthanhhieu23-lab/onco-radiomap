"""Prediction-safe V1.1 challenge evidence overlay.

The challenge set is deliberately kept outside the frozen V1.0 LightRAG
stores.  Only upgraded source facts with both canonical entity identifiers are
materialized, and every virtual record belongs exclusively to the challenge
dataset.  This module contains no network or provider-backed code.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .entity_candidates import SharedCandidateResolver

CHALLENGE_DATASET_ID = "internal-challenge-pre2023-v1.1"
CHALLENGE_OVERLAY_CONTRACT_VERSION = "challenge-virtual-evidence-overlay-v1.1.0"
NAIVE_CHALLENGE_CONTRACT_VERSION = "naive-rag-challenge-overlay-v1.1.0"
VIRTUAL_CHUNK_PREFIX = f"VIRTUAL:{CHALLENGE_DATASET_ID}:"
MAX_VIRTUAL_RECORD_CHARS = 4000

CHALLENGE_CASES_RELATIVE_PATH = Path("data/v1.1/datasets/internal-challenge-pre2023-v1.1/cases.jsonl")
CHALLENGE_FACTS_RELATIVE_PATH = Path("evidence/v1.1/upgraded_challenge_source_facts.jsonl")
CHALLENGE_AUDIT_RELATIVE_PATH = Path("evidence/v1.1/upgraded_challenge_source_audit.jsonl")
SOURCE_DETAILS_RELATIVE_PATH = Path("evidence/executor_phase2/source_fact_details.jsonl")
AUXILIARY_REGISTRY_RELATIVE_PATH = Path("data/v1.1/registries/radiopharmaceutical_auxiliary.json")
ALIAS_EXCLUSIONS_RELATIVE_PATH = Path("data/v1.1/registries/alias_exclusions.json")
CHALLENGE_CONTRACT_RELATIVE_PATHS = (
    CHALLENGE_CASES_RELATIVE_PATH,
    CHALLENGE_FACTS_RELATIVE_PATH,
    CHALLENGE_AUDIT_RELATIVE_PATH,
    AUXILIARY_REGISTRY_RELATIVE_PATH,
    ALIAS_EXCLUSIONS_RELATIVE_PATH,
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+|(?<=[.!?])(?=[A-Z])")


@dataclass(frozen=True)
class ChallengeOverlay:
    """Loaded, validated virtual evidence for one isolated challenge dataset."""

    cases: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]
    records: tuple[dict[str, Any], ...]
    facts_by_id: Mapping[str, dict[str, Any]]
    facts_by_source: Mapping[str, tuple[dict[str, Any], ...]]
    facts_by_pair_use: Mapping[tuple[str, str, str], tuple[dict[str, Any], ...]]
    records_by_fact_id: Mapping[str, dict[str, Any]]
    records_by_chunk: Mapping[str, dict[str, Any]]
    source_ids: frozenset[str]


def challenge_contract_paths(root: Path) -> list[Path]:
    """Return inputs whose bytes define the challenge overlay contract."""

    return [root / relative for relative in CHALLENGE_CONTRACT_RELATIVE_PATHS]


def load_challenge_overlay(root: Path) -> ChallengeOverlay:
    """Build prediction inputs and virtual records from the current V1.1 files."""

    raw_cases = _read_jsonl(root / CHALLENGE_CASES_RELATIVE_PATH)
    all_facts = _read_jsonl(root / CHALLENGE_FACTS_RELATIVE_PATH)
    facts = [
        fact
        for fact in all_facts
        if str(fact.get("tumor_subtype_id") or "").strip() and str(fact.get("radiopharmaceutical_id") or "").strip()
    ]
    if not facts:
        raise RuntimeError("challenge overlay has no facts with both entity IDs")

    facts_by_id = _unique_index(facts, "fact_id", "challenge fact")
    details_by_id = _unique_index(
        _read_jsonl(root / SOURCE_DETAILS_RELATIVE_PATH),
        "fact_id",
        "source detail",
    )
    audits_by_id = _unique_index(
        _read_jsonl(root / CHALLENGE_AUDIT_RELATIVE_PATH),
        "fact_id",
        "challenge audit",
    )

    missing_details = sorted(set(facts_by_id).difference(details_by_id))
    missing_audits = sorted(set(facts_by_id).difference(audits_by_id))
    if missing_details:
        raise RuntimeError(f"challenge facts missing source details: {missing_details[:5]}")
    if missing_audits:
        raise RuntimeError(f"challenge facts missing upgrade audit rows: {missing_audits[:5]}")

    resolver = SharedCandidateResolver.from_project(root)
    prediction_cases: list[dict[str, Any]] = []
    atomic_fact_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for raw_case in raw_cases:
        dataset_id = str(raw_case.get("dataset_id", ""))
        if dataset_id != CHALLENGE_DATASET_ID:
            raise RuntimeError(f"challenge case has unexpected dataset_id={dataset_id!r}")
        case_id = str(raw_case["case_id"])
        if case_id in seen_case_ids:
            raise RuntimeError(f"duplicate challenge case_id {case_id}")
        seen_case_ids.add(case_id)
        common = {
            "case_id": case_id,
            "dataset_id": CHALLENGE_DATASET_ID,
            "case_type": str(raw_case["case_type"]),
            "clinical_use": str(raw_case["clinical_use"]),
            "cutoff_date": str(raw_case["cutoff_date"]),
        }
        if raw_case["case_type"] == "atomic_source_record":
            fact_id = str(raw_case.get("provenance", {}).get("source_fact_id", ""))
            fact = facts_by_id.get(fact_id)
            if fact is None:
                raise RuntimeError(f"challenge atomic case {case_id} does not reference a double-entity upgraded fact")
            atomic_fact_ids.add(fact_id)
            if str(raw_case.get("tumor_subtype_id")) != str(fact["tumor_subtype_id"]) or str(
                raw_case.get("radiopharmaceutical_id")
            ) != str(fact["radiopharmaceutical_id"]):
                raise RuntimeError(f"challenge atomic case/fact pair mismatch for {case_id}")
            detail = details_by_id[fact_id]
            source_text = _source_text(detail)
            row = {
                **common,
                "source_fact_id": fact_id,
                "source_id": str(fact["source_id"]),
                "evidence_public_date": str(
                    fact.get("evidence_public_date", "")
                ),
                "source_type": str(fact.get("source_type", "not_reported")),
                "record_locator": str(fact.get("record_locator") or detail.get("result_locator") or "not_reported"),
                "source_text": source_text,
                "publication_types": sorted(
                    {
                        str(item.get("name", "")).strip()
                        for item in detail.get("publication_types", [])
                        if str(item.get("name", "")).strip()
                    }
                ),
                "population_index_terms": [],
                "human_indexed": None,
                "retrieval_query": (f"{detail.get('title', '')} {detail.get('abstract', '')[:800]}").strip(),
                # These candidates remain source-text-derived.  The retriever's
                # challenge pair expansion is instead anchored to source_fact_id,
                # so ambiguous aliases cannot silently select a different pair.
                **resolver.resolve_source_text(source_text),
            }
        elif raw_case["case_type"] == "pair_snapshot":
            tumor_id = str(raw_case["tumor_subtype_id"])
            radio_id = str(raw_case["radiopharmaceutical_id"])
            population_scope = str(raw_case.get("provenance", {}).get("population_scope", "not_reported"))
            row = {
                **common,
                "tumor_subtype_id": tumor_id,
                "radiopharmaceutical_id": radio_id,
                "population_scope": population_scope,
                "candidate_tumors": [],
                "candidate_radiopharmaceuticals": [],
                "retrieval_query": (
                    f"{tumor_id} {radio_id} {raw_case['clinical_use']} human "
                    f"evidence population {population_scope} before "
                    f"{raw_case['cutoff_date']}"
                ),
            }
        else:
            raise RuntimeError(f"unsupported challenge case_type={raw_case['case_type']!r}")
        denied = [
            key
            for key in _nested_keys(row)
            if key.startswith(("reference_", "prediction_", "predicted_", "full_system_"))
        ]
        if denied:
            raise RuntimeError(f"challenge prediction input leaked fields: {denied}")
        prediction_cases.append(row)

    if not atomic_fact_ids:
        raise RuntimeError("challenge overlay has no atomic source cases")

    records: list[dict[str, Any]] = []
    facts_by_source_mutable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    facts_by_pair_use_mutable: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    records_by_fact_id: dict[str, dict[str, Any]] = {}
    records_by_chunk: dict[str, dict[str, Any]] = {}
    for fact in sorted(facts, key=lambda row: str(row["fact_id"])):
        fact_id = str(fact["fact_id"])
        source_id = str(fact["source_id"])
        chunk_id = f"{VIRTUAL_CHUNK_PREFIX}{fact_id}"
        record = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_fact_id": fact_id,
            "study_family_id": str(fact.get("study_family_id", "")),
            "source_role": str(fact.get("source_role", "not_reported")),
            "tumor_subtype_id": str(fact["tumor_subtype_id"]),
            "radiopharmaceutical_id": str(fact["radiopharmaceutical_id"]),
            "clinical_use": str(fact["clinical_use"]),
            "evidence_public_date": str(fact.get("evidence_public_date", "")),
            "population_membership": [str(fact.get("applicability", {}).get("population_scope", "not_reported"))],
            "dataset_membership": [CHALLENGE_DATASET_ID],
            "virtual_overlay_dataset_id": CHALLENGE_DATASET_ID,
            "record_locator": str(fact.get("record_locator", "not_reported")),
            "text": _virtual_record_text(
                fact=fact,
                detail=details_by_id[fact_id],
                audit=audits_by_id[fact_id],
                chunk_id=chunk_id,
            ),
        }
        if len(record["text"]) > MAX_VIRTUAL_RECORD_CHARS:
            raise RuntimeError(f"virtual challenge block exceeds budget: {fact_id}")
        records.append(record)
        records_by_fact_id[fact_id] = record
        records_by_chunk[chunk_id] = record
        facts_by_source_mutable[source_id].append(fact)
        facts_by_pair_use_mutable[
            (
                str(fact["tumor_subtype_id"]),
                str(fact["radiopharmaceutical_id"]),
                str(fact["clinical_use"]),
            )
        ].append(fact)

    return ChallengeOverlay(
        cases=tuple(
            sorted(
                prediction_cases,
                key=lambda row: (str(row["dataset_id"]), str(row["case_id"])),
            )
        ),
        facts=tuple(sorted(facts, key=lambda row: str(row["fact_id"]))),
        records=tuple(records),
        facts_by_id=facts_by_id,
        facts_by_source={
            key: tuple(sorted(value, key=lambda row: str(row["fact_id"])))
            for key, value in facts_by_source_mutable.items()
        },
        facts_by_pair_use={
            key: tuple(sorted(value, key=lambda row: str(row["fact_id"])))
            for key, value in facts_by_pair_use_mutable.items()
        },
        records_by_fact_id=records_by_fact_id,
        records_by_chunk=records_by_chunk,
        source_ids=frozenset(facts_by_source_mutable),
    )


def challenge_naive_result(
    case: Mapping[str, Any],
    overlay: ChallengeOverlay,
    *,
    top_k: int = 4,
    max_chars: int,
) -> dict[str, Any]:
    """Retrieve a whole-block BM25 context from only the challenge overlay."""

    from .naive_rag import BM25EvidenceIndex

    if str(case.get("dataset_id")) != CHALLENGE_DATASET_ID:
        raise ValueError("challenge_naive_result requires a challenge case")
    eligible = _eligible_records(case, overlay)
    index = BM25EvidenceIndex(eligible)
    hits = index.search(
        str(case["retrieval_query"]),
        dataset_id=CHALLENGE_DATASET_ID,
        top_k=max(len(eligible), top_k),
    )
    score_by_chunk = {str(hit.record["chunk_id"]): float(hit.score) for hit in hits}
    ordered = [hit.record for hit in hits]
    if case["case_type"] == "atomic_source_record":
        own = overlay.records_by_fact_id[str(case["source_fact_id"])]
        ordered = [own, *(record for record in ordered if record is not own)]

    selected: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    seen_families: set[str] = set()
    for record in ordered:
        chunk_id = str(record["chunk_id"])
        family_id = str(record.get("study_family_id", ""))
        if chunk_id in seen_chunks or family_id in seen_families:
            continue
        seen_chunks.add(chunk_id)
        if family_id:
            seen_families.add(family_id)
        selected.append(record)
        if len(selected) >= top_k:
            break

    blocks: list[str] = []
    included: list[dict[str, Any]] = []
    dropped = 0
    for rank, record in enumerate(selected, start=1):
        chunk_id = str(record["chunk_id"])
        header = (
            f"[BM25 overlay rank={rank} score={score_by_chunk.get(chunk_id, 0.0):.4f}; "
            f"virtual_chunk={chunk_id}; study_family="
            f"{record.get('study_family_id', 'not_reported')}]"
        )
        block = f"{header}\n{record['text']}"
        current_chars = sum(len(value) for value in blocks) + 2 * max(len(blocks) - 1, 0)
        separator_chars = 2 if blocks else 0
        if current_chars + separator_chars + len(block) > max_chars:
            dropped += 1
            continue
        blocks.append(block)
        included.append(record)
    context = "\n\n".join(blocks)
    pair = _case_explicit_pair(case, overlay)
    return {
        "context": context,
        "retrieved_chunk_ids": [str(record["chunk_id"]) for record in included],
        "retrieved_source_ids": list(dict.fromkeys(str(record["source_id"]) for record in included)),
        "retrieved_study_family_ids": list(
            dict.fromkeys(
                str(record["study_family_id"]) for record in included if str(record.get("study_family_id", ""))
            )
        ),
        "source_roles": [str(record["source_role"]) for record in included],
        "retrieved_origin_dataset_ids": ([CHALLENGE_DATASET_ID] if included else []),
        "explicit_pair": list(pair),
        "dropped_for_budget_count": dropped,
        "hard_truncated": False,
        "context_chars": len(context),
        "max_chars": max_chars,
        "overlay_dataset_id": CHALLENGE_DATASET_ID,
        "retrieval_contract_version": NAIVE_CHALLENGE_CONTRACT_VERSION,
    }


def _eligible_records(case: Mapping[str, Any], overlay: ChallengeOverlay) -> list[dict[str, Any]]:
    tumor_id, radio_id = _case_explicit_pair(case, overlay)
    clinical_use = str(case["clinical_use"])
    cutoff_date = str(case["cutoff_date"])
    population_scope = str(case["population_scope"]) if case["case_type"] == "pair_snapshot" else None
    result: list[dict[str, Any]] = []
    for fact in overlay.facts_by_pair_use.get((tumor_id, radio_id, clinical_use), ()):
        if bool(fact.get("retracted")):
            continue
        public_date = str(fact.get("evidence_public_date") or "")
        if not public_date or public_date > cutoff_date:
            continue
        if (
            population_scope is not None
            and str(fact.get("applicability", {}).get("population_scope", "not_reported")) != population_scope
        ):
            continue
        result.append(overlay.records_by_fact_id[str(fact["fact_id"])])
    return sorted(result, key=lambda record: str(record["chunk_id"]))


def _case_explicit_pair(case: Mapping[str, Any], overlay: ChallengeOverlay) -> tuple[str, str]:
    if case["case_type"] == "pair_snapshot":
        return (
            str(case["tumor_subtype_id"]),
            str(case["radiopharmaceutical_id"]),
        )
    fact = overlay.facts_by_id.get(str(case["source_fact_id"]))
    if fact is None:
        raise RuntimeError(f"challenge atomic case has unknown fact {case['source_fact_id']}")
    return str(fact["tumor_subtype_id"]), str(fact["radiopharmaceutical_id"])


def _virtual_record_text(
    *,
    fact: Mapping[str, Any],
    detail: Mapping[str, Any],
    audit: Mapping[str, Any],
    chunk_id: str,
) -> str:
    population = str(fact.get("applicability", {}).get("population_scope", "not_reported"))
    essential = [
        f"Source {fact['source_id']}",
        f"Virtual chunk: {chunk_id}",
        f"Title: {_clean(detail.get('title') or 'not reported')}",
        f"Record locator: {fact.get('record_locator', 'not_reported')}",
        f"Public date: {fact.get('evidence_public_date', 'not_reported')}",
        (f"Explicit entity pair: {fact['tumor_subtype_id']}|{fact['radiopharmaceutical_id']}"),
        f"Clinical use: {fact['clinical_use']}",
        f"Population: {population}",
        f"Study family: {fact.get('study_family_id', 'not_reported')}",
        f"Source role: {fact.get('source_role', 'not_reported')}",
    ]
    lines = list(essential)
    evidence_lines: list[str] = []
    seen_evidence: set[str] = set()
    traces = audit.get("evidence_traces", {})
    if isinstance(traces, Mapping):
        for field, trace in sorted(traces.items(), key=lambda item: str(item[0])):
            if not isinstance(trace, Mapping):
                continue
            sentence = _clean(trace.get("evidence_sentence"))
            if not sentence or sentence in seen_evidence:
                continue
            seen_evidence.add(sentence)
            section = _clean(trace.get("section") or "not_reported")
            evidence_lines.append(f"- {field} [{section}]: {sentence}")
    if not evidence_lines:
        for snippet in detail.get("result_snippets", []):
            if not isinstance(snippet, Mapping):
                continue
            sentence = _clean(snippet.get("text"))
            if sentence and sentence not in seen_evidence:
                seen_evidence.add(sentence)
                evidence_lines.append(f"- result [{_clean(snippet.get('section') or 'not_reported')}]: {sentence}")
    if not evidence_lines:
        abstract = _clean(detail.get("abstract"))
        if abstract:
            evidence_lines.append(f"- abstract: {abstract}")

    if evidence_lines:
        lines.append("Source-grounded evidence traces:")
        _append_complete_lines(lines, evidence_lines, MAX_VIRTUAL_RECORD_CHARS)
    true_design_fields = sorted(str(key) for key, value in fact.get("design", {}).items() if value is True)
    tail = [
        "Reported design flags: " + (", ".join(true_design_fields) if true_design_fields else "none reported"),
        "Reported result fields: " + json.dumps(fact.get("result", {}), ensure_ascii=False, sort_keys=True),
    ]
    _append_complete_lines(lines, tail, MAX_VIRTUAL_RECORD_CHARS)
    return "\n".join(lines)


def _append_complete_lines(lines: list[str], candidates: Iterable[str], max_chars: int) -> None:
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        current = len("\n".join(lines))
        if current + 1 + len(candidate) <= max_chars:
            lines.append(candidate)
            continue
        remaining = max_chars - current - 1
        excerpt = _complete_sentence_excerpt(candidate, max_chars=remaining)
        if excerpt:
            lines.append(excerpt)


def _complete_sentence_excerpt(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    selected: list[str] = []
    current = 0
    for sentence in _SENTENCE_BOUNDARY_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        separator = 1 if selected else 0
        if current + separator + len(sentence) > max_chars:
            continue
        selected.append(sentence)
        current += separator + len(sentence)
    return " ".join(selected)


def _source_text(detail: Mapping[str, Any]) -> str:
    return (
        f"Title: {detail.get('title', '')}\n"
        f"Abstract: {detail.get('abstract', '')}\n"
        f"DOI: {detail.get('doi') or 'not reported'}"
    )


def _clean(value: object | None) -> str:
    return " ".join(str(value or "").split())


def _unique_index(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise RuntimeError(f"duplicate {label} {value}")
        result[value] = row
    return result


def _nested_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_keys(nested)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
