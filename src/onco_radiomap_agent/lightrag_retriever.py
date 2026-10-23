"""Read-only, scope-safe retrieval over the frozen LightRAG graph and stores.

V1.1 deliberately treats the graph as a pair index, not as permission to walk
arbitrary neighbours. Every emitted source block is re-checked against the
dataset, population, cutoff, clinical-use, and study-family scope before it is
shown to a model. Blocks are packed whole; this module never slices a context
at an arbitrary character boundary.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .baseline_data import DATASET_IDS, load_prediction_inputs
from .challenge_overlay import CHALLENGE_DATASET_ID, load_challenge_overlay
from .embedding_client import ArkVisionEmbedding
from .prediction_facts import build_prediction_facts, load_prediction_safe_corpus
from .prediction_rules import (
    population_scope_compatible,
    query_pair_candidate_status,
)

RETRIEVAL_CONTRACT_VERSION = "lightrag-atomic-self-pair-attribution-design-v1.2.4"
DEFAULT_MAX_CHARS = 10000
DEFAULT_TOP_K = 40
AMBIGUOUS_FAMILY_RESERVE = 8
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
_DESIGN_SIGNAL_ORDER = (
    "prospective",
    "multicenter",
    "randomized",
    "controlled",
    "credible_comparator",
    "phase_iii",
    "first_in_human",
    "case_report_or_small_series",
    "single_arm_defined_cohort",
    "safety_reported",
    "biodistribution_or_dosimetry_reported",
    "prespecified_diagnostic_performance",
    "gold_standard_comparison",
    "head_to_head_tracer_comparison",
    "management_impact_evaluated",
    "patient_relevant_diagnostic_utility_endpoint",
    "prespecified_objective_efficacy",
    "patient_relevant_treatment_endpoint",
    "target_visualization_reported",
)


@dataclass(frozen=True)
class EvidenceBlock:
    text: str
    source_id: str
    study_family_id: str
    chunk_id: str
    source_role: str
    origin_dataset_id: str = ""
    pair_match_basis: str = "not_applicable"
    rule_aggregation_eligible: bool = False


def pack_complete_blocks(
    blocks: Iterable[EvidenceBlock],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[str, list[EvidenceBlock], int]:
    """Pack whole blocks and return context, included blocks, and drop count."""

    included: list[EvidenceBlock] = []
    current_chars = 0
    dropped = 0
    for block in blocks:
        separator_chars = 2 if included else 0
        if current_chars + separator_chars + len(block.text) > max_chars:
            dropped += 1
            continue
        included.append(block)
        current_chars += separator_chars + len(block.text)
    return "\n\n".join(block.text for block in included), included, dropped


def _complete_sentence_excerpt(text: str, *, max_chars: int) -> str:
    """Create a bounded excerpt without cutting a line or sentence in half."""

    normalized = text.replace("\r\n", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    units: list[str] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Title:", "DOI:")):
            units.append(line)
            continue
        if line.startswith("Abstract:"):
            prefix, body = "Abstract:", line[len("Abstract:") :].strip()
            units.append(prefix)
            units.extend(part.strip() for part in _SENTENCE_BOUNDARY_RE.split(body) if part.strip())
            continue
        units.extend(part.strip() for part in _SENTENCE_BOUNDARY_RE.split(line) if part.strip())

    selected: list[str] = []
    current = 0
    for unit in units:
        separator = 1 if selected else 0
        if current + separator + len(unit) > max_chars:
            continue
        selected.append(unit)
        current += separator + len(unit)
    return "\n".join(selected)


def _date_on_or_before(value: object, cutoff_date: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and text <= cutoff_date


def _source_role_rank(role: str) -> int:
    return {
        "primary_human_result": 0,
        "regulatory_record": 1,
        "guideline_recommendation": 2,
        "clinical_trial_registration": 3,
        "systematic_review_or_meta_analysis": 4,
        "review_or_discovery": 5,
    }.get(role, 6)


def _design_signal_summary(fact: Mapping[str, Any]) -> str:
    """Serialize only positive, raw-text-derived design cues for the LLM.

    These cues are prediction-side metadata, not reference labels.  Omitting a
    signal means that it was not explicit in the available title/abstract; it
    must not be interpreted as a proven negative study-design fact.
    """

    design = fact.get("design", {})
    if not isinstance(design, Mapping):
        return "none_reported"
    selected = [signal for signal in _DESIGN_SIGNAL_ORDER if design.get(signal) is True]
    return ",".join(selected) if selected else "none_reported"


class FrozenLightRAGRetriever:
    """Use the frozen graph as an exact-pair index with optional vector reranking."""

    def __init__(
        self,
        root: Path,
        *,
        prediction_facts: list[dict[str, Any]] | None = None,
        source_details_path: Path | None = None,
        evidence_chunks_path: Path | None = None,
        graph_mapping_chunks_path: Path | None = None,
    ) -> None:
        self.root = root
        self.index_dir = root / "data" / "processed" / "lightrag_index"
        self.vdb = NanoVectorDB(
            embedding_dim=2048,
            metric="cosine",
            storage_file=str(self.index_dir / "vdb_chunks.json"),
        )
        self.graph = nx.read_graphml(self.index_dir / "graph_chunk_entity_relation.graphml")
        self.text_chunks: dict[str, dict[str, Any]] = json.loads(
            (self.index_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8")
        )
        self.full_docs: dict[str, dict[str, Any]] = json.loads(
            (self.index_dir / "kv_store_full_docs.json").read_text(encoding="utf-8")
        )
        self.corpus = load_prediction_safe_corpus(
            root,
            source_details_path=source_details_path,
            evidence_chunks_path=evidence_chunks_path,
        )
        raw_graph_corpus = _read_jsonl(
            graph_mapping_chunks_path
            or (root / "data" / "seed" / "evidence_chunks.jsonl")
        )
        # Pair eligibility, entities, design, and results come only from the
        # answer-free raw-text prediction facts.  The frozen GraphML and corpus
        # below are used solely to map an already eligible source to its vector
        # chunk for optional within-scope reranking.
        self.facts = prediction_facts or build_prediction_facts(
            [
                *load_prediction_inputs(root, dataset_ids=DATASET_IDS),
                *load_challenge_overlay(root).cases,
            ]
        )

        records_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Exact text equality is needed only to recover the frozen vector
        # chunk -> source mapping.  Raw graph text is never rendered or passed
        # to a model.
        for record in raw_graph_corpus:
            records_by_text[str(record.get("text", ""))].append(record)
        self.source_ids_by_chunk: dict[str, frozenset[str]] = {}
        self.chunk_ids_by_source: dict[str, set[str]] = defaultdict(set)
        for chunk_id, chunk in self.text_chunks.items():
            document = self.full_docs.get(str(chunk.get("full_doc_id", "")), {})
            records = tuple(records_by_text.get(str(document.get("content", "")), []))
            source_ids = frozenset(str(record["source_id"]) for record in records)
            self.source_ids_by_chunk[chunk_id] = source_ids
            for source_id in source_ids:
                self.chunk_ids_by_source[source_id].add(chunk_id)

        self.facts_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.facts_by_pair_use: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.ambiguous_facts_by_pair_use: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for fact in self.facts:
            self.facts_by_source[str(fact["source_id"])].append(fact)
            clinical_use = fact.get("clinical_use")
            tumor_ids = set(fact.get("tumor_candidate_ids", []))
            radio_ids = set(fact.get("radiopharmaceutical_candidate_ids", []))
            if fact.get("tumor_subtype_id"):
                tumor_ids.add(str(fact["tumor_subtype_id"]))
            if fact.get("radiopharmaceutical_id"):
                radio_ids.add(str(fact["radiopharmaceutical_id"]))
            direct_pairs: set[tuple[str, str]] = set()
            if fact.get("tumor_subtype_id") and fact.get("radiopharmaceutical_id"):
                direct_pairs.add(
                    (
                        str(fact["tumor_subtype_id"]),
                        str(fact["radiopharmaceutical_id"]),
                    )
                )
            direct_pairs.update(
                (
                    str(binding.get("tumor_subtype_id")),
                    str(binding.get("radiopharmaceutical_id")),
                )
                for binding in fact.get("candidate_pair_bindings", [])
                if binding.get("tumor_subtype_id") and binding.get("radiopharmaceutical_id")
            )
            if not clinical_use:
                continue
            for tumor_id, radio_id in direct_pairs:
                self.facts_by_pair_use[(tumor_id, radio_id, str(clinical_use))].append(fact)
            for tumor_id in tumor_ids:
                for radio_id in radio_ids:
                    pair = (str(tumor_id), str(radio_id))
                    if pair in direct_pairs:
                        continue
                    self.ambiguous_facts_by_pair_use[(pair[0], pair[1], str(clinical_use))].append(fact)

        unmapped = [chunk_id for chunk_id, source_ids in self.source_ids_by_chunk.items() if not source_ids]
        if unmapped:
            raise RuntimeError(f"LightRAG source mapping failed for {len(unmapped)} frozen chunks")
        # Keep deterministic graph retrieval usable without credentials. The
        # Ark client is constructed only when vector reranking is requested.
        self.embedder: ArkVisionEmbedding | None = None

    def embed_queries(self, queries: list[str]) -> list[np.ndarray]:
        if self.embedder is None:
            # V1.1 provider runs are never-called-only and zero-retry: each
            # query is billed at most once, including CJK input failures.
            self.embedder = ArkVisionEmbedding(
                max_attempts=1,
                allow_cjk_fallback=False,
            )
        return [np.asarray(row, dtype=np.float32) for row in self.embedder.embed(queries)]

    def retrieve_case(
        self,
        case: Mapping[str, Any],
        *,
        query_vector: np.ndarray | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        """Retrieve a case-scoped context plus an auditable trace."""

        if case["case_type"] == "atomic_source_record":
            blocks, graph_pair = self._atomic_blocks(
                case,
                query_vector=query_vector,
                top_k=top_k,
                max_chars=max_chars,
            )
        else:
            blocks, graph_pair = self._snapshot_blocks(
                case,
                query_vector=query_vector,
                top_k=top_k,
                max_chars=max_chars,
            )
        context, included, dropped = pack_complete_blocks(blocks, max_chars=max_chars)
        dataset_id = str(case["dataset_id"])
        candidate_family_ids = list(dict.fromkeys(block.study_family_id for block in blocks if block.study_family_id))
        retrieved_family_ids = list(dict.fromkeys(block.study_family_id for block in included if block.study_family_id))
        return {
            "context": context,
            "retrieved_chunk_ids": [block.chunk_id for block in included],
            "retrieved_source_ids": list(dict.fromkeys(block.source_id for block in included)),
            "candidate_study_family_ids": candidate_family_ids,
            "retrieved_study_family_ids": retrieved_family_ids,
            "dropped_study_family_ids": [
                family_id for family_id in candidate_family_ids if family_id not in set(retrieved_family_ids)
            ],
            "candidate_family_count": len(candidate_family_ids),
            "packed_family_count": len(retrieved_family_ids),
            "source_roles": [block.source_role for block in included],
            "pair_match_bases": [block.pair_match_basis for block in included],
            "direct_aggregation_source_ids": list(
                dict.fromkeys(block.source_id for block in included if block.rule_aggregation_eligible)
            ),
            "ambiguous_attribution_source_ids": list(
                dict.fromkeys(
                    block.source_id
                    for block in included
                    if block.pair_match_basis == "ambiguous_candidate_pair_for_llm"
                )
            ),
            "retrieved_origin_dataset_ids": list(
                dict.fromkeys(block.origin_dataset_id for block in included if block.origin_dataset_id)
            ),
            "graph_pair": graph_pair,
            "dropped_for_budget_count": dropped,
            "hard_truncated": False,
            "context_chars": len(context),
            "max_chars": max_chars,
            "top_k": top_k,
            "ambiguous_family_reserve": AMBIGUOUS_FAMILY_RESERVE,
            "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
            "vector_reranking_used": (query_vector is not None and dataset_id != CHALLENGE_DATASET_ID),
            "overlay_dataset_id": (CHALLENGE_DATASET_ID if dataset_id == CHALLENGE_DATASET_ID else None),
        }

    def _atomic_blocks(
        self,
        case: Mapping[str, Any],
        *,
        query_vector: np.ndarray | None,
        top_k: int,
        max_chars: int,
    ) -> tuple[list[EvidenceBlock], list[str] | None]:
        source_id = str(case["source_id"])
        source_facts = [
            fact
            for fact in self.facts_by_source.get(source_id, [])
            if str(fact.get("origin_dataset_id")) == str(case["dataset_id"])
        ]
        own_fact = next(
            (
                fact
                for fact in source_facts
                if str(fact.get("origin_case_id")) == str(case["case_id"])
                and str(fact.get("clinical_use")) == str(case["clinical_use"])
                and _date_on_or_before(fact.get("evidence_public_date"), str(case["cutoff_date"]))
            ),
            source_facts[0] if source_facts else {},
        )
        own_header = (
            f"[LightRAG source SELF:{source_id}; study_family="
            f"{own_fact.get('study_family_id', 'not_reported')}; source_role="
            f"{own_fact.get('source_role', 'not_reported')}; "
            f"record_locator={own_fact.get('record_locator', case.get('record_locator', 'not_reported'))}; "
            "content_mode=original_or_sentence_complete_excerpt]"
        )
        body_budget = max(max_chars - len(own_header) - 1, 256)
        body = _complete_sentence_excerpt(str(case.get("source_text", "")), max_chars=body_budget)
        own_block = EvidenceBlock(
            text=f"{own_header}\n{body}",
            source_id=source_id,
            study_family_id=str(own_fact.get("study_family_id", "")),
            chunk_id=f"SELF:{source_id}",
            source_role=str(own_fact.get("source_role", "not_reported")),
            origin_dataset_id=str(case["dataset_id"]),
        )

        # Atomic relation/entity/conclusion scoring concerns this source record
        # alone.  Pair-neighbour expansion adds unrelated studies and can leak
        # aggregate evidence into an atomic decision, even when both candidates
        # happen to be unique.  Vector reranking is therefore intentionally
        # unused for atomic cases.
        del query_vector, top_k
        return [own_block], None

    def _snapshot_blocks(
        self,
        case: Mapping[str, Any],
        *,
        query_vector: np.ndarray | None,
        top_k: int,
        max_chars: int,
    ) -> tuple[list[EvidenceBlock], list[str]]:
        tumor_id = str(case["tumor_subtype_id"])
        radio_id = str(case["radiopharmaceutical_id"])
        return (
            self._pair_blocks(
                tumor_id=tumor_id,
                radiopharmaceutical_id=radio_id,
                clinical_use=str(case["clinical_use"]),
                dataset_id=str(case["dataset_id"]),
                population_scope=str(case["population_scope"]),
                cutoff_date=str(case["cutoff_date"]),
                query_vector=query_vector,
                top_k=top_k,
                max_chars=max_chars,
                excluded_source_ids=set(),
            ),
            [tumor_id, radio_id],
        )

    def _pair_blocks(
        self,
        *,
        tumor_id: str,
        radiopharmaceutical_id: str,
        clinical_use: str,
        dataset_id: str,
        population_scope: str | None,
        cutoff_date: str,
        query_vector: np.ndarray | None,
        top_k: int,
        max_chars: int,
        excluded_source_ids: set[str],
    ) -> list[EvidenceBlock]:
        if top_k <= 0:
            return []
        pair_key = (tumor_id, radiopharmaceutical_id, clinical_use)
        candidate_facts = [(fact, True) for fact in self.facts_by_pair_use.get(pair_key, [])]
        candidate_facts.extend((fact, False) for fact in self.ambiguous_facts_by_pair_use.get(pair_key, []))
        candidates: list[tuple[str, str, dict[str, Any], str, bool]] = []
        seen_fact_ids: set[str] = set()
        for fact, direct_aggregation in candidate_facts:
            fact_id = str(fact.get("fact_id", ""))
            if fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact_id)
            source_id = str(fact["source_id"])
            if source_id in excluded_source_ids:
                continue
            if str(fact.get("origin_dataset_id")) != dataset_id:
                continue
            if fact.get("relation_decision_status") == "clear_ineligible":
                continue
            if bool(fact.get("retracted")) or not _date_on_or_before(
                fact.get("evidence_public_date"),
                cutoff_date,
            ):
                continue
            if population_scope is not None and not population_scope_compatible(
                fact.get("applicability", {}).get("population_scope"),
                population_scope,
            ):
                continue
            mapped_chunks = sorted(self.chunk_ids_by_source.get(source_id, set()))
            chunk_id = mapped_chunks[0] if mapped_chunks else str(fact["fact_id"])
            pair_match_basis = query_pair_candidate_status(
                fact,
                tumor_subtype_id=tumor_id,
                radiopharmaceutical_id=radiopharmaceutical_id,
            )
            if pair_match_basis is None:
                continue
            candidates.append(
                (
                    chunk_id,
                    source_id,
                    fact,
                    pair_match_basis,
                    direct_aggregation,
                )
            )

        eligible_chunk_ids = {chunk_id for chunk_id, _, _, _, _ in candidates if chunk_id in self.text_chunks}
        vector_scores: dict[str, float] = {}
        if query_vector is not None and eligible_chunk_ids and dataset_id != CHALLENGE_DATASET_ID:
            hits = self.vdb.query(
                query_vector,
                top_k=min(max(len(eligible_chunk_ids), top_k), 64),
                filter_lambda=lambda row: str(row.get("__id__", "")) in eligible_chunk_ids,
            )
            vector_scores = {str(hit["__id__"]): float(hit.get("__metrics__", 0.0)) for hit in hits}

        candidates.sort(
            key=lambda item: (
                0 if item[4] else 1,
                _source_role_rank(str(item[2].get("source_role", ""))),
                -vector_scores.get(item[0], 0.0),
                str(item[2].get("evidence_public_date", "")),
                item[1],
                item[0],
            )
        )
        direct_selected: list[tuple[str, str, dict[str, Any], str, bool]] = []
        ambiguous_selected: list[tuple[str, str, dict[str, Any], str, bool]] = []
        seen_families: set[str] = set()
        for candidate in candidates:
            family_id = str(candidate[2].get("study_family_id", ""))
            if family_id in seen_families:
                continue
            seen_families.add(family_id)
            (direct_selected if candidate[4] else ambiguous_selected).append(candidate)
        ambiguous_quota = min(
            AMBIGUOUS_FAMILY_RESERVE,
            len(ambiguous_selected),
            top_k,
        )
        direct_quota = min(len(direct_selected), top_k - ambiguous_quota)
        selected = direct_selected[:direct_quota] + ambiguous_selected[:ambiguous_quota]
        remaining = top_k - len(selected)
        if remaining:
            selected.extend(direct_selected[direct_quota : direct_quota + remaining])
            remaining = top_k - len(selected)
        if remaining:
            selected.extend(ambiguous_selected[ambiguous_quota : ambiguous_quota + remaining])

        return self._render_snapshot_candidates(selected, max_chars=max_chars)

    def render_snapshot_source_order(
        self,
        case: Mapping[str, Any],
        source_family_chunk_order: Iterable[tuple[str, str, str]],
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> list[EvidenceBlock]:
        """Render a previously audited source order without a provider call.

        This is valid only for render-only contract changes.  Every requested
        source/family/chunk triple is rechecked against the current pair,
        dataset, clinical-use, population, cutoff, and attribution rules.
        """

        if case.get("case_type") != "pair_snapshot":
            raise ValueError("source-order rendering is available only for pair snapshots")
        tumor_id = str(case["tumor_subtype_id"])
        radio_id = str(case["radiopharmaceutical_id"])
        dataset_id = str(case["dataset_id"])
        clinical_use = str(case["clinical_use"])
        population_scope = str(case["population_scope"])
        cutoff_date = str(case["cutoff_date"])
        selected: list[tuple[str, str, dict[str, Any], str, bool]] = []
        for source_id, family_id, chunk_id in source_family_chunk_order:
            matches: list[tuple[dict[str, Any], str, bool]] = []
            for fact in self.facts_by_source.get(source_id, []):
                if str(fact.get("study_family_id", "")) != family_id:
                    continue
                if str(fact.get("origin_dataset_id", "")) != dataset_id:
                    continue
                if str(fact.get("clinical_use", "")) != clinical_use:
                    continue
                if fact.get("relation_decision_status") == "clear_ineligible":
                    continue
                if bool(fact.get("retracted")) or not _date_on_or_before(fact.get("evidence_public_date"), cutoff_date):
                    continue
                if not population_scope_compatible(
                    fact.get("applicability", {}).get("population_scope"),
                    population_scope,
                ):
                    continue
                pair_match_basis = query_pair_candidate_status(
                    fact,
                    tumor_subtype_id=tumor_id,
                    radiopharmaceutical_id=radio_id,
                )
                if pair_match_basis is None:
                    continue
                direct_aggregation = pair_match_basis != "ambiguous_candidate_pair_for_llm"
                matches.append((fact, pair_match_basis, direct_aggregation))
            if len(matches) != 1:
                raise RuntimeError(
                    "prior source/family no longer resolves to exactly one current fact: "
                    f"source={source_id!r}, family={family_id!r}, matches={len(matches)}"
                )
            fact, pair_match_basis, direct_aggregation = matches[0]
            mapped_chunks = self.chunk_ids_by_source.get(source_id, set())
            prediction_fact_chunk = not mapped_chunks and chunk_id == str(
                fact.get("fact_id", "")
            )
            if chunk_id not in mapped_chunks and not prediction_fact_chunk:
                raise RuntimeError(
                    f"prior chunk {chunk_id!r} no longer maps to source {source_id!r}"
                )
            selected.append((chunk_id, source_id, fact, pair_match_basis, direct_aggregation))
        return self._render_snapshot_candidates(selected, max_chars=max_chars)

    @staticmethod
    def _render_snapshot_candidates(
        selected: list[tuple[str, str, dict[str, Any], str, bool]],
        *,
        max_chars: int,
    ) -> list[EvidenceBlock]:
        """Render selected pair candidates while keeping every block whole."""

        headers: list[str] = []
        compact_family_index = len(selected) > 20
        for _, source_id, fact, pair_match_basis, direct_aggregation in selected:
            aggregation_flag = str(direct_aggregation).casefold()
            design_signals = _design_signal_summary(fact)
            if compact_family_index:
                headers.append(
                    f"[LightRAG source_id={source_id};family={fact.get('study_family_id', 'not_reported')};"
                    f"role={fact.get('source_role', 'not_reported')};"
                    f"date={fact.get('evidence_public_date', 'not_reported')};"
                    f"pair_match_basis={pair_match_basis};"
                    f"rule_aggregation_eligible={aggregation_flag};"
                    f"raw_text_design_signals={design_signals}]"
                )
            else:
                headers.append(
                    f"[LightRAG source_id={source_id}; family={fact.get('study_family_id', 'not_reported')}; "
                    f"role={fact.get('source_role', 'not_reported')}; "
                    f"date={fact.get('evidence_public_date', 'not_reported')}; "
                    f"pair_match_basis={pair_match_basis}; "
                    f"rule_aggregation_eligible={aggregation_flag}; "
                    f"raw_text_design_signals={design_signals}; "
                    f"locator={fact.get('record_locator', 'not_reported')}]"
                )
        separator_chars = max(len(headers) - 1, 0) * 2
        body_budget_total = max(
            max_chars - sum(len(header) + 1 for header in headers) - separator_chars,
            0,
        )
        body_budget_each = body_budget_total // len(headers) if headers else 0
        blocks: list[EvidenceBlock] = []
        for header, (
            chunk_id,
            source_id,
            fact,
            pair_match_basis,
            direct_aggregation,
        ) in zip(headers, selected, strict=True):
            family_id = str(fact.get("study_family_id", "not_reported"))
            role = str(fact.get("source_role", "not_reported"))
            body = _complete_sentence_excerpt(
                str(fact.get("evidence_excerpt", "")),
                max_chars=body_budget_each,
            )
            blocks.append(
                EvidenceBlock(
                    text=f"{header}\n{body}" if body else header,
                    source_id=source_id,
                    study_family_id=family_id,
                    chunk_id=chunk_id,
                    source_role=role,
                    origin_dataset_id=str(fact.get("origin_dataset_id", "")),
                    pair_match_basis=pair_match_basis,
                    rule_aggregation_eligible=direct_aggregation,
                )
            )
        return blocks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
