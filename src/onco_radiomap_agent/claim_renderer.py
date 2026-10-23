from __future__ import annotations

from collections.abc import Iterable, Mapping

from dataclasses import dataclass

from typing import Any

DEFAULT_MAX_CHARS = 10000

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
    @staticmethod
    def _render_snapshot_candidates(
        selected: list[tuple[str, str, dict[str, Any], str, bool]],
        *,
        max_chars: int,
    ) -> list[EvidenceBlock]:
        """Render selected pair candidates while keeping every block whole."""

        # Do not fill the entire budget with headers and call it retrieved
        # evidence. Reserve source prose before emitting a candidate block.
        selected = selected[:max(1, max_chars // 900)]
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
            body = balanced_excerpt(
                str(fact.get('raw_source_text') or fact.get("evidence_excerpt", "")),
                max_chars=body_budget_each,
            )
            if not body:
                continue
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

from .evidence_attribution import balanced_excerpt
