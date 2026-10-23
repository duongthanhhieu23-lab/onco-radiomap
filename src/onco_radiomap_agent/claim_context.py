from __future__ import annotations

import hashlib

import re

from .claim_renderer import FrozenLightRAGRetriever, pack_complete_blocks

from .claim_scope import _eligible, query_pair_candidate_status
from .prediction_rules import _parse_date, population_scope_compatible

def pair_context(case, facts, previous_context=""):
    """Recheck scope after corrections; retain old retrieval order when usable.

    No corpus expansion, vector call, gold labels or old structured conclusions.
    Missing candidate sources follow in stable source order, with family dedup.
    """
    prior = re.findall(r"source_id=([^;\]\s]+)", previous_context)
    ranks = {sid: i for i, sid in enumerate(prior)}
    candidates = []
    cutoff = _parse_date(case["cutoff_date"])
    for fact in facts:
        public_date = _parse_date(fact.get("evidence_public_date"))
        if (
            fact["origin_dataset_id"] != case["dataset_id"]
            or fact["clinical_use"] != case["clinical_use"]
            or fact.get("hard_exclusion")
            or fact.get("retracted")
            or cutoff is None
            or public_date is None
            or public_date > cutoff
            or not population_scope_compatible(
                fact.get("applicability", {}).get("population_scope"), case["population_scope"]
            )
        ):
            continue
        basis = query_pair_candidate_status(
            fact, tumor_subtype_id=case["tumor_subtype_id"], radiopharmaceutical_id=case["radiopharmaceutical_id"]
        )
        if basis is None:
            continue
        if fact.get("atomic_review", {}).get("audited") and fact["atomic_review"]["relation_eligible"] is False:
            continue
        candidates.append(
            (
                fact["fact_id"],
                fact["source_id"],
                fact,
                basis,
                _eligible(fact) and basis != "ambiguous_candidate_pair_for_llm",
            )
        )
    candidates.sort(key=lambda c: (ranks.get(c[1], len(ranks)), c[1]))
    seen = set()
    selected = []
    for item in candidates:
        family = item[2]["study_family_id"]
        if family not in seen:
            selected.append(item)
            seen.add(family)
        if len(selected) == 40:
            break
    blocks = FrozenLightRAGRetriever._render_snapshot_candidates(selected, max_chars=10000)
    context, included, _ = pack_complete_blocks(blocks, max_chars=10000)
    return context, {
        "candidate_sources": len(candidates),
        "included_sources": [b.source_id for b in included],
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
    }
