# Onco-RadioMap

**V1.3.2: human evidence assessment for tumor subtype–radiopharmaceutical pairs.**

This repository contains the code for the archived V1.3.2 experiments: 3 datasets, 4 methods, and 5 repeated runs, totaling 35,240 predictions. It includes rules, prompts, retrieval, evidence attribution checks, evidence maturity assessment, and evaluation. Data, retrieval contexts, and saved model responses are supplied through explicit file paths.

## Installation and offline checks

Use Python 3.11 or later. Run the following commands from the repository root:

```bash
python -m pip install -e ".[test]"
python -m onco_radiomap_agent --version
python -m pytest -q
python examples/offline_demo.py
```

Install `.[live]` for model API calls, or `.[retrieval]` to build or query LightRAG indices. Dependency versions match the environment used to verify this package. Run Python with assertions enabled; do not use `-O`.

## Repository contents

| Location | Contents |
| --- | --- |
| `src/onco_radiomap_agent/engine.py` | Unified entry point for inference with all four methods, response normalization, and final processing |
| `preparation.py`, `task_contract.py` | Frozen classification requests, rule proposals, and prompts |
| `arbiter.py` | Permitted decision overrides and source checks for the Full System |
| `prediction_facts.py`, `prediction_rules.py` | Fact extraction and deterministic rules without access to reference answers |
| `naive_rag.py`, `lightrag_*.py` | Retrieval and LightRAG interfaces for entity pairs |
| `claim_scope.py`, `claim_context.py`, `claim_renderer.py` | Source eligibility, pair attribution, and evidence contexts |
| `claims.py`, `conclusion_alignment.py` | Source-constrained verification and aggregation of final evidence conclusions |
| `ownership*.py`, `maturity*.py` | Final evidence attribution, study design, and evidence maturity processing |
| `evaluation.py`, `expert_metrics.py`, `repeated_metrics.py` | Scoring against reference labels, expert agreement, and case-level bootstrap across five repeated runs |
| `resources/` | Required entity registries, supplemental open-world entities, prediction schema, and graph construction prompts |
| `tests/` | Offline regression tests using synthetic inputs |
| `code_provenance.json` | Source provenance, source file SHA-256 hashes, and retained code scope |

Reproducing the full results requires the corresponding frozen data, retrieval contexts, and saved model responses. These artifacts are stored separately from the code package.

`prediction_rules` generates initial rule proposals, which `arbiter` uses to validate permitted overrides. `conclusion_alignment.aggregate_claims` aggregates the final conclusions for valid pairs: a `conflicting` label requires opposing results from independent studies with comparable endpoints, regimens, and clinical settings. `claim_scope` filters sources by eligibility and pair attribution. Prompt version identifiers retain their archived values for request identity verification.

## Running your own inputs

Copy and populate `examples/bundle.json`. This file is a path template; supply the files it references under `inputs/`. Each invocation processes one dataset. The case pool should contain the complete set of atomic evidence records for that dataset.

| Field | File contents |
| --- | --- |
| `cases` | Prediction inputs in JSONL format, without answer fields: `atomic_source_record` or `pair_snapshot` |
| `rule_facts` | Facts for classification rules in JSONL format, including `origin_dataset_id` and `origin_case_id` |
| `naive_contexts`, `lightrag_contexts` | JSONL with `case_id` and `context` on each line |
| `claim_facts` | Facts for final conclusion review in JSONL format, also retaining `raw_source_text` |
| `claim_contexts` | Contexts with ranked sources for the final conclusion stage, in the same format as above |

`baseline_data.load_prediction_inputs`, `prediction_facts.build_prediction_facts`, `BM25EvidenceIndex`, and `FrozenLightRAGRetriever` provide input preparation and retrieval interfaces. Retrieval inputs must respect separation by dataset, date, clinical use, population, and study family.

Prepare classification requests without making API calls:

```bash
python -m onco_radiomap_agent prepare --bundle examples/bundle.json --output outputs/requests.json
```

Run offline using saved model responses:

```bash
python -m onco_radiomap_agent predict --bundle examples/bundle.json --receipts receipts/run1 --run-id 1 --output outputs/predictions.jsonl
```

Each saved response file is named `call_id.json` and must contain the matching `identity` and `run_id`, plus `finish_reason: "stop"`, `response`, and `provider_model`. Legacy single-run response files may omit `run_id`; these are accepted only for run 1. Each `call_id` incorporates the run number and request identity hash. Missing responses, responses from a different run, or mismatched responses stop execution. For the Full System, conclusion review requests are generated after classification for the pairs determined to be valid.

Explicitly enable new model API calls:

```bash
python -m pip install -e ".[live]"
# Set DEEPSEEK_API_KEY in the current terminal; see .env.example.
python -m onco_radiomap_agent predict --bundle examples/bundle.json --live --max-calls 20 --save-receipts receipts/new-run --output outputs/new-predictions.jsonl
```

`--max-calls` limits the number of API calls, not their cost. Each request is attempted once, and responses are saved to a new directory. The historical model alias is `deepseek-flash`. The current API under that alias may differ from the historical model, so new calls do not provide exact reproduction of the archived results.

Evaluation loads reference answers separately:

```bash
python -m onco_radiomap_agent evaluate --cases data/reference_cases.jsonl --predictions outputs/predictions.jsonl --roles data/dataset_roles.json --output outputs/evaluation.json
```

The inference entry point rejects answer fields prefixed with `reference_*`, `gold_*`, `expert_*`, or `final_*`, with an exception for the study design field `gold_standard_comparison`. The relation task is evaluated on atomic records, and maturity is evaluated on pair snapshots. Valid predictions, abstentions, and invalid outputs are reported separately.

## Verification

`release_verification.json` records the results and scope of offline tests, frozen response replay, and local packaging checks. See `code_provenance.json` for source provenance and packaging modifications.

All repository text files use LF line endings. After extracting the archive, run `sha256sum -c SHA256SUMS` from the repository root to verify file integrity.

The code is licensed under Apache-2.0. External literature and datasets remain subject to their respective licenses.
