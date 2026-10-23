"""Build prediction-only case inputs without exposing frozen reference answers."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .entity_candidates import SharedCandidateResolver

DATASET_IDS = (
    "dev-human-evidence-pre2023",
    "internal-source-heldout-pre2023",
    "external-temporal-2023-20260823",
)
CHALLENGE_DATASET_IDS = ("internal-challenge-pre2023-v1.1",)
ALL_DATASET_IDS = (*DATASET_IDS, *CHALLENGE_DATASET_IDS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _registry_names(path: Path, id_field: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row[id_field]): str(row["preferred_name"])
            for row in csv.DictReader(handle)
        }


def _nested_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_keys(nested)


def _dataset_cases_path(
    root: Path,
    dataset_id: str,
    *,
    datasets_root: Path | None = None,
) -> Path:
    if datasets_root is not None:
        path = datasets_root / dataset_id / "cases.jsonl"
        if path.is_file():
            return path
        raise FileNotFoundError(f"missing cases.jsonl for {dataset_id!r}: {path}")
    candidates = (
        root / "data" / "v1.1" / "datasets" / dataset_id / "cases.jsonl",
        root / "data" / "datasets" / dataset_id / "cases.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing cases.jsonl for {dataset_id!r}; checked {[str(path) for path in candidates]}"
    )


@lru_cache(maxsize=128)
def _pubmed_mesh_by_source(xml_path: str) -> dict[str, tuple[str, ...]]:
    """Read source-level MeSH descriptors from the locally frozen PubMed XML."""

    path = Path(xml_path)
    if not path.is_file():
        return {}
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return {}
    output: dict[str, tuple[str, ...]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = str(article.findtext("./MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue
        descriptors = tuple(
            sorted(
                {
                    str(node.text).strip()
                    for node in article.findall(
                        "./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"
                    )
                    if node.text and str(node.text).strip()
                }
            )
        )
        output[f"PMID:{pmid}"] = descriptors
    return output


def _source_index_metadata(root: Path, detail: dict[str, Any]) -> dict[str, Any]:
    source_id = str(detail.get("source_id", ""))
    raw_path = str(
        detail.get("raw_provenance", {})
        .get("raw_response", {})
        .get("path", "")
    )
    descriptors: tuple[str, ...] = ()
    if source_id.startswith("PMID:") and raw_path:
        descriptors = _pubmed_mesh_by_source(str((root / raw_path).resolve())).get(
            source_id,
            (),
        )
    age_terms = {
        "Infant",
        "Child",
        "Adolescent",
        "Young Adult",
        "Adult",
        "Middle Aged",
        "Aged",
        "Aged, 80 and over",
    }
    return {
        "population_index_terms": sorted(set(descriptors).intersection(age_terms)),
        "human_indexed": True if "Humans" in descriptors else None,
        "publication_types": sorted(
            {
                str(item.get("name", "")).strip()
                for item in detail.get("publication_types", [])
                if str(item.get("name", "")).strip()
            }
        ),
    }


def load_prediction_inputs(
    root: Path,
    *,
    dataset_ids: Sequence[str] | None = None,
    datasets_root: Path | None = None,
    source_details_path: Path | None = None,
    tumor_registry_path: Path | None = None,
    radiopharmaceutical_registry_path: Path | None = None,
    provisional_radiopharmaceutical_path: Path | None = None,
    alias_exclusions_path: Path | None = None,
    auxiliary_radiopharmaceutical_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a strict whitelist view of every case.

    Atomic inputs get raw title/abstract text from ``source_fact_details`` and
    do not receive canonical IDs from ``cases.jsonl``. Pair snapshots are, by
    design, queries for an already standardized pair and therefore retain the
    two input IDs plus the population stratum and cutoff. The original three
    frozen datasets remain the default; callers opt into supplementary sets
    with ``dataset_ids=ALL_DATASET_IDS`` or an explicit subset.
    """
    selected_dataset_ids = DATASET_IDS if dataset_ids is None else tuple(dataset_ids)
    if len(selected_dataset_ids) != len(set(selected_dataset_ids)):
        raise ValueError("prediction-input dataset IDs must be unique")
    unknown = sorted(set(selected_dataset_ids).difference(ALL_DATASET_IDS))
    if unknown:
        raise ValueError(f"unknown prediction-input dataset IDs: {unknown}")
    source_details_path = source_details_path or (
        root / "evidence" / "executor_phase2" / "source_fact_details.jsonl"
    )
    tumor_registry_path = tumor_registry_path or (
        root / "data" / "seed" / "entities_a.csv"
    )
    radiopharmaceutical_registry_path = radiopharmaceutical_registry_path or (
        root / "data" / "seed" / "entities_b.csv"
    )
    details = {
        str(row["fact_id"]): row
        for row in read_jsonl(source_details_path)
    }
    tumor_names = _registry_names(tumor_registry_path, "tumor_subtype_id")
    radio_names = _registry_names(
        radiopharmaceutical_registry_path, "radiopharmaceutical_id"
    )
    provisional_path = provisional_radiopharmaceutical_path or (
        root
        / "evidence"
        / "executor_phase2"
        / "provisional_external_radiopharmaceuticals.json"
    )
    provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
    radio_names.update(
        {
            str(record["radiopharmaceutical_id"]): str(record["preferred_name"])
            for record in provisional.get("records", [])
        }
    )
    auxiliary_path = auxiliary_radiopharmaceutical_path or (
        root
        / "data"
        / "v1.1"
        / "registries"
        / "radiopharmaceutical_auxiliary.json"
    )
    if auxiliary_path.is_file():
        auxiliary = json.loads(auxiliary_path.read_text(encoding="utf-8"))
        radio_names.update(
            {
                str(record["radiopharmaceutical_id"]): str(
                    record["preferred_name"]
                )
                for record in auxiliary.get("records", [])
            }
        )
    candidate_resolver = SharedCandidateResolver.from_project(
        root,
        tumor_registry_path=tumor_registry_path,
        radiopharmaceutical_registry_path=radiopharmaceutical_registry_path,
        provisional_radiopharmaceutical_path=provisional_path,
        alias_exclusions_path=alias_exclusions_path,
        auxiliary_radiopharmaceutical_path=auxiliary_path,
    )
    inputs: list[dict[str, Any]] = []
    for dataset_id in selected_dataset_ids:
        path = _dataset_cases_path(root, dataset_id, datasets_root=datasets_root)
        for case in read_jsonl(path):
            common = {
                "case_id": str(case["case_id"]),
                "dataset_id": dataset_id,
                "case_type": str(case["case_type"]),
                "clinical_use": str(case["clinical_use"]),
                "cutoff_date": str(case["cutoff_date"]),
            }
            provenance = case["provenance"]
            if case["case_type"] == "atomic_source_record":
                fact_id = str(provenance["source_fact_id"])
                detail = details.get(fact_id)
                if detail is None:
                    raise RuntimeError(f"missing source fact detail for {fact_id}")
                source_text = (
                    f"Title: {detail.get('title', '')}\n"
                    f"Abstract: {detail.get('abstract', '')}\n"
                    f"DOI: {detail.get('doi') or 'not reported'}"
                )
                resolved_candidates = candidate_resolver.resolve_source_text(
                    source_text
                )
                if dataset_id != "external-temporal-2023-20260823":
                    resolved_candidates["candidate_radiopharmaceuticals"] = [
                        candidate
                        for candidate in resolved_candidates[
                            "candidate_radiopharmaceuticals"
                        ]
                        if not str(
                            candidate.get("radiopharmaceutical_id", "")
                        ).startswith("RPNEW:")
                    ]
                if dataset_id not in CHALLENGE_DATASET_IDS:
                    resolved_candidates["candidate_radiopharmaceuticals"] = [
                        candidate
                        for candidate in resolved_candidates[
                            "candidate_radiopharmaceuticals"
                        ]
                        if not str(
                            candidate.get("radiopharmaceutical_id", "")
                        ).startswith("RPV11:")
                    ]
                row = {
                    **common,
                    "source_fact_id": fact_id,
                    "source_id": str(detail["source_id"]),
                    # Publication date and source type are source metadata, not
                    # reference answers.  They are exposed so prediction-side
                    # evidence pools can enforce cutoff isolation without
                    # reading the structured SourceFact used to create labels.
                    "evidence_public_date": str(
                        provenance.get("evidence_public_date") or ""
                    ),
                    "source_type": str(
                        provenance.get("source_type") or "not_reported"
                    ),
                    "record_locator": str(
                        detail.get("result_locator")
                        or detail.get("raw_provenance", {}).get("source_record_id")
                        or "not_reported"
                    ),
                    "source_text": source_text,
                    **_source_index_metadata(root, detail),
                    "retrieval_query": (
                        f"{detail.get('title', '')} {detail.get('abstract', '')[:800]}"
                    ).strip(),
                    **resolved_candidates,
                }
            else:
                tumor_id = str(case["tumor_subtype_id"])
                radio_id = str(case["radiopharmaceutical_id"])
                population_scope = str(provenance.get("population_scope", "not_reported"))
                row = {
                    **common,
                    "tumor_subtype_id": tumor_id,
                    "tumor_subtype_name": tumor_names.get(tumor_id, "unresolved name"),
                    "radiopharmaceutical_id": radio_id,
                    "radiopharmaceutical_name": radio_names.get(radio_id, "unresolved name"),
                    "population_scope": population_scope,
                    # Pair IDs are declared query inputs in the frozen case
                    # contract, not hidden labels. Candidate lists remain
                    # present for interface parity but are not synthesized
                    # from reference answers.
                    "candidate_tumors": [],
                    "candidate_radiopharmaceuticals": [],
                    "retrieval_query": (
                        f"{tumor_id} {tumor_names.get(tumor_id, '')} {radio_id} "
                        f"{radio_names.get(radio_id, '')} {case['clinical_use']} human evidence "
                        f"population {population_scope} before {case['cutoff_date']}"
                    ).strip(),
                }
            denied = [
                key
                for key in _nested_keys(row)
                if key.startswith(("reference_", "prediction_", "predicted_", "full_system_"))
            ]
            if denied:
                raise RuntimeError(f"prediction input leaked answer fields: {denied}")
            inputs.append(row)
    return sorted(inputs, key=lambda row: (row["dataset_id"], row["case_id"]))
