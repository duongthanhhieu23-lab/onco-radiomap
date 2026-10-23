"""Dataset-centred scoring for the frozen Onco-RadioMap case contract.

Reference labels are read exclusively from ``cases.jsonl`` records. Prediction
records are joined by ``(dataset_id, case_id)`` and are rejected if they carry
any ``reference_*`` field, including inside nested provenance objects.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

RELATION_LABELS: tuple[bool, bool] = (False, True)
ATOMIC_CONCLUSION_LABELS: tuple[str, ...] = (
    "supportive",
    "negative",
    "inconclusive",
)
SNAPSHOT_CONCLUSION_LABELS: tuple[str, ...] = (
    "supportive",
    "negative",
    "inconclusive",
    "conflicting",
)
DIAGNOSTIC_MATURITY_LABELS: tuple[str, ...] = ("D1", "D2", "D3", "D4")
THERAPEUTIC_MATURITY_LABELS: tuple[str, ...] = ("T1", "T2", "T3", "T4")
CASE_TYPES: tuple[str, str] = ("atomic_source_record", "pair_snapshot")
CLINICAL_USES: tuple[str, str] = ("diagnostic", "therapeutic")
METHOD_IDS: tuple[str, ...] = ("pure_llm", "naive_rag", "lightrag", "full_system")
PREDICTION_STATUSES: tuple[str, ...] = ("valid", "abstain", "invalid")
ENTITY_RESOLUTION_STATUSES: tuple[str, ...] = (
    "resolved_both",
    "tumor_only",
    "radiopharmaceutical_only",
    "unresolved",
)

_CASE_COMMON_FIELDS = {
    "case_id",
    "dataset_id",
    "case_type",
    "tumor_subtype_id",
    "radiopharmaceutical_id",
    "clinical_use",
    "cutoff_date",
    "provenance",
}
_PREDICTION_COMMON_FIELDS = {
    "case_id",
    "dataset_id",
    "case_type",
    "clinical_use",
    "method_id",
    "run_id",
    "predicted_tumor_subtype_id",
    "predicted_radiopharmaceutical_id",
    "entity_resolution_status",
    "graph_insertable",
    "prediction_status",
    "prediction_provenance",
}
_MISSING_PREDICTION = "__missing_prediction__"


class EvaluationContractError(ValueError):
    """Raised when reference or prediction records violate the frozen contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file and attach useful line numbers to errors."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise EvaluationContractError(
                    f"{path}:{line_number}: each JSONL line must be an object"
                )
            records.append(record)
    return records


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_keys(nested)


def _require_fields(record: Mapping[str, Any], fields: set[str], *, kind: str) -> None:
    missing = sorted(fields.difference(record))
    if missing:
        raise EvaluationContractError(
            f"{kind} {record.get('case_id', '<unknown>')!r} is missing fields: {missing}"
        )


def _forbid_extra_fields(record: Mapping[str, Any], allowed: set[str], *, kind: str) -> None:
    extras = sorted(set(record).difference(allowed))
    if extras:
        raise EvaluationContractError(
            f"{kind} {record.get('case_id', '<unknown>')!r} contains fields outside the frozen schema: {extras}"
        )


def _validate_optional_id(value: Any, *, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise EvaluationContractError(f"{field} must be a non-empty string or null")


def _validate_case(record: Mapping[str, Any]) -> None:
    _require_fields(record, _CASE_COMMON_FIELDS, kind="reference case")
    forbidden = sorted(
        key
        for key in _nested_keys(record)
        if key.startswith(("predicted_", "prediction_", "full_system_"))
    )
    if forbidden:
        raise EvaluationContractError(
            f"reference case {record['case_id']!r} contains prediction fields: {forbidden}"
        )

    case_type = record["case_type"]
    clinical_use = record["clinical_use"]
    if case_type not in CASE_TYPES:
        raise EvaluationContractError(f"unknown case_type {case_type!r}")
    if clinical_use not in CLINICAL_USES:
        raise EvaluationContractError(f"unknown clinical_use {clinical_use!r}")
    if not isinstance(record["provenance"], Mapping) or not record["provenance"]:
        raise EvaluationContractError(
            f"reference case {record['case_id']!r} requires non-empty provenance"
        )
    _validate_optional_id(record["tumor_subtype_id"], field="tumor_subtype_id")
    _validate_optional_id(record["radiopharmaceutical_id"], field="radiopharmaceutical_id")

    if case_type == "atomic_source_record":
        atomic_fields = {
            "reference_relation_eligible",
            "reference_atomic_conclusion",
            "study_family_id",
        }
        _forbid_extra_fields(record, _CASE_COMMON_FIELDS | atomic_fields, kind="atomic reference case")
        _require_fields(
            record,
            atomic_fields,
            kind="atomic reference case",
        )
        eligible = record["reference_relation_eligible"]
        conclusion = record["reference_atomic_conclusion"]
        if not isinstance(eligible, bool):
            raise EvaluationContractError("reference_relation_eligible must be boolean")
        if eligible and conclusion not in ATOMIC_CONCLUSION_LABELS:
            raise EvaluationContractError(
                "eligible atomic references require a fixed three-class conclusion"
            )
        if not eligible and conclusion is not None:
            raise EvaluationContractError(
                "ineligible atomic references require reference_atomic_conclusion=null"
            )
    else:
        snapshot_fields = {
            "reference_maturity_code",
            "reference_evidence_conclusion",
            "supporting_assertion_ids",
        }
        _forbid_extra_fields(record, _CASE_COMMON_FIELDS | snapshot_fields, kind="snapshot reference case")
        _require_fields(
            record,
            snapshot_fields,
            kind="snapshot reference case",
        )
        if record["tumor_subtype_id"] is None or record["radiopharmaceutical_id"] is None:
            raise EvaluationContractError("pair snapshots require both canonical entity IDs")
        maturity = record["reference_maturity_code"]
        allowed_maturity = (
            DIAGNOSTIC_MATURITY_LABELS
            if clinical_use == "diagnostic"
            else THERAPEUTIC_MATURITY_LABELS
        )
        if maturity not in allowed_maturity:
            raise EvaluationContractError(
                f"{clinical_use} snapshot has invalid maturity code {maturity!r}"
            )
        if record["reference_evidence_conclusion"] not in SNAPSHOT_CONCLUSION_LABELS:
            raise EvaluationContractError("snapshot conclusion is outside the fixed four classes")


def _validate_prediction(record: Mapping[str, Any]) -> None:
    _require_fields(record, _PREDICTION_COMMON_FIELDS, kind="prediction")
    forbidden = sorted(key for key in _nested_keys(record) if key.startswith("reference_"))
    if forbidden:
        raise EvaluationContractError(
            f"prediction {record['case_id']!r} contains reference fields: {forbidden}"
        )

    case_type = record["case_type"]
    clinical_use = record["clinical_use"]
    if case_type not in CASE_TYPES:
        raise EvaluationContractError(f"unknown prediction case_type {case_type!r}")
    if clinical_use not in CLINICAL_USES:
        raise EvaluationContractError(f"unknown prediction clinical_use {clinical_use!r}")
    if record["method_id"] not in METHOD_IDS:
        raise EvaluationContractError(f"unknown method_id {record['method_id']!r}")
    if isinstance(record["run_id"], bool) or not isinstance(record["run_id"], int):
        raise EvaluationContractError("run_id must be an integer")
    if record["run_id"] < 1:
        raise EvaluationContractError("run_id must be >= 1")
    _validate_optional_id(
        record["predicted_tumor_subtype_id"],
        field="predicted_tumor_subtype_id",
    )
    _validate_optional_id(
        record["predicted_radiopharmaceutical_id"],
        field="predicted_radiopharmaceutical_id",
    )
    prediction_status = record["prediction_status"]
    if prediction_status not in PREDICTION_STATUSES:
        raise EvaluationContractError(
            f"prediction_status must be one of {PREDICTION_STATUSES!r}"
        )
    entity_status = record["entity_resolution_status"]
    if entity_status not in ENTITY_RESOLUTION_STATUSES:
        raise EvaluationContractError(
            f"entity_resolution_status must be one of {ENTITY_RESOLUTION_STATUSES!r}"
        )
    if not isinstance(record["graph_insertable"], bool):
        raise EvaluationContractError("graph_insertable must be boolean")

    tumor_resolved = record["predicted_tumor_subtype_id"] is not None
    radio_resolved = record["predicted_radiopharmaceutical_id"] is not None
    expected_entity_status = (
        "resolved_both"
        if tumor_resolved and radio_resolved
        else "tumor_only"
        if tumor_resolved
        else "radiopharmaceutical_only"
        if radio_resolved
        else "unresolved"
    )
    if entity_status != expected_entity_status:
        raise EvaluationContractError(
            f"entity_resolution_status={entity_status!r} is inconsistent with entity IDs; "
            f"expected {expected_entity_status!r}"
        )

    if case_type == "atomic_source_record":
        atomic_fields = {"predicted_relation_eligible", "predicted_atomic_conclusion"}
        _forbid_extra_fields(
            record,
            _PREDICTION_COMMON_FIELDS | atomic_fields | {"prediction_provenance"},
            kind="atomic prediction",
        )
        _require_fields(
            record,
            atomic_fields,
            kind="atomic prediction",
        )
        eligible = record["predicted_relation_eligible"]
        conclusion = record["predicted_atomic_conclusion"]
        if prediction_status == "valid":
            if not isinstance(eligible, bool):
                raise EvaluationContractError(
                    "valid atomic predictions require boolean predicted_relation_eligible"
                )
            if eligible and conclusion not in ATOMIC_CONCLUSION_LABELS:
                raise EvaluationContractError(
                    "eligible atomic predictions require a fixed three-class conclusion"
                )
            if not eligible and conclusion is not None:
                raise EvaluationContractError(
                    "ineligible atomic predictions require predicted_atomic_conclusion=null"
                )
        elif eligible is not None or conclusion is not None:
            raise EvaluationContractError(
                "invalid/abstain atomic predictions require null relation and conclusion"
            )
        expected_graph_insertable = bool(
            prediction_status == "valid" and eligible is True and tumor_resolved and radio_resolved
        )
    else:
        snapshot_fields = {"predicted_maturity_code", "predicted_evidence_conclusion"}
        _forbid_extra_fields(
            record,
            _PREDICTION_COMMON_FIELDS | snapshot_fields | {"prediction_provenance"},
            kind="snapshot prediction",
        )
        _require_fields(
            record,
            snapshot_fields,
            kind="snapshot prediction",
        )
        allowed_maturity = (
            DIAGNOSTIC_MATURITY_LABELS
            if clinical_use == "diagnostic"
            else THERAPEUTIC_MATURITY_LABELS
        )
        maturity = record["predicted_maturity_code"]
        conclusion = record["predicted_evidence_conclusion"]
        if prediction_status == "valid":
            if maturity not in allowed_maturity:
                raise EvaluationContractError(
                    f"{clinical_use} prediction has invalid maturity code {maturity!r}"
                )
            if conclusion not in SNAPSHOT_CONCLUSION_LABELS:
                raise EvaluationContractError(
                    "predicted snapshot conclusion is outside the fixed four classes"
                )
        elif maturity is not None or conclusion is not None:
            raise EvaluationContractError(
                "invalid/abstain snapshot predictions require null maturity and conclusion"
            )
        expected_graph_insertable = bool(
            prediction_status == "valid" and tumor_resolved and radio_resolved
        )

    if record["graph_insertable"] != expected_graph_insertable:
        raise EvaluationContractError(
            "graph_insertable is inconsistent with prediction/entity status"
        )


def _safe_float(value: Any) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _label_for_output(label: Any) -> Any:
    if isinstance(label, bool):
        return label
    return str(label)


def _classification_task(
    *,
    task_id: str,
    clinical_use: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
    weighted_kappa: bool = False,
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise EvaluationContractError(f"{task_id}: truth and prediction lengths differ")

    normalized_pred = [
        _MISSING_PREDICTION if value is None else value for value in y_pred
    ]
    if y_true:
        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_true,
            normalized_pred,
            labels=list(labels),
            zero_division=0,
        )
        macro_f1 = float(
            f1_score(
                y_true,
                normalized_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        )
    else:
        precision = recall = class_f1 = [0.0] * len(labels)
        support = [0] * len(labels)
        macro_f1 = None
    classes = []
    for index, label in enumerate(labels):
        classes.append(
            {
                "label": _label_for_output(label),
                "support": int(support[index]),
                "predicted_support": sum(value == label for value in normalized_pred),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
            }
        )

    metrics: dict[str, float | None] = {
        "macro_f1": macro_f1,
        "exact_accuracy": float(accuracy_score(y_true, normalized_pred)) if y_true else None,
    }
    missing_reference_classes = [
        _label_for_output(row["label"]) for row in classes if row["support"] == 0
    ]
    if task_id == "relation_eligibility":
        negative = classes[0]
        positive = classes[1]
        metrics.update(
            {
                "eligible_precision": positive["precision"] if y_true else None,
                "eligible_recall": positive["recall"] if y_true else None,
                "eligible_sensitivity": positive["recall"] if y_true else None,
                "eligible_specificity": negative["recall"] if y_true else None,
                "eligible_f1": positive["f1"] if y_true else None,
            }
        )
    if weighted_kappa:
        if len(y_true) < 2 or len(set(y_true).union(normalized_pred)) < 2:
            kappa = None
        else:
            kappa = _safe_float(
                cohen_kappa_score(
                    y_true,
                    normalized_pred,
                    labels=list(labels),
                    weights="quadratic",
                )
            )
        metrics["weighted_kappa"] = kappa

    return {
        "task_id": task_id,
        "clinical_use": clinical_use,
        "fixed_labels": [_label_for_output(label) for label in labels],
        "n_cases": len(y_true),
        "metrics": metrics,
        "classes": classes,
        "evaluation_status": (
            "not_evaluable_no_valid_predictions"
            if not y_true
            else "not_evaluable_due_to_class_support"
            if missing_reference_classes
            else "evaluable"
        ),
        "missing_reference_classes": missing_reference_classes,
        "zero_division": 0,
        "kappa_weights": "quadratic" if weighted_kappa else None,
    }


def _entity_task(
    *,
    task_id: str,
    clinical_use: str,
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    reference_field: str,
    prediction_field: str,
) -> dict[str, Any]:
    eligible_pairs = [
        (case[reference_field], prediction[prediction_field])
        for case, prediction in zip(cases, predictions, strict=True)
        if case[reference_field] is not None
    ]
    correct_n = sum(reference == predicted for reference, predicted in eligible_pairs)
    eligible_n = len(eligible_pairs)
    return {
        "task_id": task_id,
        "clinical_use": clinical_use,
        "fixed_labels": None,
        "n_cases": len(cases),
        "eligible_n": eligible_n,
        "excluded_reference_id_missing_n": len(cases) - eligible_n,
        "correct_n": correct_n,
        "metrics": {
            "exact_accuracy": correct_n / eligible_n if eligible_n else None,
        },
        "classes": [],
        "evaluation_status": (
            "not_evaluable_no_valid_predictions"
            if not cases
            else "not_evaluable_no_reference_ids"
            if not eligible_n
            else "evaluable"
        ),
    }


def _filter_use(
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    clinical_use: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if clinical_use == "all":
        return list(paired)
    return [item for item in paired if item[0]["clinical_use"] == clinical_use]


def _score_atomic(
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for clinical_use in ("all", *CLINICAL_USES):
        subset = _filter_use(paired, clinical_use)
        cases = [case for case, _ in subset]
        predictions = [prediction for _, prediction in subset]
        tasks.extend(
            [
                _entity_task(
                    task_id="tumor_subtype_normalization",
                    clinical_use=clinical_use,
                    cases=cases,
                    predictions=predictions,
                    reference_field="tumor_subtype_id",
                    prediction_field="predicted_tumor_subtype_id",
                ),
                _entity_task(
                    task_id="radiopharmaceutical_normalization",
                    clinical_use=clinical_use,
                    cases=cases,
                    predictions=predictions,
                    reference_field="radiopharmaceutical_id",
                    prediction_field="predicted_radiopharmaceutical_id",
                ),
                _classification_task(
                    task_id="relation_eligibility",
                    clinical_use=clinical_use,
                    y_true=[case["reference_relation_eligible"] for case in cases],
                    y_pred=[prediction["predicted_relation_eligible"] for prediction in predictions],
                    labels=RELATION_LABELS,
                ),
            ]
        )
        eligible = [
            (case, prediction)
            for case, prediction in subset
            if case["reference_relation_eligible"]
        ]
        tasks.append(
            _classification_task(
                task_id="atomic_conclusion",
                clinical_use=clinical_use,
                y_true=[case["reference_atomic_conclusion"] for case, _ in eligible],
                y_pred=[prediction["predicted_atomic_conclusion"] for _, prediction in eligible],
                labels=ATOMIC_CONCLUSION_LABELS,
            )
        )
    return tasks


def _score_snapshots(
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for clinical_use in ("all", *CLINICAL_USES):
        subset = _filter_use(paired, clinical_use)
        cases = [case for case, _ in subset]
        predictions = [prediction for _, prediction in subset]
        tasks.append(
            _classification_task(
                task_id="snapshot_conclusion",
                clinical_use=clinical_use,
                y_true=[case["reference_evidence_conclusion"] for case in cases],
                y_pred=[
                    prediction["predicted_evidence_conclusion"]
                    for prediction in predictions
                ],
                labels=SNAPSHOT_CONCLUSION_LABELS,
            )
        )

    diagnostic = _filter_use(paired, "diagnostic")
    tasks.append(
        _classification_task(
            task_id="diagnostic_maturity",
            clinical_use="diagnostic",
            y_true=[case["reference_maturity_code"] for case, _ in diagnostic],
            y_pred=[prediction["predicted_maturity_code"] for _, prediction in diagnostic],
            labels=DIAGNOSTIC_MATURITY_LABELS,
            weighted_kappa=True,
        )
    )
    therapeutic = _filter_use(paired, "therapeutic")
    tasks.append(
        _classification_task(
            task_id="therapeutic_maturity",
            clinical_use="therapeutic",
            y_true=[case["reference_maturity_code"] for case, _ in therapeutic],
            y_pred=[prediction["predicted_maturity_code"] for _, prediction in therapeutic],
            labels=THERAPEUTIC_MATURITY_LABELS,
            weighted_kappa=True,
        )
    )
    return tasks


def _metric_points(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for group in groups:
        base = {
            "dataset_id": group["dataset_id"],
            "dataset_role": group["dataset_role"],
            "method_id": group["method_id"],
            "run_id": group["run_id"],
            "case_type": group["case_type"],
        }
        for task in group["tasks"]:
            task_base = {
                **base,
                "task_id": task["task_id"],
                "clinical_use": task["clinical_use"],
                "n_cases": task["n_cases"],
                "eligible_n": task.get("eligible_n"),
                "excluded_n": task.get("excluded_reference_id_missing_n"),
            }
            for metric, value in task["metrics"].items():
                if value is not None:
                    points.append(
                        {
                            **task_base,
                            "metric": metric,
                            "class_label": None,
                            "value": value,
                        }
                    )
            for class_row in task["classes"]:
                for metric in ("precision", "recall", "f1"):
                    points.append(
                        {
                            **task_base,
                            "metric": metric,
                            "class_label": class_row["label"],
                            "value": class_row[metric],
                        }
                    )
    return points


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    key_fields = (
        "dataset_id",
        "dataset_role",
        "method_id",
        "case_type",
        "task_id",
        "clinical_use",
        "metric",
        "class_label",
    )
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for point in points:
        buckets[tuple(point[field] for field in key_fields)].append(float(point["value"]))

    summaries = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summaries.append(
            {
                **dict(zip(key_fields, key, strict=True)),
                "mean": statistics.fmean(values),
                "std": _sample_std(values),
                "n_runs": len(values),
            }
        )
    return summaries


def _compute_gaps(points: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internals = sorted(
        {point["dataset_id"] for point in points if point["dataset_role"] == "internal_validation"}
    )
    externals = sorted(
        {point["dataset_id"] for point in points if point["dataset_role"] == "external_validation"}
    )
    metric_key_fields = (
        "method_id",
        "run_id",
        "case_type",
        "task_id",
        "clinical_use",
        "metric",
        "class_label",
    )
    by_dataset: dict[str, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for point in points:
        key = tuple(point[field] for field in metric_key_fields)
        by_dataset[point["dataset_id"]][key] = float(point["value"])

    gaps: list[dict[str, Any]] = []
    for internal_id in internals:
        for external_id in externals:
            shared = sorted(
                set(by_dataset[internal_id]).intersection(by_dataset[external_id]),
                key=lambda key: tuple(str(value) for value in key),
            )
            for key in shared:
                internal_value = by_dataset[internal_id][key]
                external_value = by_dataset[external_id][key]
                gaps.append(
                    {
                        "internal_dataset_id": internal_id,
                        "external_dataset_id": external_id,
                        **dict(zip(metric_key_fields, key, strict=True)),
                        "internal_value": internal_value,
                        "external_value": external_value,
                        "gap": internal_value - external_value,
                        "gap_definition": "internal_validation_minus_external_validation",
                    }
                )

    summary_fields = (
        "internal_dataset_id",
        "external_dataset_id",
        "method_id",
        "case_type",
        "task_id",
        "clinical_use",
        "metric",
        "class_label",
    )
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for gap in gaps:
        buckets[tuple(gap[field] for field in summary_fields)].append(float(gap["gap"]))
    summaries = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summaries.append(
            {
                **dict(zip(summary_fields, key, strict=True)),
                "mean_gap": statistics.fmean(values),
                "std_gap": _sample_std(values),
                "n_runs": len(values),
                "gap_definition": "internal_validation_minus_external_validation",
            }
        )
    return gaps, summaries


def evaluate_predictions(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    dataset_roles: Mapping[str, str] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Evaluate predictions without ever treating prediction fields as references.

    Each ``dataset_id × method_id × run_id × case_type`` is scored separately.
    ``require_complete=True`` rejects partial method/run coverage rather than
    silently evaluating an easier subset.
    """

    roles = dict(dataset_roles or {})
    case_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    cases_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        _validate_case(case)
        key = (str(case["dataset_id"]), str(case["case_id"]))
        if key in case_index:
            raise EvaluationContractError(f"duplicate reference case key: {key}")
        case_index[key] = case
        cases_by_dataset[key[0]].append(case)

    prediction_groups: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for prediction in predictions:
        _validate_prediction(prediction)
        case_key = (str(prediction["dataset_id"]), str(prediction["case_id"]))
        case = case_index.get(case_key)
        if case is None:
            raise EvaluationContractError(f"prediction references unknown case: {case_key}")
        for field in ("dataset_id", "case_type", "clinical_use"):
            if prediction[field] != case[field]:
                raise EvaluationContractError(
                    f"prediction {case_key} has {field}={prediction[field]!r}; "
                    f"reference has {case[field]!r}"
                )
        group_key = (
            str(prediction["dataset_id"]),
            str(prediction["method_id"]),
            int(prediction["run_id"]),
        )
        if prediction["case_id"] in prediction_groups[group_key]:
            raise EvaluationContractError(
                f"duplicate prediction for dataset/method/run/case: {(*group_key, prediction['case_id'])}"
            )
        prediction_groups[group_key][str(prediction["case_id"])] = prediction

    if not case_index:
        raise EvaluationContractError("no reference cases were loaded")
    if not prediction_groups:
        raise EvaluationContractError("no predictions were loaded")

    groups: list[dict[str, Any]] = []
    for (dataset_id, method_id, run_id), group_predictions in sorted(prediction_groups.items()):
        reference_cases = cases_by_dataset[dataset_id]
        reference_ids = {str(case["case_id"]) for case in reference_cases}
        missing_ids = sorted(reference_ids.difference(group_predictions))
        if require_complete and missing_ids:
            preview = missing_ids[:5]
            raise EvaluationContractError(
                f"incomplete predictions for {(dataset_id, method_id, run_id)}: "
                f"missing {len(missing_ids)} cases; first={preview}"
            )

        for case_type in CASE_TYPES:
            selected_cases = [
                case
                for case in reference_cases
                if case["case_type"] == case_type and str(case["case_id"]) in group_predictions
            ]
            paired = [
                (case, group_predictions[str(case["case_id"])]) for case in selected_cases
            ]
            valid_paired = [
                (case, prediction)
                for case, prediction in paired
                if prediction["prediction_status"] == "valid"
            ]
            reference_case_type_count = sum(
                case["case_type"] == case_type for case in reference_cases
            )
            tasks = (
                _score_atomic(valid_paired)
                if case_type == "atomic_source_record"
                else _score_snapshots(valid_paired)
            )
            status_counts = {
                status: sum(
                    prediction["prediction_status"] == status for _, prediction in paired
                )
                for status in PREDICTION_STATUSES
            }
            groups.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": roles.get(dataset_id, "unknown"),
                    "method_id": method_id,
                    "run_id": run_id,
                    "case_type": case_type,
                    "reference_case_count": reference_case_type_count,
                    "predicted_case_count": len(paired),
                    "missing_prediction_count": reference_case_type_count - len(paired),
                    "valid_prediction_count": status_counts["valid"],
                    "abstain_prediction_count": status_counts["abstain"],
                    "invalid_prediction_count": status_counts["invalid"],
                    "formal_scoring_coverage": (
                        status_counts["valid"] / reference_case_type_count
                        if reference_case_type_count
                        else None
                    ),
                    "tasks": tasks,
                }
            )

    points = _metric_points(groups)
    run_summaries = _summarize_points(points)
    gaps, gap_summaries = _compute_gaps(points)
    return {
        "schema_version": "onco-radiomap-evaluation-v1.1",
        "metadata": {
            "reference_source": "cases_jsonl_only",
            "prediction_reference_fields_allowed": False,
            "dataset_pooling_allowed": False,
            "fixed_class_lists": {
                "relation_eligibility": list(RELATION_LABELS),
                "atomic_conclusion": list(ATOMIC_CONCLUSION_LABELS),
                "snapshot_conclusion": list(SNAPSHOT_CONCLUSION_LABELS),
                "diagnostic_maturity": list(DIAGNOSTIC_MATURITY_LABELS),
                "therapeutic_maturity": list(THERAPEUTIC_MATURITY_LABELS),
            },
            "macro_f1_zero_division": 0,
            "weighted_kappa_weights": "quadratic",
            "gap_definition": "internal_validation_minus_external_validation",
            "run_summary_std": "sample_sd_ddof_1; zero when n_runs=1",
            "require_complete_predictions": require_complete,
            "formal_scoring_policy": (
                "prediction_status=valid only; abstain/invalid reported as coverage"
            ),
            "pair_snapshot_entity_scoring": (
                "not_applicable_pair_ids_are_query_inputs"
            ),
        },
        "dataset_ids": sorted(cases_by_dataset),
        "groups": groups,
        "run_summaries": run_summaries,
        "gaps": gaps,
        "gap_summaries": gap_summaries,
    }


_CSV_FIELDS = (
    "record_type",
    "dataset_id",
    "dataset_role",
    "internal_dataset_id",
    "external_dataset_id",
    "method_id",
    "run_id",
    "case_type",
    "clinical_use",
    "task_id",
    "metric",
    "class_label",
    "value",
    "mean",
    "std",
    "mean_gap",
    "std_gap",
    "internal_value",
    "external_value",
    "n_runs",
    "n_cases",
    "eligible_n",
    "excluded_n",
    "support",
    "predicted_support",
    "gap_definition",
)


def _csv_label(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    return value


def report_to_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the structured report without pooling datasets."""

    rows: list[dict[str, Any]] = []
    for group in report["groups"]:
        base = {
            "dataset_id": group["dataset_id"],
            "dataset_role": group["dataset_role"],
            "method_id": group["method_id"],
            "run_id": group["run_id"],
            "case_type": group["case_type"],
        }
        rows.append(
            {
                **base,
                "record_type": "coverage",
                "metric": "predicted_case_count",
                "value": group["predicted_case_count"],
                "n_cases": group["reference_case_count"],
            }
        )
        rows.append(
            {
                **base,
                "record_type": "coverage",
                "metric": "missing_prediction_count",
                "value": group["missing_prediction_count"],
                "n_cases": group["reference_case_count"],
            }
        )
        for metric in (
            "valid_prediction_count",
            "abstain_prediction_count",
            "invalid_prediction_count",
            "formal_scoring_coverage",
        ):
            rows.append(
                {
                    **base,
                    "record_type": "coverage",
                    "metric": metric,
                    "value": group[metric],
                    "n_cases": group["reference_case_count"],
                }
            )
        for task in group["tasks"]:
            task_base = {
                **base,
                "clinical_use": task["clinical_use"],
                "task_id": task["task_id"],
                "n_cases": task["n_cases"],
                "eligible_n": task.get("eligible_n"),
                "excluded_n": task.get("excluded_reference_id_missing_n"),
            }
            for metric, value in task["metrics"].items():
                rows.append(
                    {
                        **task_base,
                        "record_type": "metric",
                        "metric": metric,
                        "value": value,
                    }
                )
            for class_row in task["classes"]:
                for metric in ("precision", "recall", "f1"):
                    rows.append(
                        {
                            **task_base,
                            "record_type": "class_metric",
                            "metric": metric,
                            "class_label": _csv_label(class_row["label"]),
                            "value": class_row[metric],
                            "support": class_row["support"],
                            "predicted_support": class_row["predicted_support"],
                        }
                    )
    for summary in report["run_summaries"]:
        rows.append({**summary, "record_type": "run_summary"})
    for gap in report["gaps"]:
        rows.append({**gap, "record_type": "gap", "value": gap["gap"]})
    for summary in report["gap_summaries"]:
        rows.append({**summary, "record_type": "gap_summary"})
    return rows


def write_evaluation_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write strict JSON plus a long-format CSV for downstream table builders."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for raw_row in report_to_csv_rows(report):
            row = {field: _csv_label(raw_row.get(field)) for field in _CSV_FIELDS}
            writer.writerow(row)
