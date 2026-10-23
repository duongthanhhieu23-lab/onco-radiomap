"""CSV adapters and integrity helpers for the two frozen entity registries."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .normalize import normalized_alias_key
from .schemas import Radiopharmaceutical, TumorSubtype

ENTITY_A_HEADERS = [
    "tumor_subtype_id",
    "preferred_name",
    "aliases",
    "canonical_system",
    "canonical_code",
    "parent_tumor_subtype_id",
    "external_mappings",
    "evidence_sources",
]

ENTITY_B_HEADERS = [
    "radiopharmaceutical_id",
    "preferred_name",
    "radionuclide",
    "ligand_or_carrier",
    "chemical_construct",
    "aliases",
    "external_mappings",
    "evidence_sources",
]

JSON_LIST_FIELDS = {"aliases", "evidence_sources"}
JSON_DICT_FIELDS = {"external_mappings"}


def deterministic_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_registry(path: Path, headers: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for raw_row in rows:
            row = dict(raw_row)
            for field in JSON_LIST_FIELDS | JSON_DICT_FIELDS:
                if field in row:
                    row[field] = deterministic_json(row[field])
            writer.writerow(row)
    os.replace(temporary, path)


def decode_registry_row(row: Mapping[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = dict(row)
    for field in JSON_LIST_FIELDS:
        decoded[field] = json.loads(decoded.get(field) or "[]")
    for field in JSON_DICT_FIELDS:
        decoded[field] = json.loads(decoded.get(field) or "{}")
    for field in ("parent_tumor_subtype_id", "chemical_construct"):
        if field in decoded and not decoded[field]:
            decoded[field] = None
    return decoded


def load_registry(path: Path, headers: list[str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers:
            raise ValueError(f"{path}: unexpected header {reader.fieldnames!r}; expected {headers!r}")
        return [decode_registry_row(row) for row in reader]


def validate_tumor_rows(rows: Iterable[Mapping[str, Any]]) -> list[TumorSubtype]:
    return [TumorSubtype.model_validate(dict(row)) for row in rows]


def validate_radiopharmaceutical_rows(rows: Iterable[Mapping[str, Any]]) -> list[Radiopharmaceutical]:
    return [Radiopharmaceutical.model_validate(dict(row)) for row in rows]


def ambiguous_aliases(
    entities: Iterable[tuple[str, str, Iterable[str]]],
) -> dict[str, list[str]]:
    """Return exact normalized aliases that resolve to more than one entity ID."""

    owners: dict[str, set[str]] = {}
    for entity_id, preferred_name, aliases in entities:
        for value in (preferred_name, *aliases):
            key = normalized_alias_key(value)
            if key:
                owners.setdefault(key, set()).add(entity_id)
    return {key: sorted(ids) for key, ids in owners.items() if len(ids) > 1}
