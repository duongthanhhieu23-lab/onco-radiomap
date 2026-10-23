"""Shared, answer-free entity candidate generation for all baselines.

Candidates are produced only by exact alias matches between original source
text and the frozen registries.  Reference IDs from ``cases.jsonl`` are never
consulted for atomic records.  Ambiguous aliases are ignored rather than
forced to one canonical entity.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalize import plain_text
from .registry import (
    ENTITY_A_HEADERS,
    ENTITY_B_HEADERS,
    load_registry,
)

_ISOTOPE_MASS_FIRST_RE = re.compile(
    r"\b(\d{1,3}m?)\s+(lu|ga|ac|cu|zr|tc|sr|sm|ra|re|ho|at|pb|bi|th|f|i|y|c|n|o)\b",
    re.IGNORECASE,
)
_ISOTOPE_ELEMENT_FIRST_RE = re.compile(
    r"\b(lu|ga|ac|cu|zr|tc|sr|sm|ra|re|ho|at|pb|bi|th|f|i|y|c|n|o)\s+(\d{1,3}m?)\b",
    re.IGNORECASE,
)
_DUPLICATED_ISOTOPE_ELEMENT_RE = re.compile(
    r"\b(\d{1,3}m?)(lu|ga|ac|cu|zr|tc|sr|sm|ra|re|ho|at|pb|bi|th|f|i|y|c|n|o)\s+\2\b",
    re.IGNORECASE,
)
_QUALIFIER_ONLY_RE = re.compile(
    r"^(?:who\s+)?(?:stage|grade)\s+(?:[ivx]+|\d+)(?:\s*[abc])?$",
    re.IGNORECASE,
)
_APPLICABILITY_QUALIFIERS = {
    "adult",
    "childhood",
    "pediatric",
    "paediatric",
    "recurrent",
    "relapsed",
    "refractory",
    "metastatic",
    "advanced",
    "progressive",
    "previously treated",
    "treatment resistant",
}
_UNSAFE_EXACT_ALIASES = {
    "diffuse",
    "intestinal",
    "not otherwise specified",
    "not reported",
}
_MAX_PLAUSIBLE_ISOTOPE_MASS = 300


def _join_isotope_tokens(match: re.Match[str], *, mass_group: int, element_group: int) -> str:
    """Join a plausible isotope pair without swallowing construct suffixes.

    Source prose commonly contains ``PSMA-617 (n=12)`` or
    ``PSMA-617 (Lu-PSMA)``.  Treating every three-digit number followed by an
    element symbol as an isotope turns those into ``617n``/``617lu`` and hides
    the legitimate ``177Lu-PSMA-617`` alias.  Known radiopharmaceutical
    isotope mass numbers in this registry are well below this conservative
    physical guardrail.
    """

    mass = match.group(mass_group)
    element = match.group(element_group)
    numeric_mass = int(mass.casefold().removesuffix("m"))
    if numeric_mass > _MAX_PLAUSIBLE_ISOTOPE_MASS:
        return match.group(0)
    return f"{mass}{element}"


def _join_mass_first_isotope(match: re.Match[str]) -> str:
    return _join_isotope_tokens(match, mass_group=1, element_group=2)


def _join_element_first_isotope(match: re.Match[str]) -> str:
    return _join_isotope_tokens(match, mass_group=2, element_group=1)


def _collapse_duplicated_isotope_element(match: re.Match[str]) -> str:
    mass = match.group(1)
    numeric_mass = int(mass.casefold().removesuffix("m"))
    if numeric_mass > _MAX_PLAUSIBLE_ISOTOPE_MASS:
        return match.group(0)
    return f"{mass}{match.group(2)}"


def _search_key(value: object | None) -> str:
    """Normalize punctuation while preserving alphanumeric token boundaries."""

    text = plain_text(value).casefold()
    normalized = "".join(character if character.isalnum() else " " for character in text)
    normalized = " ".join(normalized.split())
    normalized = _ISOTOPE_MASS_FIRST_RE.sub(_join_mass_first_isotope, normalized)
    normalized = _ISOTOPE_ELEMENT_FIRST_RE.sub(_join_element_first_isotope, normalized)
    normalized = _DUPLICATED_ISOTOPE_ELEMENT_RE.sub(
        _collapse_duplicated_isotope_element,
        normalized,
    )
    return normalized


def _usable_alias(
    value: object | None,
    *,
    preferred_name: object | None,
) -> bool:
    key = _search_key(value)
    if len(key.replace(" ", "")) < 3:
        return False
    if not any(character.isalpha() for character in key):
        return False
    if _QUALIFIER_ONLY_RE.fullmatch(key):
        return False
    if key in _UNSAFE_EXACT_ALIASES:
        return False
    preferred_key = _search_key(preferred_name)
    return not any(qualifier in key and qualifier not in preferred_key for qualifier in _APPLICABILITY_QUALIFIERS)


@dataclass(frozen=True)
class _Entity:
    entity_id: str
    preferred_name: str
    aliases: tuple[str, ...]


class AliasCandidateIndex:
    """Exact phrase matcher over aliases that have a single canonical owner."""

    def __init__(self, entities: Iterable[_Entity], *, id_field: str) -> None:
        self.id_field = id_field
        self._entities = {entity.entity_id: entity for entity in entities}
        owners: dict[str, set[str]] = defaultdict(set)
        display_aliases: dict[tuple[str, str], str] = {}
        for entity in self._entities.values():
            for alias in (entity.preferred_name, *entity.aliases):
                if not _usable_alias(alias, preferred_name=entity.preferred_name):
                    continue
                key = _search_key(alias)
                owners[key].add(entity.entity_id)
                display_aliases[(entity.entity_id, key)] = plain_text(alias)
        self._unique_aliases = {
            alias_key: next(iter(entity_ids)) for alias_key, entity_ids in owners.items() if len(entity_ids) == 1
        }
        self._display_aliases = display_aliases

    def match(self, source_text: str) -> list[dict[str, Any]]:
        normalized_source = f" {_search_key(source_text)} "
        matches: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for alias_key, entity_id in self._unique_aliases.items():
            if f" {alias_key} " not in normalized_source:
                continue
            display = self._display_aliases[(entity_id, alias_key)]
            if (
                display.isalpha()
                and display.isupper()
                and len(display) <= 5
                and re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(display)}(?![A-Za-z0-9])",
                    source_text,
                )
                is None
            ):
                continue
            matches[entity_id].append((len(alias_key), display))

        candidates: list[dict[str, Any]] = []
        for entity_id, alias_matches in matches.items():
            entity = self._entities[entity_id]
            ordered_aliases = [
                alias
                for _, alias in sorted(
                    set(alias_matches),
                    key=lambda item: (-item[0], item[1].casefold()),
                )
            ]
            candidates.append(
                {
                    self.id_field: entity_id,
                    "preferred_name": entity.preferred_name,
                    "matched_aliases": ordered_aliases,
                }
            )
        return sorted(
            candidates,
            key=lambda row: (
                -max(len(alias) for alias in row["matched_aliases"]),
                str(row[self.id_field]),
            ),
        )


def _registry_entities(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
) -> list[_Entity]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        entity_id = str(row[id_field])
        preferred_name = str(row["preferred_name"])
        record = merged.setdefault(
            entity_id,
            {"preferred_name": preferred_name, "aliases": set()},
        )
        if preferred_name != record["preferred_name"]:
            record["aliases"].add(preferred_name)
        record["aliases"].update(str(alias) for alias in row.get("aliases", []))
    return [
        _Entity(
            entity_id=entity_id,
            preferred_name=str(record["preferred_name"]),
            aliases=tuple(sorted(record["aliases"], key=str.casefold)),
        )
        for entity_id, record in sorted(merged.items())
    ]


@dataclass(frozen=True)
class SharedCandidateResolver:
    tumor_index: AliasCandidateIndex
    radiopharmaceutical_index: AliasCandidateIndex

    @classmethod
    def from_project(
        cls,
        root: Path,
        *,
        tumor_registry_path: Path | None = None,
        radiopharmaceutical_registry_path: Path | None = None,
        provisional_radiopharmaceutical_path: Path | None = None,
        alias_exclusions_path: Path | None = None,
        auxiliary_radiopharmaceutical_path: Path | None = None,
    ) -> SharedCandidateResolver:
        tumor_registry_path = tumor_registry_path or (
            root / "data" / "seed" / "entities_a.csv"
        )
        radiopharmaceutical_registry_path = radiopharmaceutical_registry_path or (
            root / "data" / "seed" / "entities_b.csv"
        )
        tumor_rows = load_registry(
            tumor_registry_path,
            ENTITY_A_HEADERS,
        )
        radiopharmaceutical_rows = load_registry(
            radiopharmaceutical_registry_path,
            ENTITY_B_HEADERS,
        )

        # Keep the V1.0 registry byte-for-byte frozen while suppressing audited
        # aliases that name a different active construct. This overlay changes
        # candidates only; it never supplies an ID from a reference case.
        exclusion_path = alias_exclusions_path or (
            root / "data" / "v1.1" / "registries" / "alias_exclusions.json"
        )
        if exclusion_path.is_file():
            exclusions = json.loads(exclusion_path.read_text(encoding="utf-8"))
            excluded_aliases: dict[str, set[str]] = defaultdict(set)
            for record in exclusions.get("exclusions", []):
                if record.get("entity_type") != "radiopharmaceutical_active_construct":
                    continue
                excluded_aliases[str(record["entity_id"])].add(_search_key(record["excluded_alias"]))
            for row in radiopharmaceutical_rows:
                blocked = excluded_aliases.get(str(row["radiopharmaceutical_id"]), set())
                if blocked:
                    row["aliases"] = [alias for alias in row.get("aliases", []) if _search_key(alias) not in blocked]

        # Post-cutoff entities are a frozen, external-only open-world registry.
        # They are permitted as candidates only when their names occur in the
        # source text; their IDs are never copied from a reference case.
        provisional_path = provisional_radiopharmaceutical_path or (
            root
            / "evidence"
            / "executor_phase2"
            / "provisional_external_radiopharmaceuticals.json"
        )
        if provisional_path.is_file():
            provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
            radiopharmaceutical_rows.extend(
                {
                    "radiopharmaceutical_id": str(record["radiopharmaceutical_id"]),
                    "preferred_name": str(record["preferred_name"]),
                    "aliases": [],
                }
                for record in provisional.get("records", [])
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
            radiopharmaceutical_rows.extend(
                {
                    "radiopharmaceutical_id": str(record["radiopharmaceutical_id"]),
                    "preferred_name": str(record["preferred_name"]),
                    "aliases": [str(alias) for alias in record.get("aliases", [])],
                }
                for record in auxiliary.get("records", [])
            )

        return cls(
            tumor_index=AliasCandidateIndex(
                _registry_entities(tumor_rows, id_field="tumor_subtype_id"),
                id_field="tumor_subtype_id",
            ),
            radiopharmaceutical_index=AliasCandidateIndex(
                _registry_entities(
                    radiopharmaceutical_rows,
                    id_field="radiopharmaceutical_id",
                ),
                id_field="radiopharmaceutical_id",
            ),
        )

    def resolve_source_text(self, source_text: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "candidate_tumors": self.tumor_index.match(source_text),
            "candidate_radiopharmaceuticals": self.radiopharmaceutical_index.match(source_text),
        }


def candidate_ids(candidates: Iterable[Mapping[str, Any]], id_field: str) -> set[str]:
    """Return IDs exposed by an answer-free candidate interface."""

    return {str(candidate[id_field]) for candidate in candidates if candidate.get(id_field)}
