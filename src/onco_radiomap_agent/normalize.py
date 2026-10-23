"""Deterministic text normalization used by the two project registries.

The functions in this module deliberately stop short of fuzzy entity merging.
Cross-standard mappings are optional and may be many-to-many; an ambiguous
alias is quarantined by the registry builder instead of being forced into a
canonical identity.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_NUCLIDE_MASS_FIRST_RE = re.compile(r"(?<!\d)(\d{1,3}m?)\s*[- ]?\s*([A-Za-z]{1,3})(?![A-Za-z])", re.IGNORECASE)
_NUCLIDE_ELEMENT_FIRST_RE = re.compile(r"(?<![A-Za-z])([A-Za-z]{1,3})\s*[- ]?\s*(\d{1,3}m?)(?!\d)", re.IGNORECASE)
_ELEMENT_NAME_FIRST_RE = re.compile(
    r"\b(lutetium|gallium|fluorine|iodine|yttrium|actinium|copper|zirconium|technetium|indium|carbon|nitrogen|oxygen)"
    r"\s*[- ]?\s*(\d{1,3}m?)\b",
    re.IGNORECASE,
)
_ELEMENT_SYMBOLS = {
    "lutetium": "Lu",
    "gallium": "Ga",
    "fluorine": "F",
    "iodine": "I",
    "yttrium": "Y",
    "actinium": "Ac",
    "copper": "Cu",
    "zirconium": "Zr",
    "technetium": "Tc",
    "indium": "In",
    "carbon": "C",
    "nitrogen": "N",
    "oxygen": "O",
}


def plain_text(value: object | None) -> str:
    """Return a compact NFKC string with HTML tags removed."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _HTML_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalized_alias_key(value: object | None) -> str:
    """Normalize an alias for exact collision checks, never for fuzzy merging."""

    text = plain_text(value).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return _SPACE_RE.sub(" ", text).strip()


def stable_unique(values: Iterable[object | None]) -> list[str]:
    """Return non-empty strings deduplicated by normalized key in stable order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = plain_text(value)
        key = normalized_alias_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def normalize_radionuclide(value: object | None) -> str:
    """Normalize ``177Lu``/``Lu-177``/HTML superscript variants to ``177Lu``."""

    text = plain_text(value)
    match = _NUCLIDE_MASS_FIRST_RE.search(text)
    if match:
        mass, element = match.groups()
    else:
        match = _NUCLIDE_ELEMENT_FIRST_RE.search(text)
        if match:
            element, mass = match.groups()
        else:
            name_match = _ELEMENT_NAME_FIRST_RE.search(text)
            if not name_match:
                return text
            element_name, mass = name_match.groups()
            element = _ELEMENT_SYMBOLS[element_name.casefold()]
    element = element[0].upper() + element[1:].lower()
    mass = mass.lower()
    normalized_mass = f"{int(mass[:-1])}m" if mass.endswith("m") else str(int(mass))
    return f"{normalized_mass}{element}"


def split_semicolon_aliases(value: object | None, *, limit: int = 50) -> list[str]:
    """Split a source synonym field without treating component synonyms as drug aliases."""

    text = plain_text(value)
    if not text or text == ".":
        return []
    return stable_unique(part for part in re.split(r"\s*;\s*", text))[:limit]


def stable_local_id(namespace: str, parts: Iterable[object | None], *, length: int = 16) -> str:
    """Create a stable local identifier from an explicit canonical identity tuple."""

    canonical = "\x1f".join(normalized_alias_key(part) for part in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length].upper()
    return f"{namespace}:{digest}"
