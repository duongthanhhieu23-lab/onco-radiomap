from __future__ import annotations

import re

SECTION = re.compile(
    r"(?:^|\s)(BACKGROUND|INTRODUCTION|PURPOSE|OBJECTIVES?|METHODS?(?: AND MATERIALS)?|"
    r"MATERIALS AND METHODS|PATIENTS AND METHODS|EXPERIMENTAL DESIGN|RESULTS?|FINDINGS|CONCLUSIONS?|"
    r"INTERPRETATION|DISCUSSION)\s*:\s*",
    re.IGNORECASE,
)

def segments(source_text: str) -> list[dict[str, str]]:
    title_match = re.search(r"(?is)^Title:\s*(.*?)(?:\s*Abstract:|$)", source_text)
    rows = [{"section": "TITLE", "text": title_match.group(1).strip()}] if title_match else []
    abstract = re.split(r"\bAbstract:\s*", source_text, maxsplit=1, flags=re.IGNORECASE)[-1]
    abstract = re.split(r"\nDOI:", abstract, maxsplit=1, flags=re.IGNORECASE)[0]
    markers = list(SECTION.finditer(abstract))
    blocks = []
    if markers:
        if markers[0].start():
            blocks.append(("UNSTRUCTURED", abstract[: markers[0].start()]))
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(abstract)
            blocks.append((marker.group(1).upper(), abstract[marker.end() : end]))
    else:
        blocks.append(("UNSTRUCTURED", abstract))
    for section, body in blocks:
        for sentence in re.split(r"(?<=[.!?])\s+|;\s*", body):
            if sentence.strip():
                rows.append({"section": section, "text": sentence.strip()})
    return rows

def balanced_excerpt(text: str, *, max_chars: int = 3600) -> str:
    """Budget complete sentences across title, methods, results and conclusion."""
    if len(text) <= max_chars:
        return text
    rows = segments(text)
    groups = [[], [], [], [], []]
    for index, row in enumerate(rows):
        section = row["section"]
        group = (
            0
            if section == "TITLE"
            else 1
            if "METHOD" in section
            else 2
            if section in {"RESULT", "RESULTS", "FINDINGS"}
            else 3
            if section in {"CONCLUSION", "CONCLUSIONS", "INTERPRETATION"}
            else 4
        )
        groups[group].append((index, row))
    selected = []
    size = 0
    # First reserve one sentence from each key section. Then add results first.
    order = [g[0] for g in groups[:4] if g]
    order += [item for group in (groups[2], groups[3], groups[1], groups[0], groups[4]) for item in group]
    used = set()
    for index, row in order:
        rendered = f"{row['section']}: {row['text']}"
        if index in used or size + len(rendered) + (1 if selected else 0) > max_chars:
            continue
        used.add(index)
        selected.append((index, rendered))
        size += len(rendered) + (1 if len(selected) > 1 else 0)
    return "\n".join(value for _, value in sorted(selected))
