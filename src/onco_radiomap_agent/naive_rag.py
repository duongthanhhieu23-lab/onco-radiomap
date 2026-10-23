"""Dataset-isolated BM25 retrieval baseline over the frozen evidence corpus."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[:._+\-/][A-Za-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class BM25Hit:
    record: dict[str, Any]
    score: float


class BM25EvidenceIndex:
    """Small dependency-free BM25 index with dataset membership filtering."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.records = list(records)
        self.k1 = k1
        self.b = b
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for record in self.records:
            counts = Counter(_tokens(str(record.get("text", ""))))
            self._term_frequencies.append(counts)
            length = sum(counts.values())
            self._lengths.append(length)
            for term in counts:
                document_frequency[term] += 1
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        n_documents = len(self.records)
        self._idf = {
            term: math.log(1.0 + (n_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        dataset_id: str,
        population_scope: str | None = None,
        top_k: int = 4,
    ) -> list[BM25Hit]:
        query_terms = list(dict.fromkeys(_tokens(query)))
        scored: list[BM25Hit] = []
        for index, record in enumerate(self.records):
            if dataset_id not in record.get("dataset_membership", []):
                continue
            if population_scope is not None and population_scope not in record.get(
                "population_membership", []
            ):
                continue
            frequencies = self._term_frequencies[index]
            document_length = self._lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                length_normalizer = 1.0 - self.b
                if self._average_length:
                    length_normalizer += self.b * document_length / self._average_length
                denominator = frequency + self.k1 * length_normalizer
                score += self._idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scored.append(BM25Hit(record=record, score=score))
        scored.sort(key=lambda hit: (-hit.score, str(hit.record.get("chunk_id", ""))))
        return scored[:top_k]

    def context(
        self,
        query: str,
        *,
        dataset_id: str,
        population_scope: str | None = None,
        top_k: int = 4,
        max_chars: int | None = None,
    ) -> str:
        blocks: list[str] = []
        for rank, hit in enumerate(
            self.search(
                query,
                dataset_id=dataset_id,
                population_scope=population_scope,
                top_k=top_k,
            ),
            start=1,
        ):
            text = str(hit.record.get("text", ""))
            block = f"[BM25 rank={rank} score={hit.score:.4f}]\n{text}"
            separator_length = 2 if blocks else 0
            current_length = sum(len(item) for item in blocks) + 2 * max(
                len(blocks) - 1, 0
            )
            if (
                max_chars is not None
                and current_length + separator_length + len(block) > max_chars
            ):
                if not blocks:
                    blocks.append(block[:max_chars].rstrip())
                break
            blocks.append(block)
        return "\n\n".join(blocks)
