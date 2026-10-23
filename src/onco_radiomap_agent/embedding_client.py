"""Volcengine Ark Doubao embedding client (doubao-embedding-vision-251215).

NOTE (v2.6): the public multimodal embedding endpoint
``/api/v3/embeddings/multimodal`` differs from the standard OpenAI-compatible
``/embeddings`` endpoint in THREE important ways that earlier docs got wrong:

1. **Request body uses ``input`` (NOT ``content``).** The old env template /
   playbook described ``{"content":[{"type":"text","text":"..."}]}`` which the
   Ark API rejects with ``MissingParameter: param=input``. The correct shape is
   ``{"model":"...","input":[{"type":"text","text":"..."}],"encoding_format":"float"}``.

2. **One input per call (no batch).** Sending multiple items in ``input`` does
   NOT error, but only the FIRST embedding is returned — silently dropping the
   rest. So we must call the endpoint once per text and fan out concurrently.

3. **Response is a single object, not a list.** ``{"data":{"embedding":[...]}}``
   instead of OpenAI's ``{"data":[{"embedding":[...]}]}``.

Additionally (empirically verified), **pure short Chinese text occasionally
returns HTTP 500 ``InternalServiceError``** while mixed EN+ZH or pure English
works. This client adds a CJK-detection fallback: on a 5xx retry-exhaustion for
text containing CJK, it prepends a benign English prefix (``"[text] "``) and
retries once — this flips pure-Chinese inputs into the working mixed shape.

The class implements the LightRAG ``embedding_func`` protocol
(``embed(texts)->list[list[float]]`` + async ``aembed``) so it can be plugged
into ``LightRAG(embedding_func=...)`` directly.

Reads from env (via ``.env`` / os.environ):
  HUANYU_EMBED_ENDPOINT_ID   preferred model field (e.g. ep-20260713...)
  HUANYU_EMBED_MODEL         fallback model field (doubao-embedding-vision-251215)
  HUANYU_EMBED_API_KEY       == ARK_DOUBAO_API_KEY
  HUANYU_EMBED_BASE_URL      https://ark.cn-beijing.volces.com/api/v3
  HUANYU_EMBED_DIM           expected dim (default 2048); mismatch => RuntimeError
  HUANYU_EMBED_CONCURRENCY   max parallel single-text calls (default 8)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# CJK Unified Ideographs ranges (for the pure-Chinese 500-error fallback).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def _has_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


class ArkVisionEmbedding:
    """LightRAG-compatible embedding_func for doubao-embedding-vision-251215.

    LightRAG calls ``embedding_func.embed(texts: list[str]) -> list[list[float]]``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint_id: str | None = None,
        embedding_dim: int | None = None,
        concurrency: int | None = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        allow_cjk_fallback: bool = True,
    ) -> None:
        # env resolution: HUANYU_EMBED_* take precedence, fall back to ARK_DOUBAO_*
        self.api_key = api_key or os.getenv("HUANYU_EMBED_API_KEY") or os.getenv("ARK_DOUBAO_API_KEY") or ""
        self.base_url = (
            base_url
            or os.getenv("HUANYU_EMBED_BASE_URL")
            or os.getenv("ARK_DOUBAO_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3"
        ).rstrip("/")
        # endpoint_id is the VOLC inference endpoint ID; model is the model family name.
        # Ark accepts EITHER as the `model` field. Prefer endpoint_id when set.
        self.endpoint_id = endpoint_id or os.getenv("HUANYU_EMBED_ENDPOINT_ID") or ""
        self.model = (
            model or os.getenv("HUANYU_EMBED_MODEL") or os.getenv("LLM_EMBED_MODEL") or "doubao-embedding-vision-251215"
        )
        self.model_field = self.endpoint_id or self.model
        self.embedding_dim = int(
            embedding_dim or os.getenv("HUANYU_EMBED_DIM") or os.getenv("ARK_DEFAULT_EMBED_DIM") or 2048
        )
        self.concurrency = int(concurrency or os.getenv("HUANYU_EMBED_CONCURRENCY") or 8)
        self.timeout = timeout
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.allow_cjk_fallback = allow_cjk_fallback
        self._url = f"{self.base_url}/embeddings/multimodal"
        self._dim_verified = False

        if not self.api_key:
            raise RuntimeError(
                "ArkVisionEmbedding: no API key. Set HUANYU_EMBED_API_KEY (or ARK_DOUBAO_API_KEY) in .env."
            )

    # ------------------------------------------------------------------ #
    # single-text call (the endpoint only supports 1 input per request)
    # ------------------------------------------------------------------ #
    def _post(self, text: str, *, cn_prefix: bool = False) -> list[float]:
        payload_text = ("[text] " + text) if cn_prefix else text
        body = {
            "model": self.model_field,
            "input": [{"type": "text", "text": payload_text}],
            "encoding_format": "float",
        }
        data = self._post_json(body)
        # response shape: {"data":{"embedding":[...]}} (single object)
        emb_block = data.get("data") if isinstance(data, dict) else None
        if isinstance(emb_block, dict) and "embedding" in emb_block:
            return list(emb_block["embedding"])
        # tolerate OpenAI-style list just in case Ark changes
        if isinstance(emb_block, list) and emb_block and "embedding" in emb_block[0]:
            return list(emb_block[0]["embedding"])
        raise RuntimeError(f"ArkVisionEmbedding: unexpected response shape from {self._url}: {str(data)[:200]}")

    def _post_json(self, body: dict) -> dict:
        """POST with configurable backoff on 429/5xx.

        On final 5xx failure for CJK text, raises _RetryExhausted so the caller
        can retry with the English prefix fallback.
        """
        req_body = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(self._url, data=req_body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                last_err = e
                # 4xx (except 429) is a hard error — don't retry
                if e.code != 429 and not (500 <= e.code < 600):
                    snippet = ""
                    try:
                        snippet = e.read().decode("utf-8", errors="ignore")[:200]
                    except (OSError, UnicodeError):
                        snippet = ""
                    raise RuntimeError(f"ArkVisionEmbedding HTTP {e.code} for {self._url}: {snippet}") from e
                # 429 / 5xx → backoff only when another attempt is allowed.
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.5 * (2**attempt))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.5 * (2**attempt))
        # exhausted retries — signal CJK fallback if applicable
        raise _RetryExhausted(last_err)

    def _embed_one(self, text: str) -> list[float]:
        """Embed one text, with the pure-Chinese 500-error fallback."""
        try:
            return self._post(text)
        except _RetryExhausted:
            if self.allow_cjk_fallback and _has_cjk(text) and not text.startswith("[text] "):
                # flip pure-Chinese into the working mixed shape
                return self._post(text, cn_prefix=True)
            raise
        except RuntimeError:
            # also try the CN fallback on an explicit InternalServiceError body
            if self.allow_cjk_fallback and _has_cjk(text) and not text.startswith("[text] "):
                try:
                    return self._post(text, cn_prefix=True)
                except Exception as fallback_error:
                    raise RuntimeError("Ark CJK fallback request failed") from fallback_error
            raise

    # ------------------------------------------------------------------ #
    # LightRAG embedding_func protocol
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        # fan out single-text calls concurrently (endpoint has no real batch)
        max_workers = max(1, min(self.concurrency, len(texts)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(self._embed_one, texts))
        # one-time dim self-check
        if not self._dim_verified:
            got = len(results[0])
            if got != self.embedding_dim:
                raise RuntimeError(
                    f"ArkVisionEmbedding dim mismatch: model returned {got}-d vectors "
                    f"but HUANYU_EMBED_DIM={self.embedding_dim}. "
                    f"Update HUANYU_EMBED_DIM to {got} or switch embedding model."
                )
            self._dim_verified = True
        return results

    async def aembed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        # LightRAG's async path; we reuse the sync thread-pool impl (Ark has no
        # async SDK requirement; thread fan-out is sufficient for index builds).
        return self.embed(texts, **kwargs)

    # ------------------------------------------------------------------ #
    # smoke test
    # ------------------------------------------------------------------ #
    def selfcheck(self) -> dict:
        """Run a tiny known-answer test. Returns a status dict.

        Verifies: (1) endpoint reachable, (2) dim matches config,
        (3) pure-Chinese fallback path works, (4) determinism (same text → same vec).
        """
        report: dict[str, Any] = {"ok": False}
        try:
            v_en = self._embed_one("hello world")
            # Chinese input (chronic obstructive pulmonary disease) tests the CJK fallback.
            v_zh = self._embed_one("\u6162\u6027\u963b\u585e\u6027\u80ba\u75be\u75c5 PM2.5")
            report["dim"] = len(v_en)
            report["dim_match"] = len(v_en) == self.embedding_dim
            report["cn_fallback_worked"] = len(v_zh) == len(v_en)
            report["model_field"] = self.model_field
            report["ok"] = bool(report["dim_match"] and report["cn_fallback_worked"])
        except Exception as e:  # noqa: BLE001 - selfcheck must return a report
            report["error"] = str(e)
        return report


class _RetryExhausted(RuntimeError):
    """Internal: configured attempts exhausted on a retriable error."""
