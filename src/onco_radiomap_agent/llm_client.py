"""OpenAI-compatible chat client used by the LightRAG adapter.

Provides complete(prompt, system), using the OpenAI SDK when installed and
urllib otherwise. Missing credentials or provider errors produce a response
marked degraded=True.

Environment settings:
  HUANYU_BULK_API_KEY       DEEPSEEK_API_KEY is the fallback
  HUANYU_BULK_MODEL         provider model ID (default deepseek-v4-flash)
  HUANYU_BULK_BASE_URL      OpenAI-compatible base URL
  HUANYU_BULK_TEMPERATURE   default 0.0
  HUANYU_BULK_MAX_TOKENS    default 8192
  HUANYU_BULK_THINKING      enabled | disabled (default disabled)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class ChatResponse:
    text: str
    degraded: bool = False
    usage: dict[str, Any] | None = None


class LLMClient:
    """OpenAI-compatible chat client.

    Parameters
    ----------
    provider : str
        Any OpenAI-compatible provider is supported through HUANYU_BULK_*.
        "stub" forces degraded mode for offline tests.
    temperature, max_tokens : override the env defaults.
    """

    def __init__(
        self,
        provider: str = "deepseek",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.degraded = False

        if provider == "stub":
            self.degraded = True
            self.api_key = ""
            self.base_url = ""
            self.model = "stub"
            self.temperature = temperature if temperature is not None else 0.0
            self.max_tokens = max_tokens or 1024
            return

        # DeepSeek-first OpenAI-compatible resolution. Project-local HUANYU_*
        # aliases win, followed by the skill-standard DeepSeek variables.
        self.api_key = (
            os.getenv("HUANYU_BULK_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            os.getenv("HUANYU_BULK_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.deepseek.com/v1"
        )
        if self.base_url.rstrip("/") == "https://api.deepseek.com":
            self.base_url = "https://api.deepseek.com/v1"
        self.model = (
            os.getenv("HUANYU_BULK_MODEL")
            or os.getenv("LLM_CHAT_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
            or "deepseek-v4-flash"
        )
        self.temperature = float(
            temperature if temperature is not None
            else os.getenv("HUANYU_BULK_TEMPERATURE", "0.0")
        )
        self.max_tokens = int(
            max_tokens or os.getenv("HUANYU_BULK_MAX_TOKENS", "8192")
        )
        self.thinking_mode = os.getenv("HUANYU_BULK_THINKING", "disabled").lower()
        if self.thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("HUANYU_BULK_THINKING must be 'enabled' or 'disabled'")

        if not self.api_key:
            # no key → degrade to stub so the build can proceed in keyword-only mode
            self.degraded = True
            self.model = "stub"

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse:
        """Return a chat completion. ``degraded=True`` if in stub mode."""
        if self.degraded:
            return ChatResponse(
                text=f"[stub] no LLM key configured; prompt was {len(prompt)} chars.",
                degraded=True,
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            text, usage = self._chat(messages, response_format=response_format)
            return ChatResponse(text=text, degraded=False, usage=usage)
        except Exception as e:  # noqa: BLE001 - provider failures degrade safely
            # network / API failure → degrade rather than crash the build
            return ChatResponse(text=f"[llm error] {e}", degraded=True)

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Try the OpenAI SDK first, fall back to raw urllib."""
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            request = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                # DeepSeek V4 enables high-effort thinking by default. Structured
                # extraction needs the visible token budget for delimiter-safe
                # output, so the project explicitly freezes the toggle.
                "extra_body": {"thinking": {"type": self.thinking_mode}},
            }
            if response_format is not None:
                request["response_format"] = response_format
            resp = client.chat.completions.create(
                **request,
            )
            usage = resp.usage.model_dump() if resp.usage is not None else {}
            return resp.choices[0].message.content or "", usage
        except ImportError:
            pass
        # raw urllib fallback (no openai SDK installed)
        return self._chat_urllib(messages, response_format=response_format)

    def _chat_urllib(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> tuple[str, dict[str, Any]]:
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": self.thinking_mode},
        }
        if response_format is not None:
            request["response_format"] = response_format
        body = json.dumps(request).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return "", dict(data.get("usage") or {})
        msg = choices[0].get("message", {})
        # Only the provider's final answer is returned; hidden reasoning is not persisted.
        return msg.get("content") or "", dict(data.get("usage") or {})


def from_env_dual():
    """Convenience factory referenced by setup_environment.py docstring.

    Returns the project chat client configured from env.
    """
    return LLMClient(provider="deepseek")
