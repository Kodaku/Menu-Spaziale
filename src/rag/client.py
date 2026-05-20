"""
LLM client factory: returns a callable(prompt, system) -> str.

Supported backends:
  "groq"              — Groq API (uses GROQ_API_KEY env var, model llama-3.1-8b-instant)
  "ollama:<model>"    — local Ollama (e.g. "ollama:qwen2.5:7b")

Both use openai.OpenAI with a custom base_url since both Groq and Ollama
expose an OpenAI-compatible chat.completions endpoint.
"""

import os
import time
from typing import Callable
from constants import API_KEY_ENV_NAME, GROQ_MODEL_LLAMA_INSTANT

from openai import OpenAI, RateLimitError

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"

_MAX_RETRIES = 3


def _make_caller(oai_client: OpenAI, model: str) -> Callable[[str, str], str]:
    def call(prompt: str, system: str, max_tokens: int = 256) -> str:
        delay = 30
        for attempt in range(_MAX_RETRIES):
            try:
                resp = oai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                print(f"\n  rate limit, retry {attempt + 2}/{_MAX_RETRIES} in {delay * 3}s: {e}")
                time.sleep(delay * 3)
                delay *= 2
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                print(f"\n  retry {attempt + 2}/{_MAX_RETRIES} in {delay}s: {e}")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("all retries exhausted")

    return call


def make_client(backend: str) -> Callable[[str, str], str]:
    """Return a callable(prompt, system) -> str for the given backend.

    Args:
        backend: "groq" or "ollama:<model_name>"
    """
    if backend == "groq":
        api_key = os.environ.get(API_KEY_ENV_NAME)
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        oai = OpenAI(base_url=_GROQ_BASE_URL, api_key=api_key)
        # Qui si decide di utilizzare Llama Instant perché si chiede solamente di elencare i numeri dei piatti non di fare chissà quale task complesso.
        return _make_caller(oai, GROQ_MODEL_LLAMA_INSTANT)

    if backend.startswith("ollama:"):
        model = backend[len("ollama:"):]
        oai = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
        return _make_caller(oai, model)

    raise ValueError(f"Unknown backend '{backend}'. Use 'groq' or 'ollama:<model>'.")
