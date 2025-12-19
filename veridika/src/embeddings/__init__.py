from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, overload

from veridika.src.api import ApiHandler

__all__ = ["Embeddings"]

_open_ai_models = {"text-embedding-3-small", "text-embedding-3-large"}
_gemini_models = {
    "gemini-embedding-exp-03-07",
    "gemini-embedding",
    "gemini-embedding-001",
}


def _load_local_embedding_config() -> dict:
    """Load local embedding config from `configs/local-model-config/embedding-config.json`.

    Returns an empty dict if the file does not exist or is invalid.
    """
    try:
        # Compute repo root: .../veridika/src/embeddings/__init__.py -> root is parents[3]
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "configs" / "local-model-config" / "embedding-config.json"
        if not cfg_path.exists():
            return {}
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        # Be conservative and fall back to remote providers on any error
        return {}


class Embeddings(ApiHandler):
    """
    Factory-style wrapper that yields either a `VLLM` or an `OpenRouter`
    instance depending on *model*.

    Example
    -------
    ```python
    llm = Embeddings("latxa")  # -> actually a VLLM instance
    other = Embeddings("gpt-4o")  # -> actually an OpenRouter instance
    ```
    """

    # — Overloads for static type checkers (optional but nice)  —
    @overload
    def __new__(cls, model: Literal["text-embedding-3-small", "text-embedding-3-large"]): ...
    @overload
    def __new__(
        cls,
        model: Literal["intfloat/e5-mistral-7b-instruct", "intfloat/multilingual-e5-large"],
    ): ...
    @overload
    def __new__(
        cls,
        model: Literal["gemini-embedding-exp-03-07", "gemini-embedding", "gemini-embedding-001"],
    ): ...

    def __new__(cls, model: str):  # noqa: D401
        cfg = _load_local_embedding_config()
        models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        if isinstance(models_cfg, dict) and model in models_cfg:
            from .vllm_embeddings import VLLMEmbeddings

            model_cfg = models_cfg[model]
            return VLLMEmbeddings(
                model=model_cfg["model_name"],
                vllm_url_env_name=model_cfg["vllm_url"],
                api_key_env_name=model_cfg["api_key_name"],
                pretty_name=model,
            )

        else:
            if model in _open_ai_models:
                from .openai import OpenAIEmbeddings

                target_cls = OpenAIEmbeddings
            elif model in _gemini_models:
                from .gemini import GeminiEmbeddings

                target_cls = GeminiEmbeddings
            else:
                raise ValueError(f"Invalid model: {model}")
            return target_cls(model=model)
