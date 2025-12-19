from __future__ import annotations

import json
from pathlib import Path
from typing import overload

from veridika.src.api import ApiHandler

__all__ = ["LLM"]


def _load_local_llm_config() -> dict:
    """Load local LLM config from `configs/local-model-config/llm-config.json`.

    Returns an empty dict if the file does not exist or is invalid.
    """
    try:
        # Compute repo root: .../veridika/src/llm/__init__.py -> root is parents[3]
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "configs" / "local-model-config" / "llm-config.json"
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


class LLM(ApiHandler):
    """
    Factory-style wrapper that yields either a `VLLM` or an `OpenRouter`
    instance depending on *model*.

    Example
    -------
    ```python
    llm = LLM("latxa")  # -> actually a VLLM instance
    other = LLM("gpt-4o")  # -> actually an OpenRouter instance
    ```
    """

    # — Overloads for static type checkers (optional but nice)  —
    @overload
    def __new__(cls, model: str): ...

    def __new__(cls, model: str):  # noqa: D401
        # Prefer configured local models when present in config file
        cfg = _load_local_llm_config()
        models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        if isinstance(models_cfg, dict) and model in models_cfg:
            from .vllm import VLLM

            model_cfg = models_cfg[model]
            # Pass through the friendly name as pretty_name and the config dict
            return VLLM(
                model=model_cfg["model_name"],
                vllm_url_env_name=model_cfg["vllm_url"],
                api_key_env_name=model_cfg["api_key_name"],
                pretty_name=model,
            )

        # Otherwise use OpenRouter
        else:
            from .openrouter import OpenRouter

            return OpenRouter(model=model)
