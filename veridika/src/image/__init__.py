from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, overload

from veridika.src.api import ApiHandler


def _load_local_flux_config() -> dict:
    """Load local Flux config from `configs/local-model-config/flux-config.json`.

    Returns an empty dict if the file does not exist or is invalid.
    """
    try:
        # Compute repo root: .../veridika/src/image/__init__.py -> root is parents[3]
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "configs" / "local-model-config" / "flux-config.json"
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


class Image(ApiHandler):
    """
    Factory-style wrapper that yields either a `ReplicateAPI` or a `Local Flux API`
    instance depending on *model*.

    Example
    -------
    ```python
    image = Image("flux")  # -> actually a Replicate API instance
    other = Image("flux-local")  # -> actually a local Flux instance
    ```
    """

    # — Overloads for static type checkers (optional but nice)  —
    @overload
    def __new__(cls, model: Literal["flux"]): ...

    def __new__(cls, model: str):  # noqa: D401
        # Prefer configured local models when present in config file
        cfg = _load_local_flux_config()
        models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        if isinstance(models_cfg, dict) and model in models_cfg:
            from .confyui import ConfiUI

            model_cfg = models_cfg[model]
            return ConfiUI(
                model=model_cfg["model_name"],
                runpod_url_env_name=model_cfg["confyui_url"],  # env with full Runpod endpoint URL
                api_key_env_name=model_cfg["api_key_name"],
                pretty_name=model,
            )

        # Otherwise use Replicate Flux
        else:
            from .replicate import Flux

            return Flux()
