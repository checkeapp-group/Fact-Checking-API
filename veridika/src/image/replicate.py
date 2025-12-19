from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import re
from typing import Any

from replicate import Client

from veridika.src.api import ApiHandler, get_api_key

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
_PRICING: dict[str, float] = {
    "black-forest-labs/flux-dev": 0.030,
    "black-forest-labs/flux-pro": 0.055,
    "black-forest-labs/flux-schnell": 0.003,
}

_ASPECT_RATIOS: tuple[str, ...] = (
    "1:1",
    "16:9",
    "21:9",
    "2:3",
    "3:2",
    "4:5",
    "5:4",
    "9:16",
    "9:21",
)

# pre-compute numeric values once; avoids eval() and repeated Fraction maths
_ASPECT_RATIO_VALUES: dict[str, float] = {
    r: Fraction(*map(int, r.split(":"))).__float__() for r in _ASPECT_RATIOS
}

_SIZE_RE = re.compile(r"^\s*(\d+)[xX](\d+)\s*$")


# ────────────────────────────────────────────────
# Base wrapper
# ────────────────────────────────────────────────
class ReplicateAPI(ApiHandler):
    """Thin convenience wrapper around the Replicate Python client."""

    def __init__(self, model: str, pretty_name: str | None = None) -> None:
        self.model = model
        self.client = Client(api_token=get_api_key("REPLICATE_API_TOKEN"))
        super().__init__(pretty_name or model)

    def __call__(self, **kwargs: Any) -> Any:
        """
        Forward all kwargs to `replicate.Client.run` and return the raw result.
        """
        return self.client.run(**kwargs)


# ────────────────────────────────────────────────
# Flux specialisation
# ────────────────────────────────────────────────
class Flux(ReplicateAPI):
    """
    Convenience class for the *Flux* image generator.

    Example:
        ```python
        gen = Flux()
        url, cost = gen("sunset over snowy mountains", "1024x768")
        ```
    """

    def __init__(self) -> None:
        super().__init__("black-forest-labs/flux-schnell", pretty_name="flux")

    # ----------------------------------------------------------
    # utility helpers
    # ----------------------------------------------------------
    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        """
        Parse a "WxH" string and return (width, height).

        Raises:
            ValueError: If `size` is not in the expected format.
        """
        m = _SIZE_RE.match(size)
        if not m:
            raise ValueError(
                f"Size must be of the form 'WIDTHxHEIGHT', e.g. '1024x768' " f"(got {size!r})"
            )
        return int(m.group(1)), int(m.group(2))

    @lru_cache(maxsize=32)
    def _closest_aspect_ratio(self, width: int, height: int) -> str:
        """
        Return the aspect-ratio label from `_ASPECT_RATIOS` that best matches
        `width / height`.
        """
        target = width / height
        return min(
            _ASPECT_RATIOS,
            key=lambda r: abs(_ASPECT_RATIO_VALUES[r] - target),
        )

    # ----------------------------------------------------------
    # public call
    # ----------------------------------------------------------
    def __call__(self, image_description: str, size: str, **kwargs) -> tuple[str, float]:
        """
        Generate an image and return (**url**, **cost**).

        Args:
            image_description: Prompt sent to Flux.
            size: Requested resolution, `"WIDTHxHEIGHT"`.
            **kwargs: Extra keyword arguments forwarded to `replicate.Client.run`.

        Returns:
            url: URL of the generated image.
            cost: Cost in USD for this invocation.
        """
        width, height = self._parse_size(size)
        aspect_ratio = self._closest_aspect_ratio(width, height)

        inputs = {
            "prompt": image_description,
            "num_outputs": 1,
            "aspect_ratio": aspect_ratio,
            "output_format": "webp",
            "output_quality": 80,
        }
        inputs.update(kwargs)  # allow caller overrides

        try:
            url = self.client.run(
                self.model,
                input=inputs,
                use_file_output=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"Replicate call failed: {exc}") from exc

        cost = _PRICING[self.model]
        self.add_cost(cost)
        return url, cost
