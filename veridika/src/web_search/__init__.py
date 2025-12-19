from __future__ import annotations

from typing import Literal, overload

from veridika.src.api import ApiHandler


class WebSearch(ApiHandler):
    """
    Factory-style wrapper that yields either a `Serper` instance.

    Example
    -------
    ```python
    web_search = WebSearch("serper")  # -> actually a Serper instance
    ```
    """

    # — Overloads for static type checkers (optional but nice)  —
    @overload
    def __new__(cls, model: Literal["serper"]): ...

    def __new__(cls, model: str):  # noqa: D401
        from .serper import Serper

        if "serper" in model:
            target_cls = Serper
        else:
            raise ValueError(f"Model {model} not found")
        # forward *all* args so their own __init__ signatures are honoured
        return target_cls()
