import ast
import json
import logging
import re
from typing import Any

_DECODER = json.JSONDecoder()
_TRAILING_COMMAS = re.compile(r",\s*([\]}])")  # , }


# --------------------------------------------------------------------------- #
# 1.  Helpers                                                                  #
# --------------------------------------------------------------------------- #
def _strip_code_fences(src: str) -> str:
    """
    Remove leading / trailing ```json ... ``` or ``` ... ``` blocks.

    Args:
        src: The source string that may contain code fences.

    Returns:
        The string with code fences removed and whitespace stripped.
    """
    src = src.strip()
    if src.startswith("```"):
        # everything until the first newline after the opening fence is decoration
        _, _, src = src.partition("\n")
    if src.endswith("```"):
        src = src[:-3]
    return src.strip()


def _remove_trailing_commas(src: str) -> str:
    """
    Best‑effort fix for a *very* common JSON error: a comma before } or ].

    Args:
        src: The JSON string that may contain trailing commas.

    Returns:
        The JSON string with trailing commas removed.
    """
    return _TRAILING_COMMAS.sub(r"\1", src)


def _scan_for_json(text: str, max_objects: int = 10) -> list[tuple[Any, int, int]]:
    """
    Return up to `max_objects` JSON values found in *order of appearance*.

    Args:
        text: The text to scan for JSON objects.
        max_objects: Maximum number of JSON objects to find. Defaults to 10.

    Returns:
        List of tuples where each item is (parsed_object, start_index, end_index).
    """
    pos, end = 0, len(text)
    results: list[tuple[Any, int, int]] = []

    while pos < end and len(results) < max_objects:
        # cheap search for the next plausible opening bracket
        m = re.search(r"[{\[]", text[pos:])
        if not m:
            break
        start = pos + m.start()

        try:
            obj, offs = _DECODER.raw_decode(text[start:])
            results.append((obj, start, start + offs))
            pos = start + offs  # advance after the JSON we just parsed
        except json.JSONDecodeError:
            # ⬇️ NEW: try Python-style literal before advancing
            py_obj, length = _try_python_literal(text[start:])
            if py_obj is not None:
                results.append((py_obj, start, start + length))
                pos = start + length  # jump past the literal we just parsed
            else:
                pos = start + 1  # move forward one char and keep looking
    return results


def _try_python_literal(txt: str):
    """
    Try to parse the *first* Python-style dict/list at the start of `txt`.

    Returns
    -------
    (obj, consumed)  on success – `obj` is the parsed value (dict or list),
                      `consumed` is how many characters were used.
    (None, 0)        if no valid literal is found.
    """
    if not txt or txt[0] not in "{[":
        return None, 0

    open_br, close_br = ("{", "}") if txt[0] == "{" else ("[", "]")
    depth = 0
    in_str: str | None = None  # track quote char we're inside (' or ")
    escaped = False

    for i, ch in enumerate(txt):
        if in_str:
            # we're inside a quoted string
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
        else:
            # not in a string → structural chars matter
            if ch in ('"', "'"):
                in_str = ch
            elif ch == open_br:
                depth += 1
            elif ch == close_br:
                depth -= 1
                if depth == 0:
                    # Found the matching closing bracket
                    segment = txt[: i + 1]
                    try:
                        obj = ast.literal_eval(segment)
                        if isinstance(obj, (dict, list)):
                            return obj, i + 1
                    except (ValueError, SyntaxError):
                        pass
                    break  # malformed literal – give up
    return None, 0


# --------------------------------------------------------------------------- #
# 2.  Public extraction API                                                    #
# --------------------------------------------------------------------------- #
def extract_json(
    response: str,
    fields: list[str] = ("búsquedas", "preguntas"),
) -> dict[str, list[str]] | None:
    """
    Locate the *first* JSON object/array in `response`, try to return
    the requested `fields`.

    Args:
        response: The response string containing JSON data.
        fields: List of field names to extract from the JSON.
                Defaults to ("búsquedas", "preguntas").

    Returns:
        Dictionary with the requested keys when everything is found,
        None otherwise.
    """
    cleaned = _remove_trailing_commas(_strip_code_fences(response))

    # 1️⃣  Find candidate JSON blocks
    candidates = _scan_for_json(cleaned, max_objects=5)
    if not candidates:
        logging.warning("No JSON data detected.")
        return None

    # 2️⃣  Pick the *largest* object/array (more likely to be the full answer)
    target_obj = max(candidates, key=lambda t: len(str(t[0])))[0]
    if not isinstance(target_obj, (dict, list)):
        logging.warning("Top‑level JSON is not a dict/array.")
        return None

    # 3️⃣  If the JSON is an array, merge dict items (common in LLM answers)
    if isinstance(target_obj, list) and all(isinstance(x, dict) for x in target_obj):
        merged: dict[str, Any] = {}
        for item in target_obj:  # last write wins
            merged.update(item)
        target_obj = merged

    # 4️⃣  Best‑effort field matching (case‑insensitive, accent‑insensitive)
    def _canon(s: str) -> str:
        return re.sub(r"\W", "", s).lower()

    found: dict[str, list[str]] = {}
    for f in fields:
        cf = _canon(f)
        for k, v in target_obj.items():
            if _canon(k).startswith(cf):
                found[f] = v
                break
        else:
            found[f] = None

    if all(found.values()):
        return found

    logging.warning(
        "Missing field(s) in extracted JSON: %s", [k for k, v in found.items() if v is None]
    )
    return None
