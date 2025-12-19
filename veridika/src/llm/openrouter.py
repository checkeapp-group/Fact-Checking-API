import json
import logging
import os
import time
from typing import Any

from pydantic import BaseModel
import requests

from veridika.src.api import ApiHandler
from veridika.src.llm.utils import extract_json


# ──────────────────────────────────────────────────────────────
# Helper: make every object in a Pydantic-generated schema strict
# ──────────────────────────────────────────────────────────────
def strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Return a JSON-Schema dict for `model_cls` where the root object
    **and all nested definitions** set  `"additionalProperties": false`.

    Works for both Pydantic v2 (`model_json_schema`) and v1 (`schema`).

    Args:
        model_cls: The Pydantic model class to generate a JSON schema for.

    Returns:
        A JSON schema dict for `model_cls` where the root object
        **and all nested definitions** set  `"additionalProperties": false`.
    """
    # 1️⃣  Get the ordinary schema from Pydantic
    if hasattr(model_cls, "model_json_schema"):  # v2
        schema: dict[str, Any] = model_cls.model_json_schema(mode="serialization")
    else:  # v1
        schema = model_cls.schema()

    # 2️⃣  Mutate it in-place
    def _forbid_extras(node: dict[str, Any]) -> None:
        """
        Recursively mutates a JSON schema node to set "additionalProperties" to False
        for all objects.

        Args:
            node: The JSON schema node (dictionary) to mutate.
        """
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)

        # Dive into nested structures that can hold sub-schemas
        for key in ("properties", "$defs", "definitions", "patternProperties"):
            if key in node and isinstance(node[key], dict):
                for child in node[key].values():
                    if isinstance(child, dict):
                        _forbid_extras(child)

        # Handle array item schemas
        if "items" in node and isinstance(node["items"], dict):
            _forbid_extras(node["items"])

    _forbid_extras(schema)
    return schema


# ──────────────────────────────────────────────────────────────
# Utility: fetch pricing once, keep it on the instance
# ──────────────────────────────────────────────────────────────
def get_model_info(model_name: str) -> tuple[float, float, float]:
    """
    Fetches the pricing information for a given model and the supported parameters from the OpenRouter API.

    Args:
        model_name: The name of the model to get pricing for.

    Returns:
        A tuple containing the prompt token price, completion token price, and request price, and the supported parameters.

    Raises:
        ValueError: If the model is not found in the OpenRouter list.
    """
    url = f"https://openrouter.ai/api/v1/models/{model_name}/endpoints"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data: list[dict[str, Any]] = response.json()["data"]

    pricing = data["endpoints"][0]["pricing"]
    parameters = set(data["endpoints"][0]["supported_parameters"])
    return (
        float(pricing["prompt"]),
        float(pricing["completion"]),
        float(pricing["request"]),
        parameters,
    )


class OpenRouter(ApiHandler):
    """
    Wrap an OpenRouter chat completion endpoint with optional
    schema-validated structured output.
    """

    def __init__(self, model: str) -> None:
        """
        Initializes the LLM handler.

        Args:
            model_name: The name of the OpenRouter model to use.
        """
        super().__init__(model)
        self.api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        self.model_name: str = model
        (
            self.input_token_price,
            self.output_token_price,
            self.request_price,
            self.parameters,
        ) = get_model_info(model)

        self.supports_structured_output = (
            "response_format" in self.parameters or "structured_output" in self.parameters
        )

    # ----------------------------------------------------------
    # Call the model
    # ----------------------------------------------------------
    def __call__(
        self,
        messages: list[dict],
        pydantic_model: type[BaseModel] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[BaseModel | str, float]:
        """
        Calls the OpenRouter chat completion API with the given messages and an optional Pydantic model for response validation.

        Args:
            messages: A list of message dictionaries to send to the API.
            pydantic_model: An optional Pydantic model to validate and parse the API response.
            tools: Optional tools that the model can call.

        Returns:
            A tuple containing the parsed API response (either a Pydantic model instance, string, or full response dict for tools) and the cost of the API call.

        Raises:
            Exception: If the API call fails or returns a non-200 status code.
        """
        data: dict[str, Any] = {"model": self.model_name, "messages": messages}

        # ── Add tools to the request if provided
        if tools is not None:
            data["tools"] = tools

        # ── If the caller supplied a Pydantic class, attach a strict schema
        if pydantic_model is not None and self.supports_structured_output:
            json_schema = strict_schema(pydantic_model)
            schema_name = json_schema.get("title", pydantic_model.__name__).lower()

            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,  # OpenRouter's own flag
                    "schema": json_schema,
                },
            }

        # ------------------------------------------------------
        # Make the request with retry logic
        # ------------------------------------------------------
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(data),
                    timeout=300,
                )
                break  # Success, exit retry loop
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise RuntimeError(
                        f"OpenRouter API failed after {max_retries} attempts. Last error: {e.__class__.__name__}: {str(e)}"
                    ) from e

                # Calculate delay with exponential backoff
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"OpenRouter API attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        # ------------------------------------------------------
        # Error handling
        # ------------------------------------------------------
        if response.status_code != 200:
            try:
                payload = response.json()
            except Exception:
                payload_text = response.text
                raise RuntimeError(f"OpenRouter HTTP {response.status_code} error: {payload_text}")
            # Include any structured error information if present
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                if isinstance(err, dict):
                    msg = err.get("message") or str(err)
                    code = err.get("code") or err.get("type") or err.get("status")
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status_code} error: {msg} (code={code})"
                    )
            raise RuntimeError(
                f"OpenRouter HTTP {response.status_code} error: {json.dumps(payload, ensure_ascii=False)}"
            )

        response_data = response.json()

        # If the provider reports an error inside a 200 OK payload, surface it clearly
        def _extract_provider_error(payload: dict[str, Any]) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None
            # Top-level error
            if payload.get("error"):
                return payload.get("error")  # type: ignore[return-value]
            # Error inside choices[0]
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict) and first.get("error"):
                    return first.get("error")  # type: ignore[return-value]
            return None

        provider_error = _extract_provider_error(response_data)
        if provider_error is not None:
            # Collect helpful fields
            if isinstance(provider_error, dict):
                msg = provider_error.get("message") or str(provider_error)
                code = provider_error.get("code") or provider_error.get("type")
                metadata = (
                    provider_error.get("metadata")
                    if isinstance(provider_error.get("metadata"), dict)
                    else None
                )
                provider_name = (
                    metadata.get("provider_name") if isinstance(metadata, dict) else None
                )
                raw = metadata.get("raw") if isinstance(metadata, dict) else None
                raw_code = raw.get("code") if isinstance(raw, dict) else None
                raw_msg = raw.get("message") if isinstance(raw, dict) else None

                details_parts: list[str] = []
                if provider_name:
                    details_parts.append(f"provider={provider_name}")
                if code:
                    details_parts.append(f"code={code}")
                if raw_code:
                    details_parts.append(f"raw_code={raw_code}")
                if raw_msg:
                    details_parts.append(f"raw_msg={raw_msg}")
                details = f" ({', '.join(details_parts)})" if details_parts else ""
                raise RuntimeError(f"OpenRouter provider error: {msg}{details}")
            else:
                raise RuntimeError(f"OpenRouter provider error: {provider_error}")

        # ------------------------------------------------------
        # Cost accounting
        # ------------------------------------------------------
        cost = self.request_price
        usage = response_data.get("usage")
        if isinstance(usage, dict):
            try:
                prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = float(usage.get("completion_tokens", 0) or 0)
                cost = (
                    prompt_tokens * self.input_token_price
                    + completion_tokens * self.output_token_price
                    + self.request_price
                )
            except Exception:
                # Fallback to request price only if usage is malformed
                cost = self.request_price
        self.add_cost(cost)

        # ------------------------------------------------------
        # Return full response for tool calls, otherwise process content
        # ------------------------------------------------------
        if tools is not None:
            # When tools are provided, return the full response so caller can access tool_calls
            return response_data, cost

        # Safely extract content; empty string is allowed and will be validated below
        content = (
            (response_data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if isinstance(response_data.get("choices"), list) and response_data.get("choices")
            else ""
        )

        if pydantic_model is not None:
            if not self.supports_structured_output:
                # Extract JSON from content that may have additional text
                json_content = extract_json(
                    content, fields=list(pydantic_model.model_fields.keys())
                )
                if json_content is None:
                    preview = (
                        content
                        if isinstance(content, str)
                        else json.dumps(content, ensure_ascii=False)
                    )
                    if len(preview) > 800:
                        preview = preview[:800] + "... [truncated]"
                    raise ValueError(
                        f"No JSON content found in the response. Content preview: {preview}"
                    )
                content = json_content

            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            try:
                content = pydantic_model.model_validate_json(content)
            except Exception as e:
                preview = (
                    content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                )
                if isinstance(preview, str) and len(preview) > 800:
                    preview = preview[:800] + "... [truncated]"
                error_type = e.__class__.__name__
                raise ValueError(
                    f"Error validating structured JSON ({error_type}): {str(e)}\nContent preview: {preview}"
                ) from e

        return content, cost
