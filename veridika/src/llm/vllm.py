import json
import logging
import time

from openai import OpenAI
from pydantic import BaseModel

from veridika.src.api import ApiHandler, get_api_key
from veridika.src.llm.utils import extract_json


class VLLM(ApiHandler):
    def __init__(
        self,
        model: str,
        vllm_url_env_name: str,
        api_key_env_name: str,
        pretty_name: str | None = None,
    ):
        self.model = model
        try:
            vllm_url = get_api_key(vllm_url_env_name)
        except KeyError:
            raise ValueError(f"VLLM URL environment variable {vllm_url_env_name} not found")

        try:
            api_key = get_api_key(api_key_env_name)
        except KeyError:
            raise ValueError(f"VLLM API key environment variable {api_key_env_name} not found")

        # print(f"Using VLLM URL: {vllm_url}")

        self.client = OpenAI(base_url=vllm_url, api_key=api_key)
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self, messages: list[dict], pydantic_model: type[BaseModel] | None = None
    ) -> tuple[BaseModel | str, float]:
        """
        Call the OpenAI API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            temperature (Optional[int]): The temperature to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        # ------------------------------------------------------
        # Retry logic similar to OpenRouter implementation
        # ------------------------------------------------------
        max_retries = 3
        base_delay = 1

        # Build base request payload
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            # "extra_body": {"repetition_penalty": 1.1},
        }

        # If a Pydantic model is provided, request server-side JSON schema enforcement
        # using OpenAI-compatible response_format with a strict json_schema.
        if pydantic_model is not None:
            try:
                schema = pydantic_model.model_json_schema()
            except Exception:
                schema = None
            if schema:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": pydantic_model.__name__,
                        "schema": schema,
                    },
                    "strict": True,
                }

        # Try with response_format first (if present), then fall back to a plain call with retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                break
            except Exception as e:  # Broad catch: SDK may raise various transport/API exceptions
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"vLLM chat.completions attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        cost = 0.0  # Local vLLM server: cost accounting not applicable

        self.add_cost(cost)

        content = response.choices[0].message.content

        # ------------------------------------------------------
        # Structured output handling using Pydantic (schema-validated)
        # ------------------------------------------------------
        if pydantic_model is not None:
            # 1) Prefer direct validation assuming server enforced the schema
            try:
                validated: BaseModel = pydantic_model.model_validate_json(content)
                return validated, cost
            except Exception:
                # 2) Fallback: best-effort JSON extraction then validate
                json_content = extract_json(
                    content, fields=list(pydantic_model.model_fields.keys())
                )
                if json_content is None:
                    raise ValueError(f"No JSON content found in the response: {content}")

                if isinstance(json_content, dict):
                    json_text = json.dumps(json_content, ensure_ascii=False)
                else:
                    json_text = str(json_content)

                try:
                    validated = pydantic_model.model_validate_json(json_text)
                except Exception as e:
                    raise ValueError(f"Error validating JSON: {json_text}\n{e}")

                return validated, cost

        return content, cost
