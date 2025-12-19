from .openai import OpenAIEmbeddings


class VLLMEmbeddings(OpenAIEmbeddings):
    def __init__(
        self,
        model: str,
        vllm_url_env_name: str,
        api_key_env_name: str,
        pretty_name: str | None = None,
    ):
        print(f"Using VLLM Embedding URL: {vllm_url_env_name}")
        super().__init__(
            model=model, vllm_url_env_name=vllm_url_env_name, api_key_env_name=api_key_env_name
        )

    def _get_pricing(self, model: str) -> float:
        """
        Get the pricing for a given model.
        """
        return 0.0
