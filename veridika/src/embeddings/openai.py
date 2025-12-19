from __future__ import annotations

from collections.abc import Iterable
import concurrent.futures as cf
from itertools import islice
import logging
import time

from openai import OpenAI, OpenAIError
import torch

from veridika.src.api import ApiHandler, get_api_key


def _chunks(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    """Yield successive chunks from *iterable* of length ≤ *size*."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


# ────────────────────────────────────────────────
# Embeddings wrapper
# ────────────────────────────────────────────────
class OpenAIEmbeddings(ApiHandler):
    """
    OpenAI embedding endpoint with batching, retry logic and cost accounting.
    """

    # class-level tuning knobs
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # seconds; multiplied by attempt index
    THREADS = 2  # queries + passages processed in parallel

    def _get_pricing(self, model: str) -> float:
        """
        Get the pricing for a given model.
        """
        if model == "text-embedding-3-small":
            return 0.02 / 1_000_000
        elif model == "text-embedding-3-large":
            return 0.13 / 1_000_000
        else:
            raise ValueError(f"Invalid model: {model}")

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        api_key_env_name: str = "OPENAI_API_KEY",
        vllm_url_env_name: str | None = None,
    ) -> None:
        """
        Args:
            model: OpenAI embedding model id (e.g. ``"text-embedding-3-small"``).
            device: Torch device for the output tensors (``"cpu"``, ``"cuda"``, …).
        """
        self.model = model
        self.device = device
        try:
            if vllm_url_env_name is not None:
                vllm_url = get_api_key(vllm_url_env_name)
            else:
                vllm_url = None
        except KeyError:
            raise ValueError(f"Embedding URL environment variable {vllm_url_env_name} not found")

        try:
            api_key = get_api_key(api_key_env_name)
        except KeyError:
            raise ValueError(f"Embedding API key environment variable {api_key_env_name} not found")

        self.client = OpenAI(api_key=api_key, base_url=vllm_url)
        super().__init__(model)

    # ----------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """
        Embed a single batch with retries.

        Args:
            texts: A list of texts whose total length ≤ ``BATCH_SIZE``.

        Returns:
            Tuple where the first element is a list of embedding vectors and
            the second element is the number of tokens used.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = self.client.embeddings.create(input=texts, model=self.model)
                embeddings = [item.embedding for item in resp.data]

                # Validate that we got the expected number of embeddings
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"OpenAI API returned {len(embeddings)} embeddings for {len(texts)} input texts. "
                        f"This could be due to content filtering or API issues. "
                        f"Input texts: {[text[:50] + '...' if len(text) > 50 else text for text in texts]}"
                    )

                return embeddings, resp.usage.total_tokens
            except (OpenAIError, OSError) as exc:
                last_exc = exc
                wait = self.RETRY_BACKOFF * attempt
                logging.warning(
                    f"[{self.model}] batch failed (attempt {attempt}/{self.MAX_RETRIES}) "
                    f"— retrying in {wait:.1f}s\n→ {exc}"
                )
                time.sleep(wait)

        # All retries exhausted
        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts") from last_exc

    def _embed_texts(
        self, texts: list[str], *, desc: str, batch_size: int
    ) -> tuple[torch.Tensor, float]:
        """
        Embed an arbitrary list of texts.

        Args:
            texts: List of strings to embed.
            desc: Description used only for logging / tracing.

        Returns:
            Tuple where the first element is a **tensor** of shape
            ``(len(texts), dim)`` on ``self.device`` and the second element is
            the cost in USD.
        """
        if not texts:
            return torch.empty(0, 0, device=self.device), 0.0

        embeddings: list[list[float]] = []
        total_tokens = 0
        chunk_idx = -1

        for chunk_idx, chunk in enumerate(_chunks(texts, batch_size)):
            batch_vecs, tokens = self._embed_batch(chunk)
            embeddings.extend(batch_vecs)
            total_tokens += tokens

        cost = total_tokens * self._get_pricing(self.model)
        tensor = torch.tensor(embeddings, device=self.device)

        # Final validation: ensure tensor size matches input size
        if tensor.size(0) != len(texts):
            raise ValueError(
                f"OpenAI embedding size mismatch: expected {len(texts)} embeddings, got {tensor.size(0)}. "
                f"This indicates content filtering or API issues. Total batches processed: {chunk_idx + 1}. "
                f"Batch size used: {batch_size}."
            )

        return tensor, cost

    # ----------------------------------------------------------
    # public call
    # ----------------------------------------------------------
    def __call__(
        self,
        *,
        queries: list[str] | None = None,
        passages: list[str] | None = None,
        batch_size: int | None = 256,  # preserved for API compatibility
        **_,
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], float]:
        """
        Compute embeddings for *queries* and/or *passages*.

        Args:
            queries: List of query strings. ``None`` if not required.
            passages: List of passage strings. ``None`` if not required.
            batch_size: Ignored parameter (kept for backward compatibility).

        Returns:
            A tuple ``((query_embeddings, passage_embeddings), total_cost)`` where
            *query_embeddings* and *passage_embeddings* are Torch tensors (or
            ``None`` if the corresponding input list was ``None``) and
            *total_cost* is the price in USD.
        """
        if queries is None and passages is None:
            raise ValueError("Either *queries* or *passages* must be provided.")

        tasks = {}
        with cf.ThreadPoolExecutor(max_workers=self.THREADS) as pool:
            if queries is not None:
                tasks["queries"] = pool.submit(
                    self._embed_texts, queries, desc="queries", batch_size=batch_size
                )
            if passages is not None:
                tasks["passages"] = pool.submit(
                    self._embed_texts, passages, desc="passages", batch_size=batch_size
                )

        q_emb, p_emb, cost = None, None, 0.0
        for key, fut in tasks.items():
            tensor, part_cost = fut.result()
            cost += part_cost
            if key == "queries":
                q_emb = tensor
            else:
                p_emb = tensor

        self.add_cost(cost)
        return (q_emb, p_emb), cost
