from __future__ import annotations

from google import genai
from google.genai import types
import torch

from veridika.src.api import ApiHandler, get_api_key
from veridika.src.embeddings.utils import batch  # same helper you already use

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
_PRICE_BATCH = 0.00
_TASK_TYPE = "FACT_VERIFICATION"


class GeminiEmbeddings(ApiHandler):
    """
    Google Gemini embedding wrapper with explicit batching.

    Args:
        model: Gemini embedding model ID. Defaults to
            ``"gemini-embedding-exp-03-07"``.
        pretty_name: Alias used for cost tracking. Defaults to *model*.
    """

    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client(api_key=get_api_key("GEMINI_API_KEY"))
        super().__init__(model)

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        """Send one *embed_content* call and return raw vectors."""
        cfg = types.EmbedContentConfig(task_type=_TASK_TYPE)
        resp = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=cfg,
        )
        embeddings = [emb.values for emb in resp.embeddings]

        # Validate that we got the expected number of embeddings
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Gemini API returned {len(embeddings)} embeddings for {len(texts)} input texts. "
                f"This could be due to content filtering or API issues. "
                f"Input texts: {[text[:50] + '...' if len(text) > 50 else text for text in texts]}"
            )

        return embeddings

    def _embed_list(
        self,
        texts: list[str],
        batch_size: int,
        desc: str,
    ) -> tuple[torch.Tensor, float]:
        """
        Embed *texts* in batches.

        Args:
            texts: List of strings to embed.
            batch_size: Max texts per API call.
            desc: Label for the (optional) progress bar.

        Returns:
            (embeddings, cost) where *embeddings* is a CPU Tensor of shape
            ``(len(texts), dim)`` and *cost* is the USD charge.
        """
        embeddings: list[list[float]] = []
        cost = 0.0

        for chunk in batch(texts, batch_size):
            embeddings.extend(self._embed_chunk(chunk))

        return torch.tensor(embeddings), cost

    def __call__(
        self,
        *,
        queries: list[str] | None = None,
        passages: list[str] | None = None,
        batch_size: int = 100,
        **_,
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], float]:
        """
        Compute embeddings for *queries* and/or *passages*.

        Args:
            queries: List of query strings or ``None``.
            passages: List of passage strings or ``None``.
            batch_size: Maximum number of texts per Gemini request.

        Returns:
            Tuple ``((query_embeddings, passage_embeddings), cost)`` where
            *cost* is the total USD charge.
        """
        if queries is None and passages is None:
            raise ValueError("Either *queries* or *passages* must be provided.")

        total_cost = 0.0
        q_emb, p_emb = None, None

        if queries is not None:
            q_emb, c = self._embed_list(queries, batch_size, "Queries")
            total_cost += c

        if passages is not None:
            p_emb, c = self._embed_list(passages, batch_size, "Passages")
            total_cost += c

        self.add_cost(total_cost)
        return (q_emb, p_emb), total_cost
