from collections.abc import Sequence
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import torch.nn.functional as F

from veridika.src.embeddings import Embeddings


class RAG:
    """Retriever-Augmented Generation helper.

    Stores a growing corpus of passages, embeds them lazily, and answers
    similarity queries.  Optimisations included:

    * Concurrent embedding of new documents and queries in a single model pass.
    * Chunked tensor storage to avoid *O(N²)* concatenation overhead.
    * Optional fp16 storage of document vectors.
    * One-time L2-normalisation for cosine similarity via simple dot product.

    Args:
        embedding_model: Dual-encoder model that accepts ``queries`` and
            ``passages`` lists and returns (query_emb, doc_emb) tensors.
        chunk_size: Character window for the recursive splitter.
        chunk_overlap: Overlap in characters between consecutive chunks.
        split_separators: Sequence of separators fed to
            :class:`RecursiveCharacterTextSplitter`.
        fp16_storage: If ``True``, keep document embeddings in ``float16`` to
            halve memory usage.  Queries remain ``float32``.
        l2_normalise: Perform L2-normalisation on embeddings so that cosine
            similarity equals dot product.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        split_separators: Sequence[str] = ("\n",),
        fp16_storage: bool = True,
        l2_normalise: bool = True,
    ):
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

        self.model = embedding_model
        self.cost = 0.0

        # Embedding storage — list of chunks plus a cached flat tensor.
        self._chunks: list[torch.Tensor] = []
        self._flat: torch.Tensor | None = None
        self._dirty = False  # concat cache invalidation flag
        self.fp16_storage = fp16_storage
        self.l2_normalise = l2_normalise
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_separators = split_separators

        # Splitter setup
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=list(split_separators),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
        )

    def add_documents(self, documents: list[dict[str, str]]):
        """Add raw documents to the internal corpus.

        Each document is expected to have ``text``, ``url``, ``favicon`` and
        ``source`` keys.  Long documents are split into manageable chunks and
        filtered by length.

        Args:
            documents: List of dicts with the required keys.
        """
        if not documents:
            return

        base_meta = [
            {"url": d["url"], "favicon": d["favicon"], "source": d["source"]} for d in documents
        ]
        texts = [d["text"] for d in documents]

        # Track initial counts for debugging
        initial_doc_count = len(self.documents)

        for text, meta in zip(texts, base_meta, strict=False):
            if text is None or base_meta is None:
                print(f"Text or base_meta is None: {text} {base_meta}")

        splits = self.text_splitter.create_documents(texts=texts, metadatas=base_meta)
        pre_filter_count = len(splits)

        splits = [
            s for s in splits if len(s.page_content.split()) > 20 and len(s.page_content) < 5_000
        ]
        post_filter_count = len(splits)

        if not splits:
            # Debug: log when all documents are filtered out
            logging.warning(
                f"Warning: All {pre_filter_count} document chunks were filtered out during processing"
            )
            return

        docs, metas = zip(*((s.page_content, s.metadata) for s in splits), strict=False)
        self.documents.extend(docs)
        self.metadatas.extend(metas)

        # Validate consistency after adding documents
        final_doc_count = len(self.documents)
        final_meta_count = len(self.metadatas)

        if final_doc_count != final_meta_count:
            raise ValueError(
                f"Document/metadata count mismatch after adding documents: "
                f"documents={final_doc_count}, metadatas={final_meta_count}"
            )

        # Debug logging
        added_count = final_doc_count - initial_doc_count
        logging.info(
            f"Added {added_count} document chunks ({pre_filter_count} -> {post_filter_count} after filtering)"
        )

    @torch.inference_mode()
    def _embed_queries_and_new_docs(self, queries: list[str]) -> torch.Tensor:
        """Embed *queries* and yet-unembedded docs in one forward pass with enhanced error recovery.

        Args:
            queries: List of user questions to embed.

        Returns:
            Tuple of (query_embeddings, cost) where query_embeddings is a tensor
            of shape ``[len(queries), dim]`` and cost is the embedding cost.
        """
        start = sum(chunk.size(0) for chunk in self._chunks)
        new_docs: list[str] = self.documents[start:]

        # Enhanced retry logic for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # One model call; ``passages`` can be None or empty when no new docs.
                (q_emb, d_emb), cost = self.model(queries=queries, passages=new_docs or None)
                self.cost += cost

                # Process and store document vectors if any.
                if new_docs:
                    # Validate that we got the expected number of embeddings
                    if d_emb is None:
                        raise ValueError(
                            f"Embedding model returned None for document embeddings when {len(new_docs)} documents were provided"
                        )

                    expected_embeddings = len(new_docs)
                    actual_embeddings = d_emb.size(0)

                    if actual_embeddings != expected_embeddings:
                        raise ValueError(
                            f"Embedding mismatch: expected {expected_embeddings} document embeddings "
                            f"but got {actual_embeddings}. This could be due to API rate limiting or content filtering."
                        )

                    if self.l2_normalise:
                        d_emb = F.normalize(d_emb, dim=1)
                    if self.fp16_storage:
                        d_emb = d_emb.to(torch.float16)
                    self._chunks.append(d_emb.cpu())
                    self._dirty = True

                # Always normalise queries in fp32 on active device.
                if self.l2_normalise:
                    q_emb = F.normalize(q_emb, dim=1)
                if self.fp16_storage:
                    q_emb = q_emb.to(torch.float16)
                return q_emb, cost

            except Exception as e:
                if attempt < max_retries:
                    # Check if this looks like a rate limiting error
                    error_msg = str(e).lower()
                    is_rate_limit = any(
                        keyword in error_msg
                        for keyword in [
                            "rate limit",
                            "too many requests",
                            "quota",
                            "throttle",
                            "embedding mismatch",
                            "api",
                        ]
                    )

                    if is_rate_limit:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        logging.warning(
                            f"Rate limiting detected (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}"
                        )
                        import time

                        time.sleep(delay)
                        continue

                # If not rate limiting or max retries exceeded, re-raise
                logging.error(f"Embedding failed after {attempt + 1} attempts: {e}")
                # Clean up inconsistent state - remove documents that couldn't be embedded
                if new_docs:
                    # Remove the documents that couldn't be embedded
                    remove_count = len(new_docs)
                    self.documents = self.documents[:-remove_count]
                    self.metadatas = self.metadatas[:-remove_count]
                    logging.warning(
                        f"Removed {remove_count} documents that couldn't be embedded to maintain consistency"
                    )

                raise e

        # This should never be reached
        raise RuntimeError("Unexpected end of retry loop")

    def _all_doc_embeddings(self) -> torch.Tensor:
        """Concatenate chunk list into one tensor if cache is stale.

        Returns:
            ``[num_docs, dim]`` tensor in ``float32`` on CPU.
        """
        if self._dirty or self._flat is None:
            if not self._chunks:
                raise ValueError("No embedding chunks available but _all_doc_embeddings was called")

            self._flat = torch.cat(self._chunks, dim=0).to(
                torch.float16 if self.fp16_storage else torch.float32
            )
            self._dirty = False

            # Validate that the concatenated embeddings match the number of stored documents
            num_embeddings = self._flat.size(0)
            num_documents = len(self.documents)
            num_metadatas = len(self.metadatas)

            if num_embeddings != num_documents:
                raise ValueError(
                    f"Embedding/document count mismatch after concatenation: "
                    f"embeddings={num_embeddings}, documents={num_documents}, metadatas={num_metadatas}. "
                    f"Chunk sizes: {[chunk.size(0) for chunk in self._chunks]}"
                )

        return self._flat

    @torch.inference_mode()
    def query(
        self,
        queries: list[str],
        top_k: int = 5,
        get_scores: bool = False,
    ) -> dict[str, list[dict]]:
        """Retrieve top-k most similar passages for each query string.

        Args:
            queries: List of query strings.
            top_k: Number of passages to return per query.
            get_scores: If ``True``, include similarity scores in output.

        Returns:
            Mapping ``{query: [hits...]}``, where each *hit* is a dict with
            ``text``, ``metadata`` and optionally ``score``.
        """
        if not self.documents:
            return {q: [] for q in queries}, 0.0

        q_emb, cost = self._embed_queries_and_new_docs(queries)
        doc_emb = self._all_doc_embeddings().to(q_emb.device)

        # Validate consistency between embeddings and metadata
        num_embeddings = doc_emb.size(0)
        num_documents = len(self.documents)
        num_metadatas = len(self.metadatas)

        if num_embeddings != num_documents or num_embeddings != num_metadatas:
            raise ValueError(
                f"Mismatch detected in RAG storage: "
                f"embeddings={num_embeddings}, documents={num_documents}, metadatas={num_metadatas}. "
                f"This indicates a synchronization issue between document storage and embedding storage."
            )

        sim = q_emb @ doc_emb.T
        top_k = min(top_k, doc_emb.size(0))

        if get_scores:
            sims, idxs = torch.sort(sim, dim=1, descending=True)
        else:
            sims, idxs = torch.topk(sim, k=top_k, dim=1)

        results: dict[str, list[dict]] = {}
        for q, row_i, row_s in zip(queries, idxs, sims, strict=False):
            hits = []
            for i, s in zip(row_i, row_s, strict=False):
                idx = i.item()
                # Additional bounds check for safety
                if idx >= len(self.metadatas) or idx >= len(self.documents):
                    logging.warning(
                        f"Warning: Index {idx} out of bounds (metadatas={len(self.metadatas)}, documents={len(self.documents)}). Skipping."
                    )
                    continue

                hits.append(
                    {
                        "metadata": self.metadatas[idx],
                        "text": self.documents[idx],
                        "score": s.item() if get_scores else None,
                    }
                )
            results[q] = hits[:top_k]
        return results, cost

    def __call__(
        self,
        documents: list[dict[str, str]],
        queries: list[str],
        top_k: int = 5,
        get_scores: bool = False,
    ) -> dict[str, list[dict]]:
        """Combined ingest + retrieve helper.

        Args:
            documents: New documents to add before answering.
            queries: List of query strings.
            top_k: Number of passages per query.
            get_scores: Whether to include similarity scores.

        Returns:
            Same structure as :meth:`query`.
        """
        self.add_documents(documents)
        return self.query(queries, top_k=top_k, get_scores=get_scores)

    def reset(self):
        """Reset the RAG system state, clearing all stored documents and embeddings.

        This is useful when the same RAG instance needs to be reused for
        completely different sets of documents (e.g., between evaluation examples).

        Args:
            None

        Returns:
            None
        """
        if self.documents or self._chunks:
            logging.debug(
                f"Resetting RAG state: clearing {len(self.documents)} documents and {len(self._chunks)} embedding chunks"
            )

        self.documents.clear()
        self.metadatas.clear()
        self._chunks.clear()
        self._flat = None
        self._dirty = False

        # Reset cost tracking for this instance
        self.cost = 0.0

        logging.debug("RAG state reset complete")
