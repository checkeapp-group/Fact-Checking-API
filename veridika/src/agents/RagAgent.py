from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.embeddings import Embeddings
from veridika.src.rag.rag import RAG


class RagAgent(BaseAgent):
    def __init__(
        self,
        embedding_model: Embeddings,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        split_separators: tuple = ("\n",),
        fp16_storage: bool = True,
        l2_normalise: bool = True,
        max_history_size: int | None = None,
    ):
        """Initialize the RagAgent.

        Args:
            embedding_model (Embeddings): The embedding model to use for document and query embeddings
            chunk_size (int): Character window for the recursive splitter (default: 128)
            chunk_overlap (int): Overlap in characters between consecutive chunks (default: 0)
            split_separators (tuple): Sequence of separators for text splitting (default: ("\n",))
            fp16_storage (bool): Use float16 for document embeddings to save memory (default: True)
            l2_normalise (bool): Perform L2-normalisation on embeddings (default: True)
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="RagAgent",
            description="An agent that performs retrieval-augmented generation using document embeddings and similarity search",
            max_history_size=max_history_size,
        )
        self.rag = RAG(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_separators=split_separators,
            fp16_storage=fp16_storage,
            l2_normalise=l2_normalise,
        )

    def run(
        self,
        websearch_results: dict[str, list[dict]],
        queries: list[str],
        top_k: int = 5,
        get_scores: bool = False,
    ) -> tuple[dict[str, list[dict]], float, HistoryEntry]:
        """Execute RAG retrieval using documents from WebSearch results.

        Args:
            websearch_results (Dict[str, List[Dict]]): Unified format results from WebSearchAgent.
                Expected format: {"query1 | query2": [{"text": "...", "metadata": {...}}, ...]}
            queries (List[str]): List of query strings to search for in the RAG corpus
            top_k (int): Number of most similar passages to return per query (default: 5)
            get_scores (bool): Whether to include similarity scores in history (default: False)

        Returns:
            Tuple[Dict[str, List[Dict]], float, HistoryEntry]: A tuple containing:
                - Dictionary with unified format: {"query1": [documents...], "query2": [documents...]}
                  Each document has:
                  {
                      "text": "passage content...",
                      "metadata": {
                          "url": "https://example.com",
                          "source": "example.com",
                          "favicon": "https://example.com/favicon.ico",
                          "type": "rag"
                      }
                  }
                - The cost of the embedding operations
                - History entry containing RAG operation metadata
        """
        # Extract documents from websearch results and convert to RAG format
        documents = []
        for query_key, docs in websearch_results.items():
            for doc in docs:
                rag_doc = {
                    "text": doc["text"],
                    "url": doc["metadata"]["url"],
                    "source": doc["metadata"]["source"],
                    "favicon": doc["metadata"]["favicon"],
                }
                documents.append(rag_doc)

        # Execute RAG retrieval (this also adds documents to the corpus)
        rag_results, cost = self.rag(
            documents=documents,
            queries=queries,
            top_k=top_k,
            get_scores=get_scores,
        )

        # Transform to unified format (remove scores from return)
        unified_results = {}
        for query, passages in rag_results.items():
            unified_results[query] = [
                {
                    "text": passage["text"],
                    "metadata": {
                        "url": passage["metadata"]["url"],
                        "source": passage["metadata"]["source"],
                        "favicon": passage["metadata"]["favicon"],
                        "type": "rag",
                    },
                }
                for passage in passages
            ]

        # Format results for history: {query1: [{document1: score1}, ...]}
        history_results = {}
        for query, passages in rag_results.items():
            history_results[query] = [
                {passage["text"]: passage.get("score", 0.0)} for passage in passages
            ]

        # Create history entry
        history_entry = HistoryEntry(
            data=history_results,
            type=HistoryEntryType.rag,
            model_metadata={
                "embedding_model": self.rag.model.__class__.__name__,
                "fp16_storage": self.rag.fp16_storage,
                "l2_normalise": self.rag.l2_normalise,
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "documents_added": len(documents),
                "top_k": top_k,
                "get_scores": get_scores,
            },
        )

        return unified_results, cost, history_entry

    def add_documents_only(
        self, documents: list[dict[str, str]]
    ) -> tuple[int, float, HistoryEntry]:
        """Add documents to the RAG corpus without performing queries.

        Args:
            documents (List[Dict[str, str]]): List of documents to add to the corpus.
                Each document should have 'text', 'url', 'favicon', and 'source' keys.

        Returns:
            Tuple[int, float, HistoryEntry]: A tuple containing:
                - Number of documents added
                - The cost of the embedding operations
                - History entry containing the operation metadata
        """
        initial_doc_count = len(self.rag.documents)

        # Add documents to the corpus
        self.rag.add_documents(documents)

        final_doc_count = len(self.rag.documents)
        documents_added = final_doc_count - initial_doc_count

        # Get cost from the embedding model
        cost = self.rag.model.cost if hasattr(self.rag.model, "cost") else 0.0

        # Create history entry
        history_entry = HistoryEntry(
            data={
                "operation": "add_documents_only",
                "documents_provided": len(documents),
                "documents_added": documents_added,
                "total_documents_in_corpus": final_doc_count,
                "chunk_size": self.rag.text_splitter.chunk_size,
                "chunk_overlap": self.rag.text_splitter.chunk_overlap,
            },
            type=HistoryEntryType.rag,
            model_metadata={
                "embedding_model": self.rag.model.__class__.__name__,
                "fp16_storage": self.rag.fp16_storage,
                "l2_normalise": self.rag.l2_normalise,
            },
        )

        return documents_added, cost, history_entry

    def query_only(
        self,
        queries: list[str],
        top_k: int = 5,
        get_scores: bool = False,
    ) -> tuple[dict[str, list[dict]], float, HistoryEntry]:
        """Query the existing RAG corpus without adding new documents.

        Args:
            queries (List[str]): List of query strings to search for
            top_k (int): Number of most similar passages to return per query (default: 5)
            get_scores (bool): Whether to include similarity scores in results (default: False)

        Returns:
            Tuple[Dict[str, List[Dict]], float, HistoryEntry]: A tuple containing:
                - Dictionary mapping queries to lists of similar passages
                - The cost of the embedding operations
                - History entry containing the operation metadata
        """
        # Query the existing corpus
        results, cost = self.rag.query(
            queries=queries,
            top_k=top_k,
            get_scores=get_scores,
        )

        # Format results for history: {query1: [{document1: score1}, ...]}
        history_results = {}
        for query, passages in results.items():
            history_results[query] = [
                {passage["text"]: passage.get("score", 0.0)} for passage in passages
            ]

        # Create history entry
        history_entry = HistoryEntry(
            data=history_results,
            type=HistoryEntryType.rag,
            model_metadata={
                "embedding_model": self.rag.model.__class__.__name__,
                "fp16_storage": self.rag.fp16_storage,
                "l2_normalise": self.rag.l2_normalise,
                "operation": "query_only",
                "top_k": top_k,
                "get_scores": get_scores,
                "total_documents_in_corpus": len(self.rag.documents),
            },
        )

        return results, cost, history_entry

    def reset(self):
        """Reset the RAG agent state.

        This ensures each fact-checking run starts with a clean RAG state.
        """
        self.rag.reset()
        super().reset()
