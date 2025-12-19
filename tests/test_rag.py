import unittest

from dotenv import load_dotenv

from veridika.src.embeddings import Embeddings
from veridika.src.rag import RAG

load_dotenv()


class RAGTestCase(unittest.TestCase):
    """Unit-tests for the optimised :class:`RAG` helper."""

    @classmethod
    def setUpClass(cls):
        """Initialise the (costly) embedding backbone once for the suite."""
        cls.embedder = Embeddings("text-embedding-3-small")
        cls.total_cost = 0.0

    def _sample_docs(self) -> list[dict[str, str]]:
        """Return two small HTML-like docs for ingestion."""
        return [
            {
                "url": "https://example.com/a",
                "favicon": "https://example.com/favicon.ico",
                "source": "unit-test",
                "text": "This is a short document about machine learning. " * 10,
            },
            {
                "url": "https://example.com/b",
                "favicon": "https://example.com/favicon.ico",
                "source": "unit-test",
                "text": "Another document discussing deep learning and AI. " * 10,
            },
        ]

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost RAG tests: ${cls.total_cost:.9f}")

    def test_add_documents(self):
        rag = RAG(self.embedder)
        rag.add_documents(self._sample_docs())

        # should have equal numbers of docs and metadata
        self.assertGreater(len(rag.documents), 0)
        self.assertEqual(len(rag.documents), len(rag.metadatas))

    def test_query_returns_hits_and_scores(self):
        rag = RAG(self.embedder)
        rag.add_documents(self._sample_docs())

        query = ["What is machine learning?"]
        out, cost = rag.query(query, top_k=2, get_scores=True)

        # key present
        self.assertIn(query[0], out)

        # correct hit structure
        hits = out[query[0]]
        self.assertEqual(len(hits), 2)
        for h in hits:
            self.assertIn("text", h)
            self.assertIn("metadata", h)
            self.assertIn("score", h)
            self.assertIsInstance(h["score"], float)

        RAGTestCase.total_cost += cost

    def test_query_twice_without_new_docs(self):
        """Second call should **not** raise early‑return bug."""
        rag = RAG(self.embedder)
        rag.add_documents(self._sample_docs())
        total_cost = 0.0
        _, cost = rag.query(["Explain AI"], top_k=1)
        total_cost += cost
        second, cost = rag.query(["Explain AI again"], top_k=1)
        total_cost += cost
        RAGTestCase.total_cost += total_cost
        self.assertIn("Explain AI again", second)

    def test_top_k_exceeds_corpus(self):
        """Requesting more hits than available should degrade gracefully."""
        rag = RAG(self.embedder)
        rag.add_documents(self._sample_docs())

        res, cost = rag.query(["Deep learning"], top_k=10)
        returned_hits = len(res["Deep learning"])
        self.assertLessEqual(returned_hits, len(rag.documents))
        RAGTestCase.total_cost += cost


if __name__ == "__main__":
    unittest.main()
