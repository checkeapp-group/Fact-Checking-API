import unittest

from dotenv import load_dotenv
import torch

load_dotenv()


class TestEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_cost = 0.0

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost OpenRouter tests: ${cls.total_cost:.4f}")

    @torch.inference_mode()
    def _test_embeddings(self, model):
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        (questions_embeddings_small, documents_embeddings_small), cost = model(
            queries=questions, passages=documents
        )
        TestEmbeddings.total_cost += cost

        norms: torch.Tensor = torch.sqrt(torch.sum(questions_embeddings_small**2, axis=1))
        norms[norms == 0] = 1
        na: torch.Tensor = questions_embeddings_small / norms[:, torch.newaxis]

        norms: torch.Tensor = torch.sqrt(torch.sum(documents_embeddings_small**2, axis=1))
        norms[norms == 0] = 1
        nb: torch.Tensor = documents_embeddings_small / norms[:, torch.newaxis]

        sim = na @ nb.T

        answer = torch.argmax(sim, dim=1).tolist()

        self.assertTrue(answer == [0, 1])

    def test_openai(self):
        from veridika.src.embeddings import Embeddings

        embeddings = Embeddings("text-embedding-3-small")
        self._test_embeddings(embeddings)

        embeddings = Embeddings("text-embedding-3-large")

    def test_gemini(self):
        from veridika.src.embeddings import Embeddings

        embeddings = Embeddings("gemini-embedding-001")
        self._test_embeddings(embeddings)
