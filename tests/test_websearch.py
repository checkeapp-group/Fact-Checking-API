import unittest

from dotenv import load_dotenv

load_dotenv()


class TestWebSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_cost = 0.0

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost WebSearch tests: ${cls.total_cost:.4f}")

    def _test_websearch(self, websearch):
        response, cost = websearch(
            queries=["Roman Empire"],
            top_k=1,
            language="en",
            location="us",
            download_text=True,
        )
        response = response[0]
        TestWebSearch.total_cost += cost
        self.assertTrue(response)
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 1.0)
        print(
            f"Serper (Roman Empire - EN/US): {response.keys()}. Title: {response['title']}. Text: {response['text'][:50]}..."
        )

        response, cost = websearch(
            queries=["Roman Empire"],
            top_k=1,
            language="es",
            location="es",
            download_text=True,
        )
        response = response[0]
        TestWebSearch.total_cost += cost
        self.assertTrue(response)
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 1.0)
        print(
            f"Serper (Roman Empire - ES/ES): {response.keys()}. Title: {response['title']}. Text: {response['text'][:50]}..."
        )

    def test_serper(self):
        from veridika.src.web_search import WebSearch

        serper = WebSearch("serper")
        self._test_websearch(serper)
