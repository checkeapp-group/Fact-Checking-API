import unittest

from dotenv import load_dotenv

load_dotenv()


class TestReplicate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_cost = 0.0

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost Flux tests: ${cls.total_cost:.4f}")

    def test_flux(self):
        from veridika.src.image import Image

        flux = Image("flux")

        response, cost = flux(
            "A painting of a cat",
            size="1024x1024",
        )

        TestReplicate.total_cost += cost

        self.assertTrue(response)
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 1.0)
        print(f"Flux (Cat painting): {response}")
