import unittest

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class TestLLMs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_cost = 0.0

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost OpenRouter tests: ${cls.total_cost:.4f}")

    def _test_model(self, model: str):
        from veridika.src.llm import LLM

        llm = LLM(model)
        output, cost = llm(
            messages=[{"role": "user", "content": "What's the weather like in Spain?"}]
        )
        TestLLMs.total_cost += cost

        print(f"{model} (${cost}): {output[:100].replace('\n', ' ')}")
        self.assertIsNotNone(output)
        self.assertIsInstance(output, str)
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, -0.001)
        self.assertLess(cost, 1.00)

    def test_openai(self):
        models = ["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano"]
        for model in models:
            self._test_model(model)

    def test_anthropic(self):
        models = ["anthropic/claude-sonnet-4"]
        for model in models:
            self._test_model(model)

    def test_gemini(self):
        models = ["google/gemini-2.5-flash", "google/gemini-2.5-pro"]
        for model in models:
            self._test_model(model)

    def test_mistral(self):
        models = ["mistralai/mistral-small-3.2-24b-instruct"]
        for model in models:
            self._test_model(model)

    def test_deepseek(self):
        models = ["deepseek/deepseek-r1-0528"]
        for model in models:
            self._test_model(model)


class TestLLM_StructuredOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.total_cost = 0.0

    @classmethod
    def tearDownClass(cls):
        print(f"Total cost StructuredOutput tests: ${cls.total_cost:.4f}")

    def _test_model(self, model: str, enable_complex: bool = False):
        from veridika.src.llm import LLM

        class Weather(BaseModel):
            location: str = Field(..., description="City or location name")
            average_temperature: float = Field(..., description="Average temperature in °C")

        class ListOfWeather(BaseModel):
            weather: list[Weather]

        llm = LLM(model)
        output, cost = llm(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "What's the average temperature in Spain? "
                        "Return the answer in the following JSON format: "
                        '{"location": "Spain", "average_temperature": 15.5}'
                    ),
                }
            ],
            pydantic_model=Weather,
        )
        TestLLM_StructuredOutput.total_cost += cost

        print(f"{model} (${cost}): {output.model_dump_json()}")
        self.assertIsInstance(output, Weather)

        if enable_complex:
            output, cost = llm(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "What's the average temperature in Spain? "
                            "Return the answer in the following JSON format: "
                            '{"weather": [{"location": "Madrid", "average_temperature": 15.5}]}'
                        ),
                    }
                ],
                pydantic_model=ListOfWeather,
            )
            TestLLM_StructuredOutput.total_cost += cost
            print(f"{model} (${cost}): {output.model_dump_json()}")
            self.assertIsInstance(output, ListOfWeather)

    def test_openai(self):
        models = ["openai/gpt-4.1-mini"]
        for model in models:
            self._test_model(model, enable_complex=True)

    def test_gemini(self):
        models = ["google/gemini-2.0-flash-001"]
        for model in models:
            self._test_model(model)

    def test_unsupported_models(self):
        models = ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4"]
        for model in models:
            self._test_model(model, enable_complex=True)


if __name__ == "__main__":
    unittest.main()
