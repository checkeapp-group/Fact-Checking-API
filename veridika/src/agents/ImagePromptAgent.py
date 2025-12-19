from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.llm import LLM


class ImagePromptAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the ImagePromptAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="ImagePromptAgent",
            description="An agent that generates image descriptions for article titles using LLM",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def image_description_prompt(self, fact_checking: str) -> str:
        """Generate the image description prompt."""
        return f"""
Given the following article title, I want to generate a header image using an image-generating AI. Could you generate a suitable image description for the article?
The description should be short but concise. Respond only with the image description; do not add anything else to your response. Answer in English.
Article Title: {fact_checking}
""".strip()

    def run(
        self,
        fact_checking_title: str,
    ) -> tuple[dict[str, str], float, HistoryEntry]:
        """Generate an image description for the given article title.

        Args:
            fact_checking_title (str): The article title to generate an image description for
            lang (str): Language code (default: "es")

        Returns:
            Tuple[Dict[str, str], float, HistoryEntry]: A tuple containing:
                - Dictionary with image description: {"image_description": "description text"}
                - The cost of the LLM operation
                - History entry containing operation metadata
        """
        # Create conversation for LLM
        conversation = [
            {
                "role": "system",
                "content": self.image_description_prompt(
                    fact_checking=fact_checking_title,
                ),
            },
            {"role": "user", "content": fact_checking_title},
        ]

        # Call LLM
        response, cost = self.model(messages=conversation)
        conversation.append({"role": "assistant", "content": response})

        # Create result dictionary
        result = {"image_description": response}

        # Create history entry
        history_entry = HistoryEntry(
            data=conversation,
            type=HistoryEntryType.conversation,
            model_metadata={
                "model_name": self.model.api_name,
                "agent_type": "image_prompt_generator",
                "fact_checking_title": fact_checking_title,
                "generated_description": response,
            },
        )

        return result, cost, history_entry
