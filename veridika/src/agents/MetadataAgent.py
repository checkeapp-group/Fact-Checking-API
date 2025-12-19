import json
from typing import Literal

from pydantic import BaseModel

from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.agents.langs import langid2lang
from veridika.src.llm import LLM


class Metadata(BaseModel):
    title: str
    categories: list[str]
    label: Literal["Fake", "True", "Undetermined"]
    main_claim: str


class MetadataAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the MetadataAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="MetadataAgent",
            description="An agent that extracts structured metadata from fact-check articles",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def get_metadata_prompt(
        self, fact_checking_title: str, fact_checking: str, lang: str = "es"
    ) -> str:
        """Generate the metadata extraction prompt."""
        if lang == "es":
            example = """
{
"title": "¿Son los coches eléctricos más contaminantes que los coches de gasolina?",
"categories": ["Automoción", "Medio ambiente", "Ciencia"],
"main_claim": "Los coches eléctricos no son más contaminantes que los coches de gasolina",
"label": "Fake"
}
            """.strip()

        else:
            example = """        
{
"title": "Are electric cars more polluting than gasoline cars?",
"categories": ["Automotive", "Environment", "Science"],
"main_claim": "Electric cars are not more polluting than gasoline cars",
"label": "Fake"
}
            """.strip()

        return f"""
Given a fact-checking article, I want you to generate a JSON with the following information:
1. Article title: The article title corrected for spelling and grammar. Keep the original title if there are no errors.
2. List of categories: For example, News, Sports, Entertainment, Science, Technology, Environment, Automotive, Politics, etc... Maximum 3 categories.
3. Label: If the conclusion is "Fake", "True", or "Undetermined", extract it from the last paragraph of the article.
4. Main claim: Based on the tag, rewrite the article headline. The headline should be a statement that includes the main topic of the article and the fact-checking conclusion. Make it concise and short.

Example:
{example}

Article title: {fact_checking_title}
Article: {fact_checking}

Your answer should be in {langid2lang[lang]}.
        """.strip()

    def run(
        self,
        article_output: dict[str, str],
        fact_checking_title: str,
        lang: str = "es",
    ) -> tuple[dict[str, any], float, HistoryEntry]:
        """Extract structured metadata from a fact-check article.

        Args:
            article_output (Dict[str, str]): Output from ArticleWriterAgent containing {"article": "fact-check text"}
            fact_checking_title (str): The original title/topic of the fact-check
            lang (str): Language code for the response (default: "es")

        Returns:
            Tuple[Dict[str, any], float, HistoryEntry]: A tuple containing:
                - Dictionary with metadata: {"title": "...", "categories": [...], "label": "...", "main_claim": "..."}
                - The cost of the LLM operation
                - History entry containing operation metadata
        """
        # Extract article text
        fact_checking_text = article_output.get("article", "")

        # Create conversation for LLM
        conversation = [
            {
                "role": "system",
                "content": self.get_metadata_prompt(
                    fact_checking_title=fact_checking_title,
                    fact_checking=fact_checking_text,
                    lang=lang,
                ),
            },
            {
                "role": "user",
                "content": f"Article title: {fact_checking_title}\nArticle: {fact_checking_text}",
            },
        ]

        # Call LLM with structured output
        response, cost = self.model(messages=conversation, pydantic_model=Metadata)

        # Convert Pydantic model to dict
        metadata_dict = response.model_dump()
        conversation.append(
            {"role": "assistant", "content": json.dumps(metadata_dict, ensure_ascii=False)}
        )

        # Create history entry
        history_entry = HistoryEntry(
            data=conversation,
            type=HistoryEntryType.conversation,
            model_metadata={
                "model_name": self.model.api_name,
                "agent_type": "metadata_extractor",
                "fact_checking_title": fact_checking_title,
                "language": lang,
                "extracted_metadata": metadata_dict,
            },
        )

        return metadata_dict, cost, history_entry
