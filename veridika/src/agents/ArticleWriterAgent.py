from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.agents.langs import langid2lang
from veridika.src.llm import LLM
from veridika.src.managers.cication_manager import CitationManager


class ArticleWriterAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the ArticleWriterAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="ArticleWriterAgent",
            description="An agent that generates fact-check articles using retrieved documents with proper citations",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def _article_prompt(
        self, fact_checking_topic: str, data_block: str, question_block: str, lang: str = "es"
    ) -> str:
        """Generate the article writing prompt."""
        return f"""
You are tasked with generating a fact check for a given statement or question. You will be provided with critical questions and text sources to assist you in this task. Follow these instructions carefully:

1. Here is the statement or question to be fact-checked:
{fact_checking_topic}

2. Here you have a set of critical questions to guide your fact check, you should answer them based on the information provided in the text sources:
{question_block}

3. Here are the text sources you should use as your source of information:
{data_block}

4. Generate a fact check based on the provided text sources. 
    - Use the critical questions to guide your analysis. When referencing information from the text sources, cite them using the format [1], [2], etc., corresponding to their ID numbers.
    - You MUST include EVERY argument and piece of evidence from the text chunks in your fact check. Do not be selective or summarize just a few points - your analysis should comprehensively gather and present ALL relevant information from the provided sources.
    - It is CRUCIAL that you actively search for BOTH supporting AND refuting evidence in the text sources. Do not favor one perspective over another.
    - For each point you make, try to find if there are contrasting viewpoints or contradictory data present in the sources.
    - Ensure that your fact check is based solely on the information provided in the text sources. If you cannot find relevant information in the text sources to address any part of the statement or critical questions, clearly state that there is insufficient information to make a determination on that specific point. 
    - Use as much detail as possible to support your fact check. 
    - Your task is to present a balanced, comprehensive analysis of the evidence, not to persuade toward any particular conclusion.
    - Use **markdown bold** to highlight the more important arguments. 

5. Structure your fact check in exactly three paragraphs. Paragraphs should only be separated by a blank line. Do not add titles to the paragraphs.
    - First paragraph: You MUST thoroughly describe ALL arguments and evidence that support the statement or answer the question affirmatively. Make a deliberate effort to find these supporting points even if they are not immediately obvious. If after exhaustive searching you cannot find arguments in favor within the provided text fragments, you must explicitly explain that you have not found any.
    - Second paragraph: You MUST thoroughly describe ALL arguments and evidence that refute the statement or answer the question negatively. Make a deliberate effort to find these counterpoints even if they are not immediately obvious. If after exhaustive searching you cannot find arguments against within the provided text fragments, you must explicitly explain that you have not found any.
    - Third paragraph: You must explain your conclusion regarding the truthfulness of the statement or question. Acknowledge the strength of evidence on both sides. When the evidence is mixed, clearly explain the nuances rather than oversimplifying.

6. In your citations and analysis, refer to the text sources by their ID numbers (e.g., [1], [2]). Ensure that you only use information from the provided text sources.

7. End your fact-check with a clear decision on whether the statement is true, false, or undetermined based on the available evidence. If possible, avoid leaving the conclusion as "undetermined." Make an effort to categorize the statement as true or false. If you cannot find evidence to verify the statement, conclude that it is false. If you cannot find evidence to disprove the statement, conclude that it is true. Only say the truth cannot be determined if you find conflicting evidence or strongly opposing viewpoints about the statement.

Very important: 
    - Use **markdown bold** to highlight the more important arguments and key pieces of evidence. This will help draw attention to the most significant points in your fact check.
    - If you include any technical terms or concepts, make sure to explain them clearly.
    - If you include any acronyms, make sure to expand them the first time you use them.
    - The user cannot see the text source or the question. Therefore, make sure your answer is self-contained and provides all the necessary information.
    - Do make explicit references to the text sources in your answer, the user will not have access to them. So do not say "As mentioned in the first text sources provided..." but rather "There is evidence that...".
    - Your paragraphs can be as long as necessary to provide a comprehensive fact check, do not worry about the length.
    - If no supporting or contradicting evidence is found, state that clearly. If not enough evidence is found to answer the questions above, state that clearly.
    - ALWAYS present multiple perspectives whenever they exist in the source material, even if one viewpoint appears more dominant.

Very important: Your answer should be in {langid2lang[lang]}, even if some of the input chucks are in other languages. 
""".strip()

    def run(
        self,
        search_results: dict[str, list[dict]],
        fact_checking_topic: str,
        lang: str = "es",
        citation_manager: CitationManager | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, str]], float, HistoryEntry]:
        """Generate a fact-check article using the provided search results.

        Args:
            search_results (Dict[str, List[Dict]]): Unified format results from WebSearch/RAG agents
            fact_checking_topic (str): The statement or question to fact-check
            lang (str): Language code for the response (default: "es")
            citation_manager (Optional[CitationManager]): Shared citation manager for consistent numbering

        Returns:
            Tuple[Dict[str, str], Dict[str, Dict[str, str]], float, HistoryEntry]: A tuple containing:
                - Dictionary with article: {"article": "fact-check text"}
                - Dictionary with citation metadata: {"1": {"source": "...", "url": "...", "favicon": "..."}, ...}
                - The cost of the LLM operation
                - History entry containing operation metadata
        """
        # Use provided citation manager or create new one
        if citation_manager is None:
            citation_manager = CitationManager()

        # Extract questions from search results keys
        # Handle both combined queries (WebSearch: "q1 | q2 | q3") and individual queries (RAG: "q1")
        questions = []
        for key in search_results:
            if "|" in key:
                # Split combined queries and strip whitespace
                questions.extend([q.strip() for q in key.split("|")])
            else:
                # Individual query
                questions.append(key)

        # Prepare data block with citations using unified results
        data_block = citation_manager.prepare_from_unified_results(search_results)
        question_block = "\n".join(questions)

        # Get total number of texts for metadata
        total_texts = sum(len(documents) for documents in search_results.values())

        # Create conversation for LLM
        conversation = [
            {
                "role": "system",
                "content": self._article_prompt(
                    fact_checking_topic=fact_checking_topic,
                    data_block=data_block,
                    question_block=question_block,
                    lang=lang,
                ),
            },
            {"role": "user", "content": fact_checking_topic},
        ]

        # Call LLM
        response, cost = self.model(messages=conversation)
        conversation.append({"role": "assistant", "content": response})

        # Extract citation IDs from the response and get citation metadata
        cited_ids = citation_manager.retrieve_ids_from_text(response)

        # Create result dictionary
        result = {"article": response}

        # Create history entry
        history_entry = HistoryEntry(
            data=conversation,
            type=HistoryEntryType.conversation,
            model_metadata={
                "model_name": self.model.api_name,
                "agent_type": "article_writer",
                "fact_checking_topic": fact_checking_topic,
                "questions": questions,
                "language": lang,
                "citations_used": cited_ids,
                "total_sources": total_texts,
            },
        )

        return result, cost, history_entry
