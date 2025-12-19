import json

from pydantic import create_model

from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.agents.langs import langid2lang
from veridika.src.llm import LLM
from veridika.src.managers.cication_manager import CitationManager


class QuestionAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the QuestionAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="QuestionAgent",
            description="An agent that answers questions using retrieved documents with proper citations",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def _qa_answer_prompt(self, questions: str, data_block: str, lang: str = "es") -> str:
        """Generate the question answering prompt."""
        return f"""
You will be given a series of text sources and questions. Your task is to answer the question based on the information provided in the text sources, using appropriate citations.
First, you will receive the text sources in the following format:
{data_block}

You should answer the following questions based on the information in the text sources:
{questions}

To complete this task, follow these steps:

1. Carefully read and analyze all the text sources provided.
2. Identify the sources that contain information relevant to answering the question.
3. Formulate an answer to the questions using the information from the relevant sources.
4. When using information from a specific source, cite it using the format [X], where X is the source number.
5. If multiple sources support a statement, include all relevant citations, e.g., [1] [3] [5].
6. Ensure that your answer is in the same language as the question.
7. Present your answers in a clear and concise manner, using complete sentences.
8. The user can only see the question and the answer, not the text sources. Therefore, make sure your answer is self-contained and provides all the necessary information. Do not make any direct references to the text sources in your answer.

Output format: Return a json dictionary, with the questions as keys and answers as values. 
{{
    "¿Question 1?":"Answer 1", 
    "¿Question 2?":"Answer 2"
}}


Remember to use the information provided in the text sources and cite your sources appropriately. Do not include any information that is not present in the given text sources.

Your answer should be in {langid2lang[lang]}.
""".strip()

    def run(
        self,
        search_results: dict[str, list[dict]],
        lang: str = "es",
        citation_manager: CitationManager | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, str]], float, HistoryEntry]:
        """Answer questions using the provided search results.

        Args:
            search_results (Dict[str, List[Dict]]): Unified format results from WebSearch/RAG agents
            lang (str): Language code for the response (default: "es")
            citation_manager (Optional[CitationManager]): Shared citation manager for consistent numbering

        Returns:
            Tuple[Dict[str, str], Dict[str, Dict[str, str]], float, HistoryEntry]: A tuple containing:
                - Dictionary with question:answer pairs
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

        # Get total number of texts for metadata
        total_texts = sum(len(documents) for documents in search_results.values())

        # Format questions as string
        questions_block = "\n".join(questions)

        # Create dynamic Pydantic model for structured output
        fields = dict.fromkeys(questions, (str, ...))  # ... means required field
        QuestionAnswers = create_model("QuestionAnswers", **fields)

        # Create conversation for LLM
        conversation = [
            {
                "role": "system",
                "content": self._qa_answer_prompt(
                    questions=questions_block, data_block=data_block, lang=lang
                ),
            },
            {"role": "user", "content": questions_block},
        ]

        # Call LLM with structured output
        response, cost = self.model(messages=conversation, pydantic_model=QuestionAnswers)

        # Convert Pydantic model to dict
        qa_pairs = response.model_dump()
        conversation.append(
            {"role": "assistant", "content": json.dumps(qa_pairs, ensure_ascii=False)}
        )

        # Extract citation IDs from all answers and get citation metadata
        all_answers_text = " ".join(qa_pairs.values()) if qa_pairs else response
        cited_ids = citation_manager.retrieve_ids_from_text(all_answers_text)

        # Create history entry
        history_entry = HistoryEntry(
            data=conversation,
            type=HistoryEntryType.conversation,
            model_metadata={
                "model_name": self.model.api_name,
                "agent_type": "question_answerer",
                "questions": questions,
                "answers": qa_pairs,
                "language": lang,
                "citations_used": cited_ids,
                "total_sources": total_texts,
                "questions_count": len(questions),
            },
        )

        return qa_pairs, cost, history_entry
