import datetime
import json
from typing import Any

from pydantic import BaseModel

from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.agents.langs import langid2lang
from veridika.src.llm import LLM


class ModelOutput(BaseModel):
    plan: str
    questions: list[str]


class RefinementModelOutput(BaseModel):
    analysis: str
    questions: list[str]


class CriticalQuestionAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the CriticalQuestionAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="CriticalQuestionAgent",
            description="A research strategist agent that generates actionable research questions and plans for fact-checking statements through web searches and data gathering",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def _get_prompt(self, langid: str) -> str:
        """Generate the system prompt for the agent based on language ID.

        Args:
            langid (str): Language identifier code

        Returns:
            str: The formatted system prompt string
        """
        lang = langid2lang[langid]
        return f"""
You are a research strategist specialized in fact-checking. Your task is to analyze a statement and generate a research plan with specific information-gathering questions that can be used to verify or challenge the claim through web searches and data collection.

**Your Mission:**
1. **Create a research plan** - Outline concrete steps to gather factual information about the statement
2. **Generate research questions** - Develop specific questions that can be answered through web searches, studies, reports, and data analysis

**Research Plan Guidelines:**
Your plan should include steps to:
- Gather current factual data and evidence relevant to the claim
- Find comparative analysis, studies, or data from multiple sources
- Look for expert opinions, official statements, and authoritative evaluations
- Search for contradictory evidence, alternative perspectives, or opposing viewpoints
- Verify timing, context, and circumstances surrounding the claim
- Check for recent developments, updates, or changes that might affect the claim's validity
- Identify primary sources, official records, or direct documentation
- Cross-reference information across multiple independent sources

**Research Question Guidelines:**
Generate 4-6 specific research questions that:
- **Seek factual information** - "What are the official data/statistics about...?", "What evidence exists regarding...?"
- **Request comparisons** - "How does X compare to Y in terms of...?", "What are the differences between...?"
- **Look for expert opinions** - "What do experts/authorities say about...?", "What do official sources conclude about...?"
- **Find contradictory evidence** - "What evidence contradicts...?", "What are the main criticisms or opposing views of...?"
- **Check recent developments** - "What are the latest updates on...?", "How has the situation regarding X changed in {datetime.datetime.now().year}?"
- **Verify context** - "Under what circumstances does...?", "What is the historical/political/social context of...?"

**Quality Standards:**
- Questions must be searchable on Google or research databases
- Questions must be self-contained and include the specific subject/entity from the original statement
- Avoid pronouns (like "esa persona", "it", "that", "this") - always mention the specific subject by name
- Example: Instead of "Which is the official stance from the president about him?" write "Which is the official stance from the [President name / President of x contry...] about [Person name]?"
- Focus on objective, measurable aspects when possible
- Include both supportive and challenging perspectives
- Be specific about timeframes (use current year {datetime.datetime.now().year} when relevant)
- Do not add formating to the questions, such as numbers or other symbols.
- Avoid meta-questions about the statement itself - focus on gathering information

**Output Format:**
Respond with JSON containing:
- `"plan"`: A markdown research plan with concrete information-gathering steps
- `"questions"`: Array of specific research questions suitable for web searches

**Context:** Current year is {datetime.datetime.now().year}
**Language:** Respond entirely in {lang}
""".strip()

    def _get_conversation(
        self, statement: str, langid: str, override_prompt: str | None = None
    ) -> str:
        """Build the conversation structure for the LLM.

        Args:
            statement (str): The statement to analyze
            langid (str): Language identifier code
            override_prompt (Optional[str]): Custom prompt to use instead of default

        Returns:
            str: List of conversation messages formatted for the LLM
        """
        return [
            {
                "role": "system",
                "content": self._get_prompt(langid) if override_prompt is None else override_prompt,
            },
            {"role": "user", "content": statement},
        ]

    def run(
        self,
        statement: str,
        langid: str,
        override_prompt: str | None = None,
        override_pydantic_model: BaseModel | None = None,
    ) -> tuple[Any, float, HistoryEntry]:
        """Execute the agent to generate critical questions for analyzing a given statement.

        Args:
            statement (str): The statement to analyze and generate critical questions for
            langid (str): Language identifier code
            override_prompt (Optional[str]): Custom prompt to use instead of default
            override_pydantic_model (Optional[BaseModel]): Custom Pydantic model for response parsing

        Returns:
            Tuple[Any, float, HistoryEntry]: A tuple containing:
                - The parsed model response (plan and critical questions)
                - The cost of the API call
                - History entry containing conversation data and metadata
        """
        conversation = self._get_conversation(
            statement=statement, langid=langid, override_prompt=override_prompt
        )
        response, cost = self.model(
            messages=conversation,
            pydantic_model=ModelOutput
            if override_pydantic_model is None
            else override_pydantic_model,
        )

        response_dict = response.model_dump()

        # Validate the response has the required fields
        if "questions" not in response_dict:
            raise ValueError(
                f"CriticalQuestionAgent: Missing 'questions' field in model response: {response_dict}"
            )

        if "plan" not in response_dict:
            raise ValueError(
                f"CriticalQuestionAgent: Missing 'plan' field in model response: {response_dict}"
            )

        # Validate questions
        questions = response_dict["questions"]
        if not isinstance(questions, list):
            raise ValueError(
                f"CriticalQuestionAgent: 'questions' field is not a list: {type(questions)}"
            )

        if not questions:
            raise ValueError(
                f"CriticalQuestionAgent: Empty questions list generated for statement: '{statement}'"
            )

        # Filter out empty or invalid questions
        valid_questions = [q for q in questions if isinstance(q, str) and q.strip()]
        if not valid_questions:
            raise ValueError(
                f"CriticalQuestionAgent: No valid question strings found in: {questions}"
            )

        response_dict["questions"] = valid_questions

        # Validate plan
        plan = response_dict["plan"]
        if not isinstance(plan, str):
            raise ValueError(f"CriticalQuestionAgent: 'plan' field is not a string: {type(plan)}")

        if not plan.strip():
            raise ValueError(
                f"CriticalQuestionAgent: Empty plan generated for statement: '{statement}'"
            )

        conversation.append(
            {"role": "assistant", "content": json.dumps(response_dict, ensure_ascii=False)}
        )
        return (
            response_dict,
            cost,
            HistoryEntry(
                data=conversation,
                type=HistoryEntryType.conversation,
                model_metadata={"model_name": self.model.api_name},
            ),
        )


class CriticalQuestionRefinementAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the CriticalQuestionRefinementAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="CriticalQuestionRefinementAgent",
            description="A specialized agent that refines and improves critical questions based on user feedback and additional requirements",
            max_history_size=max_history_size,
        )
        self.model = LLM(model=model_name)

    def _get_prompt(self, langid: str) -> str:
        """Generate the system prompt for the refinement agent based on language ID.

        Args:
            langid (str): Language identifier code

        Returns:
            str: The formatted system prompt string
        """
        lang = langid2lang[langid]
        return f"""
You are a research question refinement specialist. Your task is to analyze existing critical questions and user feedback. Based on the user feedback you will update the list of critical questions. 

**INSTRUCTIONS**: 

Based on the user feedback, there two actions you can take:
1. **Rewrite the existing questions**: You will only do this in the case in which the user feedbacks includes a clarification that invalidates the current questions. For example, given the statement "Is it true that the president is corrupt?" and a set of questions about the president of the United States, if the user feedbacks includes a clarification that he wanted to ask about the president of Germany, you will need to rewrite the questions to refer to the president of Germany. You will only perform this action if there is a clear clarification that invalidates the current questions. Your answer will be a new list of questions.
2. **Add new questions**: In cases in which the user feedback includes a request for extra information that is not covered by the current questions. For example, given the statement "Are electric cars cheaper to run than gasoline cars?", if the user feedbacks is "I also want you to search information about the predictions for the future prices of gasoline", you will need to add a new question at the end of the existing list. Your answer will be a list including the existing questions **without any changes** copy them word by word, and the new questions at the end of the list. 

Unless there is a clear clarification that invalidates the current questions, you will try to preserve the already existing questions without any modifications. Do not rewrite or add any question unless the user feedback explicitly asks for it. Your task is to incorporate the user feedback into the existing questions, not to rewrite or improve them on your own.

**Research Question Guidelines:**
- **Seek factual information** - "What are the official data/statistics about...?", "What evidence exists regarding...?"
- **Request comparisons** - "How does X compare to Y in terms of...?", "What are the differences between...?"
- **Look for expert opinions** - "What do experts/authorities say about...?", "What do official sources conclude about...?"
- **Find contradictory evidence** - "What evidence contradicts...?", "What are the main criticisms or opposing views of...?"
- **Check recent developments** - "What are the latest updates on...?", "How has the situation regarding X changed in {datetime.datetime.now().year}?"
- **Verify context** - "Under what circumstances does...?", "What is the historical/political/social context of...?"

**Quality Standards:**
- Questions must be searchable on Google or research databases
- Questions must be self-contained and include the specific subject/entity from the original statement
- Avoid pronouns (like "esa persona", "it", "that", "this") - always mention the specific subject by name
- Example: Instead of "Which is the official stance from the president about him?" write "Which is the official stance from the [President name / President of x contry...] about [Person name]?"
- Focus on objective, measurable aspects when possible
- Include both supportive and challenging perspectives
- Be specific about timeframes (use current year {datetime.datetime.now().year} when relevant)
- Do not add formating to the questions, such as numbers or other symbols.
- Avoid meta-questions about the statement itself - focus on gathering information

**Output Format:**
Respond with JSON containing:
- `"analysis"`: A brief explanation of how you interpreted the user's feedback and what changes you made
- `"questions"`: Array of research questions (keeping existing ones + any additions/modifications as requested)

**Example Output for "add information about Chinese opinion":**
```json
{{
  "analysis": "User requested adding information about Chinese citizen opinion. I kept all 6 existing questions unchanged and added 1 new question about Chinese public opinion surveys.",
  "questions": [
    "Question 1 (original)",
    "Question 2 (original)", 
    "Question 3 (original)",
    "Question 4 (original)",
    "Question 5 (original)",
    "Question 6 (original)",
    "¿Qué muestran las encuestas de opinión pública en China sobre...?"
  ]
}}
```

**Context:** Current year is {datetime.datetime.now().year}
**Language:** Respond entirely in {lang}
""".strip()

    def _get_conversation(
        self,
        statement: str,
        langid: str,
        current_questions: list[str],
        user_feedback: str,
        override_prompt: str | None = None,
    ) -> list[dict]:
        """Build the conversation structure for the LLM.

        Args:
            statement (str): The original statement being fact-checked
            langid (str): Language identifier code
            current_questions (List[str]): The current list of critical questions
            user_feedback (str): User's feedback and refinement requests
            override_prompt (Optional[str]): Custom prompt to use instead of default

        Returns:
            List[dict]: List of conversation messages formatted for the LLM
        """
        current_questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(current_questions)])

        user_message = f"""**Original Statement:** {statement}

**Current Critical Questions:**
{current_questions_text}

**User Feedback and Refinement Request:**
{user_feedback}
""".strip()

        return [
            {
                "role": "system",
                "content": self._get_prompt(langid) if override_prompt is None else override_prompt,
            },
            {"role": "user", "content": user_message},
        ]

    def run(
        self,
        statement: str,
        langid: str,
        current_questions: list[str],
        user_feedback: str,
        override_prompt: str | None = None,
        override_pydantic_model: BaseModel | None = None,
    ) -> tuple[Any, float, HistoryEntry]:
        """Execute the agent to refine critical questions based on user feedback.

        Args:
            statement (str): The original statement being fact-checked
            langid (str): Language identifier code
            current_questions (List[str]): The current list of critical questions
            user_feedback (str): User's feedback and refinement requests
            override_prompt (Optional[str]): Custom prompt to use instead of default
            override_pydantic_model (Optional[BaseModel]): Custom Pydantic model for response parsing

        Returns:
            Tuple[Any, float, HistoryEntry]: A tuple containing:
                - The parsed model response (analysis and refined questions)
                - The cost of the API call
                - History entry containing conversation data and metadata
        """
        conversation = self._get_conversation(
            statement=statement,
            langid=langid,
            current_questions=current_questions,
            user_feedback=user_feedback,
            override_prompt=override_prompt,
        )

        response, cost = self.model(
            messages=conversation,
            pydantic_model=RefinementModelOutput
            if override_pydantic_model is None
            else override_pydantic_model,
        )

        response_dict = response.model_dump()

        # Validate the response has the required fields
        if "questions" not in response_dict:
            raise ValueError(
                f"CriticalQuestionRefinementAgent: Missing 'questions' field in model response: {response_dict}"
            )

        if "analysis" not in response_dict:
            raise ValueError(
                f"CriticalQuestionRefinementAgent: Missing 'analysis' field in model response: {response_dict}"
            )

        # Validate refined questions
        refined_questions = response_dict["questions"]
        if not isinstance(refined_questions, list):
            raise ValueError(
                f"CriticalQuestionRefinementAgent: 'questions' field is not a list: {type(refined_questions)}"
            )

        if not refined_questions:
            raise ValueError(
                f"CriticalQuestionRefinementAgent: Empty questions list generated for statement: '{statement}'"
            )

        # Filter out empty or invalid questions
        valid_questions = [q for q in refined_questions if isinstance(q, str) and q.strip()]
        if not valid_questions:
            raise ValueError(
                f"CriticalQuestionRefinementAgent: No valid question strings found in: {refined_questions}"
            )

        response_dict["questions"] = valid_questions

        # Validate analysis
        analysis = response_dict["analysis"]
        if not isinstance(analysis, str):
            raise ValueError(
                f"CriticalQuestionRefinementAgent: 'analysis' field is not a string: {type(analysis)}"
            )

        if not analysis.strip():
            raise ValueError(
                f"CriticalQuestionRefinementAgent: Empty analysis generated for statement: '{statement}'"
            )

        conversation.append(
            {"role": "assistant", "content": json.dumps(response_dict, ensure_ascii=False)}
        )
        return (
            response_dict,
            cost,
            HistoryEntry(
                data=conversation,
                type=HistoryEntryType.conversation,
                model_metadata={"model_name": self.model.api_name},
            ),
        )
