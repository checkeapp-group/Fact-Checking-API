import datetime
import json
import re
from typing import Any

from pydantic import BaseModel

from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.agents.langs import langid2lang
from veridika.src.llm import LLM


def clean_search_query(query: str) -> str:
    """Clean a search query by removing quotes and site: operators.

    Args:
        query (str): The raw search query to clean

    Returns:
        str: The cleaned search query without quotes or site: operators
    """
    if not isinstance(query, str):
        return query

    # Remove site: operators (e.g., site:example.com or site:subdomain.example.com)
    # Match site: followed by domain (with or without subdomain)
    cleaned = re.sub(r"\bsite:\S+\s*", "", query)

    # Remove all double quotes
    cleaned = cleaned.replace('"', "")

    # Remove all single quotes (but be careful with apostrophes in words)
    # Only remove quotes that are likely to be search operators (at word boundaries)
    cleaned = re.sub(r"(?<!\w)'|'(?!\w)", "", cleaned)

    # Clean up multiple spaces that might result from removals
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


class ModelOutput(BaseModel):
    searches: list[str]


class RefinementModelOutput(BaseModel):
    analysis: str
    searches: list[str]


class GenSearchesAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the GenSearchesAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="GenSearchesAgent",
            description="A agent that generates questions and web searches to gather information about a given statement",
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
You are a search query strategist specialized in fact-checking research. Your task is to generate targeted Google search queries that will efficiently retrieve the information needed to answer critical research questions about a statement.

**Your Mission:**
You will receive:
1. A statement that needs fact-checking
2. A set of critical research questions designed to verify or challenge the statement

Your goal is to create 3-5 strategic Google search queries that will gather the most relevant information to answer these questions comprehensively.

**Search Strategy Guidelines:**

**1. Optimize for Information Retrieval:**
- Create searches that target primary sources, official data, and authoritative content
- Use specific keywords that are likely to return high-quality, credible results
- Target specific organizations, institutions, or authoritative sources when relevant

**2. Search Query Best Practices:**
- Use specific entity names, dates, and locations from the original statement
- Combine multiple concepts in a single search when they're closely related
- Prioritize searches that can answer multiple questions simultaneously
- Each search should retrieve different web pages and different information, ensure no duplicated web searches.
- **NEVER use quotes around phrases** - Write natural search queries without quotation marks
- **NEVER use site: operators** - Do not restrict searches to specific websites (e.g., avoid site:example.com)
- Keep searches simple and natural, as if typed directly into Google by a regular user

**3. Coverage and Efficiency:**
- Ensure your searches collectively address all the critical questions
- Avoid redundant searches that would return similar information
- Prioritize searches by importance - start with the most crucial information needed
- Balance broad searches (for context) with specific searches (for detailed data)

**4. Quality and Credibility Focus:**
- Design searches to find peer-reviewed studies, official reports, and expert analyses
- Include terms that help filter for recent, relevant information (add year {datetime.datetime.now().year} when current data is needed)
- Target comparative analyses and multiple perspectives on controversial topics
- Look for both supporting and contradicting evidence

**Example Search Strategies:**
- For claims about statistics: official [topic] statistics {datetime.datetime.now().year} government data
- For controversial statements: [topic] expert analysis research studies + [topic] criticism opposing views
- For recent events: [specific event] {datetime.datetime.now().year} official statement news
- For comparisons: [entity A] vs [entity B] comparison study data

**Output Format:**
Respond with JSON containing:
- `"searches"`: Array of 3-5 strategic Google search queries

**Example Output:**
```json
{{
  "searches": [
    "official COVID-19 vaccine effectiveness statistics 2024 CDC WHO data",
    "COVID-19 vaccine side effects peer-reviewed studies clinical trials",
    "vaccine hesitancy expert analysis public health research",
    "COVID-19 vaccine safety monitoring systems VAERS Yellow Card"
  ]
}}
```

**Quality Standards:**
- Each search should be actionable and likely to return relevant, credible results
- Searches should complement each other without significant overlap
- Use specific terminology that experts and official sources would use
- Include temporal qualifiers when recent information is crucial
- Focus on factual, verifiable information rather than opinions

**Context:** Current year is {datetime.datetime.now().year}
**Language:** All searches should be in {lang}
""".strip()

    def _get_conversation(
        self,
        statement: str,
        questions: list[str],
        langid: str,
        override_prompt: str | None = None,
    ) -> str:
        """Build the conversation structure for the LLM.

        Args:
            statement (str): The statement to analyze
            langid (str): Language identifier code
            override_prompt (Optional[str]): Custom prompt to use instead of default

        Returns:
            str: List of conversation messages formatted for the LLM
        """

        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        user_message = f"""**Statement to Fact-Check:**
{statement}

**Critical Research Questions:**
{questions_text}

**Task:** Generate 3-5 strategic Google search queries that will efficiently gather the information needed to answer these research questions. Focus on finding authoritative sources, official data, and credible evidence that can verify or challenge the statement."""

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
        questions: list[str],
        langid: str,
        override_prompt: str | None = None,
        override_pydantic_model: BaseModel | None = None,
    ) -> tuple[Any, float, HistoryEntry]:
        """Execute the agent to generate questions and searches for a given statement.

        Args:
            questions (List[str]): The list of questions to generate searches for
            langid (str): Language identifier code
            override_prompt (Optional[str]): Custom prompt to use instead of default
            override_pydantic_model (Optional[BaseModel]): Custom Pydantic model for response parsing

        Returns:
            Tuple[Any, float, HistoryEntry]: A tuple containing:
                - The parsed model response (searches)
                - The cost of the API call
                - History entry containing conversation data and metadata
        """
        conversation = self._get_conversation(
            statement=statement, questions=questions, langid=langid, override_prompt=override_prompt
        )
        response, cost = self.model(
            messages=conversation,
            pydantic_model=ModelOutput
            if override_pydantic_model is None
            else override_pydantic_model,
        )

        response_dict = response.model_dump()

        # Validate the response has the required fields
        if "searches" not in response_dict:
            raise ValueError(
                f"GenSearchesAgent: Missing 'searches' field in model response: {response_dict}"
            )

        # Validate searches
        searches = response_dict["searches"]
        if not isinstance(searches, list):
            raise ValueError(f"GenSearchesAgent: 'searches' field is not a list: {type(searches)}")

        if not searches:
            raise ValueError(
                f"GenSearchesAgent: Empty searches list generated for statement: '{statement}'"
            )

        # Filter out empty or invalid searches
        valid_searches = [s for s in searches if isinstance(s, str) and s.strip()]
        if not valid_searches:
            raise ValueError(f"GenSearchesAgent: No valid search strings found in: {searches}")

        # Clean searches to remove quotes and site: operators
        cleaned_searches = [clean_search_query(s) for s in valid_searches]

        # Filter out any searches that became empty after cleaning
        cleaned_searches = [s for s in cleaned_searches if s]
        if not cleaned_searches:
            raise ValueError(
                f"GenSearchesAgent: No valid searches after cleaning: {valid_searches}"
            )

        response_dict["searches"] = cleaned_searches

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


class GenSearchesRefinementAgent(BaseAgent):
    def __init__(self, model_name: str, max_history_size: int | None = None):
        """Initialize the GenSearchesRefinementAgent.

        Args:
            model_name (str): The name of the language model to use
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="GenSearchesRefinementAgent",
            description="A specialized agent that generates additional web searches based on user feedback and previous search queries",
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
You are a search query refinement specialist. Your task is to analyze existing Google search queries and user feedback to generate ONE new strategic search query that addresses the user's additional information needs.

**INSTRUCTIONS**:

Based on the user feedback, you will generate a single new Google search query that:
1. **Addresses the user's request**: The new search should directly target the additional information requested in the user feedback
2. **Avoids duplication**: The new search must be different from all existing searches and retrieve distinct information
3. **Complements existing searches**: It should fill gaps in the current search strategy rather than overlap with existing queries

**Search Query Best Practices:**
- Use specific entity names, dates, and locations from the original statement and user feedback
- Target specific organizations, institutions, or authoritative sources when relevant
- Include terms that help filter for recent, relevant information (add year {datetime.datetime.now().year} when current data is needed)
- Focus on credible sources: official reports, peer-reviewed studies, expert analyses
- Make the query actionable and likely to return relevant, credible results
- **NEVER use quotes around phrases** - Write natural search queries without quotation marks
- **NEVER use site: operators** - Do not restrict searches to specific websites
- Keep searches simple and natural, as if typed directly into Google by a regular user

**Quality Standards:**
- The search must be distinct from all previous searches
- Use specific terminology that experts and official sources would use
- Include temporal qualifiers when recent information is crucial
- Focus on factual, verifiable information rather than opinions

**Output Format:**
Respond with JSON containing:
- `"analysis"`: A brief explanation of why this new search is needed and how it differs from existing searches
- `"searches"`: Array containing ONE new Google search query

**Example Output:**
```json
{{
  "analysis": "User requested information about Chinese manufacturing data. Existing searches focus on U.S. and European sources. The new search targets Chinese official statistics and trade data.",
  "searches": [
    "China manufacturing output statistics {datetime.datetime.now().year} official government data industrial production"
  ]
}}
```

**Context:** Current year is {datetime.datetime.now().year}
**Language:** All searches should be in {lang}
""".strip()

    def _get_conversation(
        self,
        statement: str,
        langid: str,
        current_searches: list[str],
        user_feedback: str,
        override_prompt: str | None = None,
    ) -> list[dict]:
        """Build the conversation structure for the LLM.

        Args:
            statement (str): The original statement being fact-checked
            langid (str): Language identifier code
            current_searches (List[str]): The current list of search queries
            user_feedback (str): User's feedback and refinement requests
            override_prompt (Optional[str]): Custom prompt to use instead of default

        Returns:
            List[dict]: List of conversation messages formatted for the LLM
        """
        current_searches_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(current_searches)])

        user_message = f"""**Original Statement:** {statement}

**Current Search Queries:**
{current_searches_text}

**User Feedback and Additional Information Request:**
{user_feedback}

**Task:** Generate ONE new Google search query that addresses the user's feedback while avoiding duplication with existing searches.""".strip()

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
        current_searches: list[str],
        user_feedback: str,
        override_prompt: str | None = None,
        override_pydantic_model: BaseModel | None = None,
    ) -> tuple[Any, float, HistoryEntry]:
        """Execute the agent to generate a new search query based on user feedback.

        Args:
            statement (str): The statement to analyze
            langid (str): Language identifier code
            current_searches (List[str]): The current list of searches
            user_feedback (str): User's feedback and refinement requests
            override_prompt (Optional[str]): Custom prompt to use instead of default
            override_pydantic_model (Optional[BaseModel]): Custom Pydantic model for response parsing

        Returns:
            Tuple[Any, float, HistoryEntry]: A tuple containing:
                - The parsed model response (analysis and new search)
                - The cost of the API call
                - History entry containing conversation data and metadata
        """

        conversation = self._get_conversation(
            statement=statement,
            langid=langid,
            current_searches=current_searches,
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
        if "searches" not in response_dict:
            raise ValueError(
                f"GenSearchesRefinementAgent: Missing 'searches' field in model response: {response_dict}"
            )

        if "analysis" not in response_dict:
            raise ValueError(
                f"GenSearchesRefinementAgent: Missing 'analysis' field in model response: {response_dict}"
            )

        # Validate new searches
        new_searches = response_dict["searches"]
        if not isinstance(new_searches, list):
            raise ValueError(
                f"GenSearchesRefinementAgent: 'searches' field is not a list: {type(new_searches)}"
            )

        if not new_searches:
            raise ValueError("GenSearchesRefinementAgent: Empty searches list generated")

        # Filter out empty or invalid searches
        valid_searches = [s for s in new_searches if isinstance(s, str) and s.strip()]
        if not valid_searches:
            raise ValueError(
                f"GenSearchesRefinementAgent: No valid search strings found in: {new_searches}"
            )

        # Ensure only one search is returned
        if len(valid_searches) > 1:
            valid_searches = valid_searches[:1]  # Only keep the first search

        # Clean searches to remove quotes and site: operators
        cleaned_searches = [clean_search_query(s) for s in valid_searches]

        # Filter out any searches that became empty after cleaning
        cleaned_searches = [s for s in cleaned_searches if s]
        if not cleaned_searches:
            raise ValueError(
                f"GenSearchesRefinementAgent: No valid searches after cleaning: {valid_searches}"
            )

        response_dict["searches"] = cleaned_searches

        # Validate analysis
        analysis = response_dict["analysis"]
        if not isinstance(analysis, str):
            raise ValueError(
                f"GenSearchesRefinementAgent: 'analysis' field is not a string: {type(analysis)}"
            )

        if not analysis.strip():
            raise ValueError("GenSearchesRefinementAgent: Empty analysis generated")

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
