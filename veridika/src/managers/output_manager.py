import re
from typing import Any

from veridika.src.agents.baseagent import BaseAgent
from veridika.src.managers.cication_manager import CitationManager


def escape_black_citations(text: str) -> str:
    """
    Given a string where citations like “[1]” may appear **inside** bold spans
    (Markdown `**bold**`), rewrites it so the citations sit *outside* the
    bold-face while preserving the surrounding text in bold.

    Example
    -------
    >>> s = "This is a **test [1] of a sentence** in bold"
    >>> escape_black_citations(s)
    'This is a **test** [1] **of a sentence** in bold'
    """
    bold_re = re.compile(r"\*\*(.*?)\*\*", re.DOTALL)  # every **…** block
    citation_re = re.compile(r"\[\d+\]")  # [123] style refs

    def _fix(match: re.Match) -> str:
        inner = match.group(1)
        if not citation_re.search(inner):  # nothing to do
            return match.group(0)

        pieces, pos = [], 0
        for m in citation_re.finditer(inner):
            before = inner[pos : m.start()]
            if before.strip():
                pieces.append(f"**{before.strip()}**")
            pieces.append(m.group(0))  # the citation itself
            pos = m.end()
        after = inner[pos:]
        if after.strip():
            pieces.append(f"**{after.strip()}**")
        return " ".join(pieces)

    return bold_re.sub(_fix, text)


def clean_factchecking(text: str) -> str:
    """
    Clean and filter fact-checking text to return exactly 3 substantial paragraphs.

    This function splits the input text into paragraphs and selects the 3 most
    substantial ones based on word count. It prioritizes paragraphs with more than
    35 words, and if fewer than 3 such paragraphs exist, it fills the remainder
    with the largest available paragraphs.

    Args:
        text (str): The input fact-checking text to be processed

    Returns:
        str: A cleaned text containing up to 3 substantial paragraphs,
             joined by double newlines. If the input has 3 or fewer paragraphs,
             returns the original text unchanged.
    """
    text = text.strip()
    paragraphs = text.split("\n\n")

    # Return early if we already have 3 or fewer paragraphs
    if len(paragraphs) <= 3:
        return text

    # Find paragraphs with substantial content (>35 words)
    substantial_paragraphs = []
    remaining_paragraphs = []

    for i, paragraph in enumerate(paragraphs):
        word_count = len(paragraph.split())
        if word_count > 35:
            substantial_paragraphs.append((i, paragraph))
        else:
            remaining_paragraphs.append((i, paragraph))

    # Select paragraphs to keep
    selected_paragraphs = substantial_paragraphs.copy()

    # If we need more paragraphs to reach 3, add the largest remaining ones
    if len(selected_paragraphs) < 3:
        # Sort remaining paragraphs by length (descending)
        remaining_paragraphs.sort(key=lambda x: len(x[1]), reverse=True)

        # Add paragraphs until we have 3 total
        needed_count = 3 - len(selected_paragraphs)
        selected_paragraphs.extend(remaining_paragraphs[:needed_count])

    # Sort by original index to maintain order
    selected_paragraphs.sort(key=lambda x: x[0])

    # Extract paragraph text and rejoin
    final_paragraphs = [paragraph for _, paragraph in selected_paragraphs]
    result = "\n\n".join(final_paragraphs)

    return result.strip()


def generate_factchecking_response(
    citation_manager: CitationManager,
    article_result: dict[str, str],
    metadata: dict[str, Any],
    cost: float,
    runtime: float,
    agents: list[BaseAgent] | None = None,
    config: dict[str, Any] | None = None,
    qa_results: dict[str, str] | None = None,
    image_url: str | None = None,
    image_description: str | None = None,
) -> dict[str, Any]:
    """Generate a complete fact-checking response from agent outputs.

    Args:
        citation_manager (CitationManager): Shared citation manager used across agents
        article_result (Dict[str, str]): Output from ArticleWriterAgent {"article": "text"}
        qa_results (Optional[Dict[str, str]]): Output from QuestionAgent {question: answer}
        metadata (Dict[str, Any]): Output from MetadataAgent
        cost (float): Total cost of the workflow
        runtime (float): Total runtime of the workflow
        agents (List[BaseAgent]): List of agents to collect histories from
        config (Dict[str, Any]): Configuration parameters used
        image_url (Optional[str]): URL of generated image (if any)
        image_description (Optional[str]): Description of generated image (if any)

    Returns:
        Dict[str, Any]: Complete structured fact-checking response
    """
    # Extract fact-checking article text
    fact_checking_text = article_result.get("article", "")

    # Prepare all texts for citation reordering
    all_texts = [fact_checking_text] + (list(qa_results.values()) if qa_results else [])

    # Reorder citations consistently across all texts
    fixed_texts, citation_ids = citation_manager.reorder_citations(all_texts)

    # Extract the reordered fact-checking text
    reordered_fact_checking = fixed_texts[0]

    # Create reordered QA results
    reordered_qa_results = {}
    if qa_results:
        qa_questions = list(qa_results.keys())
        for i, question in enumerate(qa_questions):
            reordered_answer = escape_black_citations(fixed_texts[i + 1])
            reordered_qa_results[question] = reordered_answer

    # Get citation metadata for all used citations
    citation_metadata = citation_manager.get_metadata(citation_ids)
    citations_dict = {
        str(cid): {"url": meta.url, "source": meta.source, "favicon": meta.favicon}
        for cid, meta in zip(citation_ids, citation_metadata, strict=False)
    }

    # Collect histories from all agents
    logs = {}

    if agents is not None:
        for agent in agents:
            agent_stats = agent.get_stats()
            agent_history = agent.get_history()

            logs[agent.name] = {"stats": agent_stats, "history": agent_history}

        # Add summary statistics
        logs["summary"] = {
            "total_cost": cost,
            "total_calls": 0,
            "agents_count": len(agents),
            "config": config,
        }

    # Build the final response
    response = {
        "question": metadata.get("title", ""),
        "answer": clean_factchecking(escape_black_citations(reordered_fact_checking)),
        "image": {"url": image_url or "", "description": image_description or ""},
        "sources": citations_dict,
        "related_questions": reordered_qa_results,
        "metadata": {**metadata, "config": config},
        "logs": logs,
    }

    return response


def collect_agent_costs(agents: list[BaseAgent]) -> dict[str, float]:
    """Collect cost information from all agents.

    Args:
        agents (List[BaseAgent]): List of agents to collect costs from

    Returns:
        Dict[str, float]: Dictionary mapping agent names to their total costs
    """
    costs = {}
    for agent in agents:
        costs[agent.name] = agent.cost
    return costs


def collect_agent_statistics(agents: list[BaseAgent]) -> dict[str, dict[str, Any]]:
    """Collect comprehensive statistics from all agents.

    Args:
        agents (List[BaseAgent]): List of agents to collect statistics from

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping agent names to their statistics
    """
    stats = {}
    for agent in agents:
        stats[agent.name] = agent.get_stats()
    return stats
