import asyncio
from typing import Any

from veridika.src.agents.ArticleWriterAgent import ArticleWriterAgent
from veridika.src.agents.CriticalQuestionAgent import (
    CriticalQuestionAgent,
    CriticalQuestionRefinementAgent,
)
from veridika.src.agents.GenSearchesAgent import (
    GenSearchesAgent,
    GenSearchesRefinementAgent,
)
from veridika.src.agents.ImageGenAgent import ImageGenAgent
from veridika.src.agents.ImagePromptAgent import ImagePromptAgent
from veridika.src.agents.MetadataAgent import MetadataAgent
from veridika.src.agents.QuestionAgent import QuestionAgent
from veridika.src.agents.RagAgent import RagAgent
from veridika.src.agents.WebSearchAgent import WebSearchAgent
from veridika.src.embeddings import Embeddings
from veridika.src.managers.cication_manager import CitationManager
from veridika.src.managers.output_manager import generate_factchecking_response
from veridika.src.web_search.utils import download_content
from veridika.src.workflows.baseworkflow import BaseStepwiseWorkflow


class CrititalQuestionWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self, statement: str, model: str, language: str, location: str
    ) -> tuple[dict[str, list[str]], float]:
        """Generate critical questions for a given statement.

        Args:
            statement (str): Input statement to analyze and fact-check.
            model (str): Model name to use for the agent.
            language (str): Language code (e.g., "es", "en").
            location (str): User or search location context (unused here).

        Returns:
            Tuple[Dict[str, List[str]], float]:
                - A dict with key "questions" mapping to the generated questions.
                - The accumulated model cost for this step.
        """
        critical_question_agent = CriticalQuestionAgent(model_name=model)
        result, cost = await critical_question_agent(statement=statement, langid=language)
        return {"questions": result["questions"]}, cost


class CrititalQuestionRefinementWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self,
        questions: list[str],
        input: str,
        refinement: str,
        language: str,
        location: str,
        model: str,
    ) -> tuple[dict[str, list[str]], float]:
        """Refine an existing list of critical questions with user feedback.

        Args:
            questions (List[str]): Current list of critical research questions.
            input (str): Original statement/topic being fact-checked.
            refinement (str): User feedback to guide the refinement.
            language (str): Language code (e.g., "es", "en").
            location (str): User or search location context (unused here).
            model (str): Model name to use for the agent.

        Returns:
            Tuple[Dict[str, List[str]], float]:
                - A dict with key "questions" mapping to the refined questions.
                - The accumulated model cost for this step.
        """
        refinement_agent = CriticalQuestionRefinementAgent(model_name=model)
        result, cost = await refinement_agent(
            statement=input,
            langid=language,
            current_questions=questions,
            user_feedback=refinement,
        )
        return {"questions": result["questions"]}, cost


class GenSearchesWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self,
        questions: list[str],
        statement: str,
        language: str,
        location: str,
        model: str,
        web_search_provider: str = "serper",
        top_k: int = 5,
        ban_domains: list[str] = None,
    ) -> tuple[list[dict[str, str]], float]:
        """Generate search queries from questions and fetch top web sources.

        Args:
            question (List[str]): Critical research questions to guide searches.
            statement (str): Original statement/topic being fact-checked.
            language (str): Search language code (e.g., "es", "en").
            location (str): Search location/region for the web search provider.
            model (str): Model name for the search-query generation agent.
            web_search_provider (str): Web search provider identifier (e.g., "serper").
            top_k (int): Number of results per query to retrieve.
            ban_domains (List[str], optional): Domains to exclude from results.

        Returns:
            Tuple[List[Dict[str, str]], float]:
                - A list of unified source dicts with keys: "title", "link",
                  "snippet", "favicon", "base_url".
                - The accumulated cost (query generation + web search).
        """
        gen_searches_agent = GenSearchesAgent(model_name=model)
        web_search_agent = WebSearchAgent(search_provider=web_search_provider)

        gen_result, cost_gen_searches = await gen_searches_agent(
            statement=statement, questions=questions, langid=language
        )
        searches = gen_result["searches"]

        web_result, cost_web_search = await web_search_agent(
            queries=searches,
            language=language,
            location=location,
            top_k=top_k,
            ban_domains=ban_domains,
            download_text=False,
            download_favicon=True,
        )

        sources = []
        for docs in web_result.values():
            for doc in docs:
                meta = doc["metadata"]
                if meta["url"] is None:
                    continue
                sources.append(
                    {
                        "title": meta["title"],
                        "link": meta["url"],
                        "snippet": meta["snippet"],
                        "favicon": meta["favicon"],
                        "base_url": meta["base_url"],
                    }
                )

        return sources, searches, cost_gen_searches + cost_web_search


class SourceRefinementWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self,
        current_searches: list[str],
        statement: str,
        language: str,
        location: str,
        model: str,
        user_feedback: str,
        web_search_provider: str = "serper",
        top_k: int = 5,
        ban_domains: list[str] = None,
    ) -> tuple[list[dict[str, str]], float]:
        """Generate one new search based on feedback and fetch updated sources.

        Args:
            current_searches (List[str]): Existing search queries.
            statement (str): Original statement/topic being fact-checked.
            language (str): Search language code (e.g., "es", "en").
            location (str): Search location/region for the web search provider.
            model (str): Model name for the refinement agent.
            user_feedback (str): Feedback describing additional information needs.
            web_search_provider (str): Web search provider identifier (e.g., "serper").
            top_k (int): Number of results per query to retrieve.
            ban_domains (List[str], optional): Domains to exclude from results.

        Returns:
            Tuple[List[Dict[str, str]], float]:
                - A list of unified source dicts with keys: "title", "link",
                  "snippet", "favicon", "base_url".
                - The accumulated cost (refinement + web search).
        """
        source_refinement_agent = GenSearchesRefinementAgent(model_name=model)
        web_search_agent = WebSearchAgent(search_provider=web_search_provider)

        gen_result, cost_gen_searches = await source_refinement_agent(
            statement=statement,
            current_searches=current_searches,
            user_feedback=user_feedback,
            langid=language,
        )
        searches = gen_result["searches"]

        web_result, cost_web_search = await web_search_agent(
            queries=searches,
            language=language,
            location=location,
            top_k=top_k,
            ban_domains=ban_domains,
            download_text=False,
            download_favicon=True,
        )

        sources = []
        for docs in web_result.values():
            for doc in docs:
                meta = doc["metadata"]
                if meta["url"] is None:
                    continue
                sources.append(
                    {
                        "title": meta["title"],
                        "link": meta["url"],
                        "snippet": meta["snippet"],
                        "favicon": meta["favicon"],
                        "base_url": meta["base_url"],
                    }
                )

        return sources, searches, cost_gen_searches + cost_web_search


class ImageGenerationWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self,
        statement: str,
        model: str,
        size: str = "1280x256",
        style: str = (
            "The style is bold and energetic, using vibrant colors and dynamic compositions to create visually engaging scenes that emphasize activity, culture, and lively atmospheres. Digital art."
        ),
        image_model: str = "flux_replicate",
    ) -> tuple[dict[str, str], float]:
        """Generate an image for the provided statement.

        Workflow:
        1) Use ImagePromptAgent to create a concise image description (English)
        2) Use ImageGenAgent (Flux) to generate an image URL from that description

        Args:
            statement (str): The article title or statement to illustrate
            model (str): LLM model name for prompt generation
            size (str): Output image size, e.g., "1280x256"
            style (str): Optional style guidance appended to the description

        Returns:
            Tuple[Dict[str, str], float]:
                - Dict containing keys: "image_description", "image_url", "size"
                - Total cost (prompt + image generation)
        """
        # Step 1: Generate image description
        prompt_agent = ImagePromptAgent(model_name=model)
        description_output, prompt_cost, _history_prompt = prompt_agent.run(
            fact_checking_title=statement
        )

        # Step 2: Generate image using Flux
        image_agent = ImageGenAgent(image_model=image_model)
        image_output, image_cost, _history_image = image_agent.run(
            image_description_output=description_output,
            size=size,
            style=style,
        )

        total_cost = prompt_cost + image_cost

        return {
            "image_description": description_output.get("image_description", ""),
            "image_url": image_output.get("image_url", None),
            "size": size,
        }, total_cost


class FactCheckingWorkflow(BaseStepwiseWorkflow):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    async def run(
        self,
        question: list[str],
        statement: str,
        language: str,
        location: str,
        sources: list[dict[str, str]],
        model: str,
        use_rag: bool,
        embedding_model: str,
        article_writer_model: str,
        question_answer_model: str,
        metadata_model: str,
        chunk_overlap: int = 0,
        chunk_size: int = 128,
        split_separators: list[str] = ["\n"],
        fp16_storage: bool = True,
        l2_normalise: bool = True,
        top_k: int = 5,
    ) -> tuple[dict[str, Any], float]:
        """Run the end-to-end fact-checking workflow and return a response.

        Args:
            question (List[str]): Critical questions used for QA/RAG and article.
            statement (str): Original statement/topic to fact-check.
            language (str): Language code for generated outputs (e.g., "es", "en").
            location (str): Search region context for upstream web results.
            sources (List[Dict[str, str]]): Input sources in unified web format
                with keys: "title", "link", "snippet", "favicon", "base_url".
            model (str): Model name (not directly passed to agents' run methods).
            use_rag (bool): Whether to build an embedding index and retrieve.
            embedding_model (str): Embedding model identifier for RAG.
            article_writer_model (str): Model for article generation.
            question_answer_model (str): Model for QA generation.
            metadata_model (str): Model for extracting article metadata.
            chunk_overlap (int): Overlap in characters for text chunking.
            chunk_size (int): Chunk size in characters for text splitting.
            split_separators (List[str]): Separators for splitting source text.
            fp16_storage (bool): Store document embeddings in float16.
            l2_normalise (bool): L2-normalise embeddings.
            top_k (int): Top passages per query for retrieval.

        Returns:
            Tuple[Dict[str, Any], float]:
                - A structured response from `generate_factchecking_response` that
                  includes article content, QA results, metadata and citations.
                - The accumulated cost of QA, article, and metadata steps
                  (RAG cost not included in total_cost comment here).
        """
        citation_manager = CitationManager()
        embedding_model = Embeddings(embedding_model)
        rag_agent = RagAgent(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_separators=tuple(split_separators),
            fp16_storage=fp16_storage,
            l2_normalise=l2_normalise,
        )
        question_agent = QuestionAgent(model_name=question_answer_model)
        article_writer_agent = ArticleWriterAgent(model_name=article_writer_model)
        metadata_agent = MetadataAgent(model_name=metadata_model)

        # Download text for sources
        # Filter out sources with None links
        valid_sources = [s for s in sources if s.get("link") is not None]
        urls = [s["link"] for s in valid_sources]
        download_results = download_content(
            urls=urls, download_text=True, download_favicon=False, download_title=False
        )
        documents = []
        for (title, text, url, favicon), source in zip(
            download_results, valid_sources, strict=False
        ):
            if len(text.split()) < 50:
                continue
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "url": url,
                        "source": source["base_url"],
                        "favicon": source["favicon"],
                        "title": source["title"],
                        "type": "web_search",
                        "base_url": source["base_url"],
                        "snippet": source["snippet"],
                    },
                }
            )

        # Prepare search_results
        if use_rag:
            rag_result, _ = await rag_agent(
                websearch_results={"dummy": documents},
                queries=question,
                top_k=top_k,
                get_scores=False,
            )
        else:
            rag_result = {" | ".join(question): documents}

        # Run QuestionAgent and ArticleWriterAgent in parallel
        question_task = question_agent(
            search_results=rag_result,
            lang=language,
            citation_manager=citation_manager,
        )
        article_task = article_writer_agent(
            search_results=rag_result,
            fact_checking_topic=statement,
            lang=language,
            citation_manager=citation_manager,
        )

        (qa_result, qa_cost), (article_result, article_cost) = await asyncio.gather(
            question_task, article_task
        )

        # Metadata
        metadata_result, metadata_cost = await metadata_agent(
            article_output=article_result,
            fact_checking_title=statement,
            lang=language,
        )

        total_cost = qa_cost + article_cost + metadata_cost  # Add rag_cost if use_rag

        response = generate_factchecking_response(
            citation_manager=citation_manager,
            article_result=article_result,
            qa_results=qa_result,
            metadata=metadata_result,
            cost=total_cost,
            runtime=None,
            agents=None,
            config=None,
            image_url=None,
            image_description=None,
        )

        return response, total_cost
