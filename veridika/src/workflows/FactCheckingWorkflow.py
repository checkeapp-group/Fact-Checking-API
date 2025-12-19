import asyncio
import time
from typing import Any

from veridika.src.agents.ArticleWriterAgent import ArticleWriterAgent
from veridika.src.agents.CriticalQuestionAgent import CriticalQuestionAgent
from veridika.src.agents.GenSearchesAgent import GenSearchesAgent
from veridika.src.agents.ImageGenAgent import ImageGenAgent
from veridika.src.agents.ImagePromptAgent import ImagePromptAgent
from veridika.src.agents.MetadataAgent import MetadataAgent
from veridika.src.agents.QuestionAgent import QuestionAgent
from veridika.src.agents.RagAgent import RagAgent
from veridika.src.agents.WebSearchAgent import WebSearchAgent
from veridika.src.embeddings import Embeddings
from veridika.src.managers.output_manager import generate_factchecking_response
from veridika.src.workflows.baseworkflow import BaseWorkflow


class FactCheckingWithPipelineWorkflow(BaseWorkflow):
    """Fact-checking workflow that includes RAG for enhanced document retrieval.

    This workflow orchestrates the following agents:
    - ImagePromptAgent -> ImageGenAgent (parallel with main workflow)
    - GenSearchesAgent -> WebSearchAgent -> RagAgent -> QuestionAgent/ArticleWriterAgent -> MetadataAgent

    QuestionAgent and ArticleWriterAgent run in parallel for efficiency.
    """

    def _initialize_agents(self) -> None:
        """Initialize all agents for the RAG-enhanced fact-checking workflow."""
        # Initialize embedding model for RAG
        embedding_model = Embeddings(self.config["models"]["embedding"])

        self.critical_question_agent = CriticalQuestionAgent(
            model_name=self.config["models"]["critical_question_agent"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        # Initialize all agents
        self.gen_searches_agent = GenSearchesAgent(
            model_name=self.config["models"]["gen_searches_agent"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.web_search_agent = WebSearchAgent(
            search_provider=self.config["models"]["web_search_provider"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.rag_agent = RagAgent(
            embedding_model=embedding_model,
            chunk_size=self.config["rag"]["chunk_size"],
            chunk_overlap=self.config["rag"]["chunk_overlap"],
            split_separators=tuple(self.config["rag"]["split_separators"]),
            fp16_storage=self.config["rag"]["fp16_storage"],
            l2_normalise=self.config["rag"]["l2_normalise"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.question_agent = QuestionAgent(
            model_name=self.config["models"]["question_agent"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.article_writer_agent = ArticleWriterAgent(
            model_name=self.config["models"]["article_writer"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.metadata_agent = MetadataAgent(
            model_name=self.config["models"]["metadata_agent"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.image_prompt_agent = ImagePromptAgent(
            model_name=self.config["models"]["image_prompt"],
            max_history_size=self.config["general"]["max_history_size"],
        )

        self.image_gen_agent = ImageGenAgent(
            max_history_size=self.config["general"]["max_history_size"],
            image_model=self.config["models"].get("image_model", "flux_replicate"),
        )

        # Add all agents to the workflow
        self._add_agent(self.gen_searches_agent)
        self._add_agent(self.web_search_agent)
        self._add_agent(self.rag_agent)
        self._add_agent(self.question_agent)
        self._add_agent(self.article_writer_agent)
        self._add_agent(self.metadata_agent)
        self._add_agent(self.image_prompt_agent)
        self._add_agent(self.image_gen_agent)

    def reset_workflow_state(self):
        """Reset all workflow state to prevent accumulation between evaluation examples.

        This ensures each fact-checking run starts with a completely clean state:
        - Resets all agent statistics and history
        - Clears RAG agent document corpus and embeddings
        - Creates a fresh citation manager
        """
        # Reset all agents' statistics and history using the base workflow method
        self.reset_all_agents()

        # Create a fresh citation manager for this run to avoid citation ID conflicts
        self.citation_manager = self.get_citation_manager(new_manager=True)

    async def run(
        self,
        statement: str,
        internal_language: str = "es",
        output_language: str = "es",
        location: str = "es",
        generate_image: bool = True,
        answer_questions: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute the complete fact-checking workflow with RAG.

        Args:
            statement (str): The fact-checking statement or question
            internal_language (str): Language code for internal processing (default: "en")
            output_language (str): Language code for output (default: "en")
            generate_image (bool): Whether to generate header image (default: True)
            answer_questions (bool): Whether to answer related questions (default: True)

        Returns:
            Dict[str, Any]: Complete formatted fact-checking response
        """
        # Reset all workflow state at the beginning of each run to prevent accumulation
        # This ensures clean state for each fact-checking operation
        self.reset_workflow_state()

        self._start_timing()
        start_time = time.perf_counter()

        # Start complete image generation in parallel with main workflow if requested
        image_generation_task = None
        if generate_image:
            # Create complete image generation task that runs in parallel
            async def generate_complete_image():
                # Generate image description
                image_description_result, image_description_cost = await self.image_prompt_agent(
                    fact_checking_title=statement
                )
                # Generate image
                image_result, image_cost = await self.image_gen_agent(
                    image_description_output=image_description_result,
                    size=self.config["image"]["size"],
                    style=self.config["image"]["style"],
                )
                return (
                    image_result.get("image_url"),
                    image_description_result.get("image_description"),
                    image_cost + image_description_cost,
                )

            # Start the complete image generation task
            image_generation_task = asyncio.create_task(generate_complete_image())

        # Step 1: Generate searches and questions
        questions, critical_question_cost = await self.critical_question_agent(
            statement=statement, langid=internal_language
        )
        questions = questions["questions"]

        if not questions:
            raise ValueError(
                f"GenSearchesAgent returned empty questions list. This will cause downstream failures. "
                f"Full agent output: {questions}"
            )

        if not isinstance(questions, list):
            raise ValueError(
                f"GenSearchesAgent returned invalid questions format. Expected list, got {type(questions)}. "
                f"Full agent output: {questions}"
            )

        if len(questions) > self.config["questions"]["max_questions"]:
            questions = questions[: self.config["questions"]["max_questions"]]

        searches, gen_searches_cost = await self.gen_searches_agent(
            statement=statement, questions=questions, langid=internal_language
        )

        searches = searches["searches"]

        if not isinstance(searches, list):
            raise ValueError(
                f"GenSearchesAgent returned invalid searches format. Expected list, got {type(searches)}. "
                f"Full agent output: {searches}"
            )

        if not searches:
            raise ValueError(
                f"GenSearchesAgent returned empty searches list. This will cause downstream failures. "
                f"Full agent output: {searches}"
            )

        if len(searches) > self.config["web_search"]["max_searches"]:
            searches = searches[: self.config["web_search"]["max_searches"]]

        # Step 2: Execute web search
        web_search_result, web_search_cost = await self.web_search_agent(
            queries=searches,
            language=output_language,
            location=location,
            top_k=self.config["web_search"]["top_k"],
            ban_domains=self.config["web_search"]["ban_domains"],
            download_text=self.config["web_search"]["download_text"],
        )

        # Step 3: Execute RAG if enabled
        if self.config["rag"].get("do_rag", False):
            rag_result, rag_cost = await self.rag_agent(
                websearch_results=web_search_result,
                queries=questions,
                top_k=self.config["rag"]["top_k"],
                get_scores=self.config["rag"]["get_scores"],
            )
        else:
            rag_result = web_search_result
            rag_cost = 0.0

        # Step 4: Run QuestionAgent and ArticleWriterAgent in parallel
        if answer_questions:
            question_task = self.question_agent(
                search_results=rag_result,
                lang=output_language,
                citation_manager=self.citation_manager,
            )
        else:
            question_task = None

        article_task = self.article_writer_agent(
            search_results=rag_result,
            fact_checking_topic=statement,
            lang=output_language,
            citation_manager=self.citation_manager,
        )

        # Wait for both tasks to complete
        if answer_questions:
            (qa_result, qa_cost), (article_result, article_cost) = await asyncio.gather(
                question_task, article_task
            )
        else:
            (article_result, article_cost) = await article_task
            qa_result = None
            qa_cost = 0.0

        # Step 5: Extract metadata from article
        metadata_result, metadata_cost = await self.metadata_agent(
            article_output=article_result, fact_checking_title=statement, lang=output_language
        )

        # Step 6: Wait for image generation to complete if it was started
        image_url = None
        image_description = None
        image_cost = 0.0
        if generate_image:
            # Wait for complete image generation task
            image_url, image_description, image_cost = await image_generation_task

        total_cost = (
            critical_question_cost
            + gen_searches_cost
            + web_search_cost
            + rag_cost
            + qa_cost
            + article_cost
            + metadata_cost
            + image_cost
        )
        # Generate final response using output manager
        response = generate_factchecking_response(
            citation_manager=self.citation_manager,
            article_result=article_result,
            qa_results=qa_result,
            metadata=metadata_result,
            cost=total_cost,
            runtime=time.perf_counter() - start_time,
            agents=self.get_agents(),
            config=self.config,
            image_url=image_url,
            image_description=image_description,
        )

        self._stop_timing()

        return (
            response,
            total_cost,
        )

    def get_workflow_name(self) -> str:
        """Get the name of this workflow.

        Returns:
            str: Workflow name
        """
        return "FactCheckingPipeline"
