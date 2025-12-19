#!/usr/bin/env python3
"""
Veridika Fact-Checking Server

A FastAPI server that processes fact-checking jobs from a Redis queue.
Each server instance processes one job at a time, but multiple servers
can work from the same Redis queue for horizontal scaling.

Environment Variables:
    WORKFLOW_CONFIG_PATH: Path to the workflow configuration file (required)
    API_KEY: Secret API key for authentication (required)
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    SERVER_ID: Unique identifier for this server instance (optional)
    JOB_QUEUE_NAME: Name of the Redis queue for jobs (default: veridika_jobs)
    RESULT_QUEUE_NAME: Name of the Redis queue for results (default: veridika_results)
    LOG_LEVEL: Logging level (default: INFO)

Usage:
    export WORKFLOW_CONFIG_PATH=configs/pipeline_configs/gemini.yaml
    export API_KEY=your-secret-api-key-here
    export REDIS_URL=redis://localhost:6379
    python veridika_server.py
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from typing import Any
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis

from veridika.src.workflows import Workflow
from veridika.src.workflows.StepwiseWorkflows import (
    CrititalQuestionRefinementWorkflow,
    CrititalQuestionWorkflow,
    FactCheckingWorkflow,
    GenSearchesWorkflow,
    ImageGenerationWorkflow,
    SourceRefinementWorkflow,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
WORKFLOW_CONFIG_PATH = os.getenv("WORKFLOW_CONFIG_PATH")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SERVER_ID = os.getenv("SERVER_ID", f"veridika_server_{uuid.uuid4().hex[:8]}")
JOB_QUEUE_NAME = os.getenv("JOB_QUEUE_NAME", "veridika_jobs")
RESULT_QUEUE_NAME = os.getenv("RESULT_QUEUE_NAME", "veridika_results")
REDIS_JOB_EXPIRY = 60 * 15  # 15 minutes
API_KEY = os.getenv("API_KEY")
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))

# Validate required environment variables
if not WORKFLOW_CONFIG_PATH:
    raise RuntimeError("WORKFLOW_CONFIG_PATH environment variable is required")

if not os.path.exists(WORKFLOW_CONFIG_PATH):
    raise RuntimeError(f"Workflow config file not found: {WORKFLOW_CONFIG_PATH}")

if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")


class FactCheckRequest(BaseModel):
    """Request model for fact-checking jobs."""

    article_topic: str = Field(
        ..., min_length=10, max_length=1000, description="The topic or statement to fact-check"
    )
    language: str = Field(..., description="Language code for processing and output")
    location: str = Field(
        ..., min_length=1, max_length=2, description="Location code for web search"
    )
    config: str = Field(
        ..., description="Legacy config parameter (unused - kept for compatibility)"
    )


class FactCheckResponse(BaseModel):
    """Response model for fact-checking job submission."""

    job_id: str


# Stepwise Workflow Request/Response Models


class CriticalQuestionRequest(BaseModel):
    """Request model for critical question generation."""

    input: str = Field(..., min_length=10, max_length=1000, description="The statement to analyze")
    model: str = Field(..., description="Model name to use")
    language: str = Field(..., description="Language code")
    location: str = Field(..., description="Location code")


class CriticalQuestionResponse(BaseModel):
    """Response model for critical question generation."""

    questions: list[str]
    cost: float


class CriticalQuestionRefinementRequest(BaseModel):
    """Request model for critical question refinement."""

    questions: list[str] = Field(..., description="Current list of questions")
    input: str = Field(..., min_length=10, max_length=1000, description="Original statement")
    refinement: str = Field(..., description="User feedback for refinement")
    language: str = Field(..., description="Language code")
    location: str = Field(..., description="Location code")
    model: str = Field(..., description="Model name to use")


class SourceSearchRequest(BaseModel):
    """Request model for source search."""

    questions: list[str] = Field(..., description="List of critical questions")
    input: str = Field(..., min_length=10, max_length=1000, description="Original statement")
    language: str = Field(..., description="Language code")
    location: str = Field(..., description="Location code")
    model: str = Field(..., description="Model name to use")
    web_search_provider: str = Field(default="serper", description="Web search provider")
    top_k: int = Field(default=5, description="Number of results per query")
    ban_domains: list[str] | None = Field(default=None, description="Domains to exclude")


class SourceDict(BaseModel):
    """Model for a single source."""

    title: str
    link: str
    snippet: str
    favicon: str | None
    base_url: str


class SourceSearchResponse(BaseModel):
    """Response model for source search."""

    sources: list[SourceDict]
    searches: list[str]
    cost: float


class SourceRefinementRequest(BaseModel):
    """Request model for source refinement."""

    input: str = Field(..., min_length=10, max_length=1000, description="Original statement")
    searches: list[str] = Field(..., description="List of current searches")
    language: str = Field(..., description="Language code")
    location: str = Field(..., description="Location code")
    refinement: str = Field(..., description="User feedback for refinement")
    model: str = Field(..., description="Model name to use")
    web_search_provider: str = Field(default="serper", description="Web search provider")
    top_k: int = Field(default=5, description="Number of results per query")
    ban_domains: list[str] | None = Field(default=None, description="Domains to exclude")


class FinalFactCheckRequest(BaseModel):
    """Request model for final fact-checking."""

    questions: list[str] = Field(..., description="List of critical questions")
    input: str = Field(..., min_length=10, max_length=1000, description="Original statement")
    language: str = Field(..., description="Language code")
    location: str = Field(..., description="Location code")
    sources: list[SourceDict] = Field(..., description="List of sources")
    model: str = Field(..., description="Model name to use")
    use_rag: bool = Field(default=True, description="Whether to use RAG")
    embedding_model: str = Field(default="gemini-embedding-001", description="Embedding model")
    article_writer_model: str | None = Field(
        default=None, description="Article writer model (defaults to model)"
    )
    question_answer_model: str | None = Field(
        default=None, description="QA model (defaults to model)"
    )
    metadata_model: str | None = Field(
        default=None, description="Metadata model (defaults to model)"
    )
    chunk_overlap: int = Field(default=0, description="Chunk overlap for RAG")
    chunk_size: int = Field(default=128, description="Chunk size for RAG")
    split_separators: list[str] = Field(default=["\n"], description="Split separators")
    fp16_storage: bool = Field(default=True, description="Use FP16 storage for embeddings")
    l2_normalise: bool = Field(default=True, description="L2 normalize embeddings")
    top_k: int = Field(default=5, description="Top k passages for retrieval")


class ImageGenerationRequest(BaseModel):
    """Request model for image generation workflow."""

    input: str = Field(
        ..., min_length=5, max_length=1000, description="Statement or title to illustrate"
    )
    model: str = Field(..., description="Model for prompt generation")
    size: str = Field(default="1280x256", description="Image size WIDTHxHEIGHT")
    style: str = Field(
        default=(
            "The style is bold and energetic, using vibrant colors and dynamic compositions to create visually engaging scenes that emphasize activity, culture, and lively atmospheres. Digital art."
        ),
        description="Optional style appended to description",
    )


class ImageGenerationResponse(BaseModel):
    """Response model for image generation result."""

    image_description: str
    image_url: str | None
    size: str
    cost: float


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Verify the API key from the X-API-Key header."""
    if not x_api_key:
        raise HTTPException(
            status_code=401, detail="Missing API key. Please provide X-API-Key header."
        )

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return x_api_key


class ServerState:
    """Manages server state including Redis connection and worker pool."""

    def __init__(self):
        self.redis_client: redis.Redis | None = None
        self.active_jobs: set[str] = set()
        self.active_jobs_lock: asyncio.Lock | None = None
        self.worker_tasks: list[asyncio.Task] = []
        self.shutdown_event: asyncio.Event | None = None

    async def initialize(self):
        """Initialize Redis connection and start worker pool."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
            logger.info(f"Redis connection established: {REDIS_URL}")

            # Init concurrency primitives
            self.active_jobs_lock = asyncio.Lock()
            self.shutdown_event = asyncio.Event()

            # Start worker tasks
            for i in range(MAX_CONCURRENT_JOBS):
                task = asyncio.create_task(worker_loop(i))
                self.worker_tasks.append(task)
            logger.info(f"Started {MAX_CONCURRENT_JOBS} worker(s) for queue: {JOB_QUEUE_NAME}")

        except Exception as e:
            logger.error(f"Failed to initialize server state: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources and stop workers."""
        # Signal workers to stop
        if self.shutdown_event is not None:
            self.shutdown_event.set()

        # Cancel and await worker tasks
        for t in self.worker_tasks:
            t.cancel()
        if self.worker_tasks:
            try:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            except Exception:
                pass
        self.worker_tasks = []

        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")


# Global server state
server_state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    await server_state.initialize()

    logger.info(f"Server {SERVER_ID} started and ready to process fact-checking requests")

    yield

    # Shutdown
    logger.info(f"Server {SERVER_ID} shutting down")
    await server_state.cleanup()


# Initialize FastAPI app
app = FastAPI(
    title="Veridika Fact-Checking Server",
    description="A distributed fact-checking server that processes jobs from Redis queue",
    version="1.0.0",
    lifespan=lifespan,
)


async def enqueue_job(job: dict[str, Any]) -> None:
    """Push a job to the Redis queue."""
    if not server_state.redis_client:
        raise RuntimeError("Redis client not initialized")
    await server_state.redis_client.rpush(JOB_QUEUE_NAME, json.dumps(job))


async def worker_loop(worker_id: int):
    """Worker loop: pops jobs from Redis and processes them."""
    logger.info(f"Worker {worker_id} started")
    assert server_state.redis_client is not None
    assert server_state.active_jobs_lock is not None
    assert server_state.shutdown_event is not None

    try:
        while not server_state.shutdown_event.is_set():
            try:
                item = await server_state.redis_client.blpop(JOB_QUEUE_NAME, timeout=5)
                if item is None:
                    continue
                _, raw = item
                job: dict[str, Any] = json.loads(raw)
                job_id = job.get("job_id")

                async with server_state.active_jobs_lock:
                    server_state.active_jobs.add(job_id)

                try:
                    result_data = await execute_job(job)
                except Exception as e:
                    logger.exception(f"Worker {worker_id} failed job {job_id}")
                    result_data = {
                        "status": "completed",
                        "result": {"error": str(e), "success": False},
                        "job_id": job_id,
                    }

                # Store result
                await server_state.redis_client.setex(
                    f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps(result_data)
                )

                async with server_state.active_jobs_lock:
                    server_state.active_jobs.discard(job_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} loop error: {e}")
                await asyncio.sleep(1)
    finally:
        logger.info(f"Worker {worker_id} exiting")


async def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a single job and return the result payload to store."""
    job_type: str = job.get("type")
    job_id: str = job.get("job_id")
    data: dict[str, Any] = job.get("data", {})

    logger.info(f"Executing job {job_id} of type {job_type}")

    if job_type == "fact_check":
        # Instantiate workflow per job
        workflow = Workflow(config_path=WORKFLOW_CONFIG_PATH)
        result, _cost = await workflow.run(
            statement=data["article_topic"],
            internal_language=data["language"],
            output_language=data["language"],
            location=data["location"],
            generate_image=True,
            answer_questions=True,
        )
        return {"status": "completed", "result": result, "job_id": job_id}

    if job_type == "critical_questions":
        wf = CrititalQuestionWorkflow()
        result_dict, cost = await wf.run(
            statement=data["input"],
            model=data["model"],
            language=data["language"],
            location=data["location"],
        )
        return {
            "status": "completed",
            "result": {"questions": result_dict["questions"], "cost": cost},
            "job_id": job_id,
        }

    if job_type == "refine_questions":
        wf = CrititalQuestionRefinementWorkflow()
        result_dict, cost = await wf.run(
            questions=data["questions"],
            input=data["input"],
            refinement=data["refinement"],
            language=data["language"],
            location=data["location"],
            model=data["model"],
        )
        return {
            "status": "completed",
            "result": {"questions": result_dict["questions"], "cost": cost},
            "job_id": job_id,
        }

    if job_type == "search_sources":
        wf = GenSearchesWorkflow()
        sources, searches, cost = await wf.run(
            questions=data["questions"],
            statement=data["input"],
            language=data["language"],
            location=data["location"],
            model=data["model"],
            web_search_provider=data.get("web_search_provider", "serper"),
            top_k=data.get("top_k", 5),
            ban_domains=data.get("ban_domains"),
        )
        return {
            "status": "completed",
            "result": {"sources": sources, "searches": searches, "cost": cost},
            "job_id": job_id,
        }

    if job_type == "refine_sources":
        wf = SourceRefinementWorkflow()
        sources, searches, cost = await wf.run(
            current_searches=data["searches"],
            statement=data["input"],
            language=data["language"],
            location=data["location"],
            model=data["model"],
            user_feedback=data["refinement"],
            web_search_provider=data.get("web_search_provider", "serper"),
            top_k=data.get("top_k", 5),
            ban_domains=data.get("ban_domains"),
        )
        return {
            "status": "completed",
            "result": {"sources": sources, "searches": searches, "cost": cost},
            "job_id": job_id,
        }

    if job_type == "generate_article":
        wf = FactCheckingWorkflow()
        # Sources may already be dicts
        sources_list = data["sources"]
        article_writer_model = data.get("article_writer_model") or data["model"]
        question_answer_model = data.get("question_answer_model") or data["model"]
        metadata_model = data.get("metadata_model") or data["model"]

        response, cost = await wf.run(
            question=data["questions"],
            statement=data["input"],
            language=data["language"],
            location=data["location"],
            sources=sources_list,
            model=data["model"],
            use_rag=data.get("use_rag", True),
            embedding_model=data.get("embedding_model", "gemini-embedding-001"),
            article_writer_model=article_writer_model,
            question_answer_model=question_answer_model,
            metadata_model=metadata_model,
            chunk_overlap=data.get("chunk_overlap", 0),
            chunk_size=data.get("chunk_size", 128),
            split_separators=data.get("split_separators", ["\n"]),
            fp16_storage=data.get("fp16_storage", True),
            l2_normalise=data.get("l2_normalise", True),
            top_k=data.get("top_k", 5),
        )
        # Ensure the cost is included in the response returned to the user
        if isinstance(response, dict):
            result_payload = {**response, "cost": cost}
        else:
            result_payload = {"output": response, "cost": cost}
        return {"status": "completed", "result": result_payload, "job_id": job_id}

    if job_type == "generate_image":
        wf = ImageGenerationWorkflow()
        result_dict, cost = await wf.run(
            statement=data["input"],
            model=data["model"],
            size=data.get("size", "1280x256"),
            style=data.get(
                "style",
                "The style is bold and energetic, using vibrant colors and dynamic compositions to create visually engaging scenes that emphasize activity, culture, and lively atmospheres. Digital art.",
            ),
        )
        return {
            "status": "completed",
            "result": {**result_dict, "cost": cost},
            "job_id": job_id,
        }

    raise ValueError(f"Unknown job type: {job_type}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server_id": SERVER_ID,
        "is_processing": len(server_state.active_jobs) > 0,
        "current_job_id": None,
        "workflow_config": WORKFLOW_CONFIG_PATH,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_jobs_count": len(server_state.active_jobs),
    }


@app.get("/status")
async def get_status(api_key: str = Depends(verify_api_key)):
    """Get detailed server status."""
    queue_length = 0
    try:
        queue_length = await server_state.redis_client.llen(JOB_QUEUE_NAME)
    except Exception as e:
        logger.warning(f"Failed to get queue length: {e}")

    return {
        "server_id": SERVER_ID,
        "is_processing": len(server_state.active_jobs) > 0,
        "current_job_id": None,
        "workflow_config": WORKFLOW_CONFIG_PATH,
        "queue_length": queue_length,
        "redis_url": REDIS_URL,
        "job_queue_name": JOB_QUEUE_NAME,
        "result_queue_name": RESULT_QUEUE_NAME,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_jobs_count": len(server_state.active_jobs),
    }


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(
    request: FactCheckRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Submit a fact-checking job for processing."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received fact-check request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "fact_check",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.get("/result/{job_id}")
async def get_result(job_id: str, api_key: str = Depends(verify_api_key)):
    """Get the result of a fact-checking job."""
    job_data = await server_state.redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = json.loads(job_data.decode("utf-8"))

    # If there's an error, return it with a 400 status code
    if job_info.get("status") == "error":
        raise HTTPException(status_code=400, detail=job_info)

    return job_info


@app.post("/process_direct")
async def process_direct(job_request: FactCheckRequest, api_key: str = Depends(verify_api_key)):
    """Process a job directly without using Redis (for testing)."""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Execute synchronously using the same execution path
        result_data = await execute_job(
            {
                "job_id": job_id,
                "type": "fact_check",
                "data": job_request.model_dump(),
            }
        )

        # Store result for consistency with queued jobs
        await server_state.redis_client.setex(
            f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps(result_data)
        )

        return result_data

    except Exception as e:
        logger.error(f"Failed to process direct job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process job: {str(e)}")


# ============================================================================
# STEPWISE WORKFLOW ENDPOINTS
# ============================================================================


@app.post("/stepwise/critical-questions", response_model=FactCheckResponse)
async def generate_critical_questions(
    request: CriticalQuestionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate critical questions for a statement."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received critical question generation request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "critical_questions",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.post("/stepwise/refine-questions", response_model=FactCheckResponse)
async def refine_critical_questions(
    request: CriticalQuestionRefinementRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Refine critical questions with user feedback."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received critical question refinement request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "refine_questions",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.post("/stepwise/search-sources", response_model=FactCheckResponse)
async def search_sources(
    request: SourceSearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate searches and fetch sources."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received source search request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "search_sources",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.post("/stepwise/refine-sources", response_model=FactCheckResponse)
async def refine_sources(
    request: SourceRefinementRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Refine sources with user feedback."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received source refinement request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "refine_sources",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.post("/stepwise/generate-article", response_model=FactCheckResponse)
async def generate_fact_check_article(
    request: FinalFactCheckRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate final fact-checking article with Q&A and metadata."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received final fact-checking request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "generate_article",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


@app.post("/stepwise/generate-image", response_model=FactCheckResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate an image from an input statement using prompt + image generator."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received image generation request. Job ID: {job_id}")

    # Store initial job state in Redis
    await server_state.redis_client.setex(
        f"job:{job_id}", REDIS_JOB_EXPIRY, json.dumps({"status": "in_progress", "result": None})
    )

    # Enqueue job
    await enqueue_job(
        {
            "job_id": job_id,
            "type": "generate_image",
            "data": request.model_dump(),
        }
    )

    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Veridika server {SERVER_ID}")
    logger.info(f"Workflow config: {WORKFLOW_CONFIG_PATH}")
    logger.info(f"Redis URL: {REDIS_URL}")
    logger.info(f"Job queue: {JOB_QUEUE_NAME}")
    logger.info(f"Result queue: {RESULT_QUEUE_NAME}")
    logger.info("API key authentication: enabled")

    uvicorn.run(
        "veridika_server:app",
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=False,
    )
