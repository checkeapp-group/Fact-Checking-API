# Offline Fact-Checking Guide

This guide explains how to run the fact-checking workflow offline using the Python notebooks provided in this repository. You can either run the full workflow in one step or execute it step by step for more control and customization.

## Prerequisites

### 1. Python Environment

This project requires Python 3.12+. We recommend using `uv` for dependency management, but you can also use `pip`.

### 2. Install Dependencies

**Option A: Using `uv` (Recommended)**

You can install uv using pip:

```bash
pip install uv
```

Or from with the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment:
```bash
uv sync

source .venv/bin/activate
```

This will create a virtual environment and install all dependencies automatically.

**Option B: Using `pip`**

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the required API keys. Here is what you need:

```bash
# Provider keys (required for using the models supported by these APIs)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...

# Web search API (required for searching sources)
SERPER_API_KEY=...

# Image generation (optional, required only for generating images)
REPLICATE_API_TOKEN=r8_...

# Self-hosted models (optional, if you want to use self-hosted/local models)
LocalModel_API_KEY=test
LocalModel_URL=http://0.0.0.0:8000/v1
```

See [supported models](supported_models.md) for more information.

> IMPORTANT!
> The `SERPER_API_KEY` is always required, as it enables web search functionality for finding sources.

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPER_API_KEY` | ✅ | Web search API key (from [serper.dev](https://serper.dev)) |
| `OPENAI_API_KEY` | ❌ | OpenAI API key for embeddings and models |
| `GEMINI_API_KEY` | ❌ | Google Gemini API key |
| `OPENROUTER_API_KEY` | ❌ | OpenRouter API key for accessing multiple AI models |
| `REPLICATE_API_TOKEN` | ❌ | Required only if generating images |


## Running the Full Workflow in One Step

The simplest way to run a fact-check is to use the `Workflow` class which executes all steps automatically with a single configuration file.

### Example Notebook

See [notebooks/run_workflow.ipynb](../notebooks/run_workflow.ipynb) for a complete example.

### Quick Start

```python
from dotenv import load_dotenv
from veridika.src.workflows import Workflow

# Load environment variables from .env file
load_dotenv(".env")

# Initialize the workflow with a configuration file
workflow = Workflow(config_path="configs/pipeline_configs/gemini_rag.yaml")

# Run the fact-check
result, cost = await workflow.run(
    statement="Los coches eléctricos son más contaminantes que los coches de gasolina",
    internal_language="es",
    output_language="es",
    location="es",
    generate_image=True,
    do_rag=True,
)

# Print results
print(f"📊 Workflow completed in {workflow.get_runtime()} seconds")
print(f"💰 Total cost: ${cost:.4f}")
print(f"📝 Question: {result['question']}")
print(f"✅ Label: {result['metadata']['label']}")
print(f"📚 Sources used: {len(result['sources'])}")
print(f"❓ Related questions: {len(result['related_questions'])}")
print(result["answer"])
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `statement` | `str` | The claim or statement to fact-check |
| `internal_language` | `str` | Language for internal processing (e.g., "es", "en") |
| `output_language` | `str` | Language for the output results |
| `location` | `str` | Geographic location context for web searches |
| `generate_image` | `bool` | Whether to generate an illustrative image |
| `do_rag` | `bool` | Whether to use Retrieval-Augmented Generation |

### Configuration Files

Configuration files are located in `configs/pipeline_configs/`. The default configuration `gemini_rag.yaml` uses Google Gemini models with RAG enabled. You can create custom configurations based on your needs. See [The Fact Checking Workflow Description](fact-checking-workflow-description.md) for more information.

## Running the Workflow Step by Step

For more control over the fact-checking process, you can run each step individually. This allows you to:

- Refine critical questions based on user feedback
- Add additional search sources
- Customize the article generation

### Example Notebook

See [notebooks/test_stepwiseworkflow.ipynb](../notebooks/test_stepwiseworkflow.ipynb) for a complete example with all steps.

### Step 1: Generate Critical Questions

The first step generates critical questions to verify the statement.

```python
import sys
sys.path.append("..")

from dotenv import load_dotenv
load_dotenv(".env")

import json
from veridika.src.workflows.StepwiseWorkflows import CrititalQuestionWorkflow

# Define the statement to fact-check
statement = "Los coches eléctricos son más contaminantes que los coches de gasolina"
language = "es"
location = "es"

# Generate critical questions
critical_question_workflow = CrititalQuestionWorkflow()
critical_questions, cost = await critical_question_workflow.run(
    statement=statement,
    language=language,
    location=location,
    model="google/gemini-2.5-flash",
)

print(json.dumps(critical_questions, indent=4, ensure_ascii=False))
print(f"💰 Step cost: {cost}")
```

### Step 1b (Optional): Refine Critical Questions

You can refine the generated questions based on specific requirements:

```python
from veridika.src.workflows.StepwiseWorkflows import CrititalQuestionRefinementWorkflow

critial_question_refinement_workflow = CrititalQuestionRefinementWorkflow()
refinement = "También quiero que consideres la contaminación del proceso de reciclaje de la batería"

new_critical_questions, cost = await critial_question_refinement_workflow.run(
    questions=critical_questions["questions"],
    input=statement,
    refinement=refinement,
    language=language,
    location=location,
    model="google/gemini-2.5-flash",
)

print(json.dumps(new_critical_questions, indent=4, ensure_ascii=False))
print(f"💰 Step cost: {cost}")
```

### Step 2: Web Search

Search the web for sources related to the critical questions:

```python
from veridika.src.workflows.StepwiseWorkflows import GenSearchesWorkflow

gen_searches_workflow = GenSearchesWorkflow()
sources, searches, cost = await gen_searches_workflow.run(
    statement=statement,
    questions=new_critical_questions["questions"],
    language=language,
    location=location,
    model="google/gemini-2.5-flash",
)

# Display the search queries used
for search in searches:
    print(search)

print(f"\n📚 Found {len(sources)} sources")
print(f"💰 Step cost: {cost}")
```

### Step 2b (Optional): Refine Sources

You can add additional sources based on user feedback:

```python
from veridika.src.workflows.StepwiseWorkflows import SourceRefinementWorkflow

source_refinement_workflow = SourceRefinementWorkflow()
user_feedback = "Quiero que también busques información oficial de la Unión Europea"

new_sources, new_searches, cost = await source_refinement_workflow.run(
    statement=statement,
    current_searches=searches,
    language=language,
    location=location,
    model="google/gemini-2.5-flash",
    user_feedback=user_feedback,
)

# Combine all sources
all_sources = sources + new_sources
all_searches = searches + new_searches

print(f"📚 Added {len(new_sources)} new sources")
print(f"💰 Step cost: {cost}")
```

### Step 3: Generate Fact-Checking Article

Generate the fact-checking article with the gathered sources:

```python
from veridika.src.workflows.StepwiseWorkflows import FactCheckingWorkflow

article_fact_checking_workflow = FactCheckingWorkflow()
result, cost = await article_fact_checking_workflow.run(
    question=new_critical_questions["questions"],
    statement=statement,
    language=language,
    location=location,
    sources=all_sources,
    model="google/gemini-2.5-flash",
    use_rag=True,
    embedding_model="gemini-embedding-001",
    article_writer_model="google/gemini-2.5-flash",
    question_answer_model="google/gemini-2.5-flash",
    metadata_model="google/gemini-2.5-flash",
)

print(f"✅ Label: {result['metadata']['label']}")
print(f"📚 Sources used: {len(result['sources'])}")
print(f"📊 Metadata: {json.dumps(result['metadata'], indent=4, ensure_ascii=False)}")
print(f"💰 Step cost: {cost}")
print("\n" + result["answer"])
```

### Step 4 (Optional): Generate Image

Generate an illustrative image for the fact-check:

```python
from veridika.src.workflows.StepwiseWorkflows import ImageGenerationWorkflow

image_generation_workflow = ImageGenerationWorkflow()
image_result, cost = await image_generation_workflow.run(
    statement=statement,
    model="google/gemini-2.5-flash",
)

print(f"💰 Step cost: {cost}")
print(f"🖼️ Image URL: {image_result.get('image_url', 'No image generated')}")
```

## Output Structure

The workflow returns a result dictionary with the following structure:

```python
{
    "question": "Original statement",
    "answer": "Detailed fact-checking article with citations",
    "metadata": {
        "title": "Article title",
        "categories": ["Category1", "Category2"],
        "label": "Fake" | "True" | "Misleading" | "Unverifiable",
        "main_claim": "Summary of the verdict"
    },
    "sources": [
        {
            "link": "https://...",
            "title": "Source title",
            "snippet": "Brief excerpt",
            "favicon": "https://..."
        },
        ...
    ],
    "related_questions": ["Question 1", "Question 2", ...],
    "image": {
        "url": "https://..."  # If image generation was enabled
    }
}
```

## Supported Models

The workflow supports multiple AI models. See [supported_models.md](supported_models.md) for a complete list of supported models and their configurations.
