# Running Evaluations

This guide explains how to run evaluations on the fact-checking system using the `Evaluator` class. The evaluator runs the fact-checking workflow against the [FactCheckingEval](https://huggingface.co/datasets/Iker/FactCheckingEval) dataset and computes performance metrics.

## Prerequisites

Before running evaluations, ensure you have:

1. Installed all dependencies (see [Offline Fact-Checking Guide](offline-fact-checking.md#install-dependencies))
2. Configured the required environment variables in your `.env` file
3. Access to the Hugging Face datasets (the evaluator automatically downloads the dataset)

## Quick Start

Run an evaluation from the command line:

```bash
python -m veridika.src.evaluator.evaluator --config configs/evaluator_configs/pipeline_gemini_rag.yaml --max_workers 4
```

## Evaluator Configuration

The evaluator uses a YAML configuration file with three main sections:

1. **Evaluator parameters** - Output directory and restart behavior
2. **Run arguments** - Parameters passed to the workflow
3. **Workflow configuration** - Reference to the pipeline configuration

### Example Configuration

```yaml
# Evaluator parameters
output_dir: ./evaluation_results/gemini_rag
overwrite_output_dir: False

run_arguments:
  internal_language: es
  output_language: es
  location: es
  generate_image: False
  answer_questions: False

workflow_config: ./configs/pipeline_configs/gemini_rag.yaml
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_dir` | `string` | Directory where evaluation results will be saved (required) |
| `overwrite_output_dir` | `boolean` | If `true`, deletes existing output directory before starting. If `false`, resumes from previous progress |
| `run_arguments` | `object` | Arguments passed to `workflow.run()` (see below) |
| `workflow_config` | `string` | Path to the workflow/pipeline configuration file |

### Run Arguments

These are passed directly to the workflow's `run()` method:

| Argument | Type | Description |
|----------|------|-------------|
| `internal_language` | `string` | Language for internal processing (e.g., "es", "en") |
| `output_language` | `string` | Language for the output results |
| `location` | `string` | Geographic location context for web searches |
| `generate_image` | `boolean` | Whether to generate illustrative images (usually `false` for evaluations) |
| `answer_questions` | `boolean` | Whether to generate related questions |

## Creating a Custom Evaluator Configuration

### Step 1: Choose or Create a Workflow Configuration

The evaluator references a workflow/pipeline configuration file. You can use an existing one from `configs/pipeline_configs/` or create your own. See [fact-checking-workflow-description.md](fact-checking-workflow-description.md) for details on workflow configuration.

### Step 2: Create the Evaluator Config File

Create a new YAML file in `configs/evaluator_configs/`:

```yaml
# configs/evaluator_configs/my_custom_evaluation.yaml

# Where to save results
output_dir: ./evaluation_results/my_custom_evaluation

# Set to True to start fresh, False to resume from previous runs
overwrite_output_dir: False

# Arguments for the fact-checking workflow
run_arguments:
  internal_language: en    # Use English for processing
  output_language: en      # Output in English
  location: us             # US-based web search context
  generate_image: False    # Skip image generation for faster evaluation
  answer_questions: False  # Skip related question generation

# Path to your workflow configuration
workflow_config: ./configs/pipeline_configs/my_custom_pipeline.yaml
```

### Step 3: Run the Evaluation

```bash
python -m veridika.src.evaluator.evaluator \
  --config configs/evaluator_configs/my_custom_evaluation.yaml \
  --max_workers 4
```

Max workers is the number of parallel fact-checking workflows that will be run. Multi parallel workflows will speed up the evaluation process, but you might hit rate limits in the APIs.

Evaluation will take between 1 hour and 3 hours to complete depending on the workflow configuration and model. For the cost and results of the evaluations we have already run, see the [evaluation results](evaluation_results.md).

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `string` | Required | Path to the evaluator configuration file |
| `--max_workers` | `int` | `4` | Number of parallel workers for processing examples |

## Using the Evaluator Programmatically

You can also use the `Evaluator` class directly in Python:

```python
from veridika.src.evaluator.evaluator import Evaluator

# Initialize with a config file
evaluator = Evaluator(config_path="configs/evaluator_configs/pipeline_gemini_rag.yaml")

# Run the evaluation with 4 parallel workers
evaluator(max_workers=4)
```

Or with a configuration dictionary:

```python
from veridika.src.evaluator.evaluator import Evaluator

config = {
    "output_dir": "./evaluation_results/my_test",
    "overwrite_output_dir": True,
    "run_arguments": {
        "internal_language": "es",
        "output_language": "es",
        "location": "es",
        "generate_image": False,
        "answer_questions": False,
    },
    "workflow_config": {
        # Inline workflow configuration (alternative to file reference)
        "workflow_type": "FactCheckingWithPipelineWorkflow",
        "models": {
            "critical_question_agent": "google/gemini-2.5-flash",
            "gen_searches_agent": "google/gemini-2.5-flash",
            # ... other model configurations
        },
        # ... other workflow settings
    }
}

evaluator = Evaluator(config=config)
evaluator(max_workers=4)
```

## Output Structure

The evaluator creates the following files in the output directory:

```
evaluation_results/gemini_rag/
├── configuration.json      # Copy of the evaluation configuration
├── evaluation_summary.json # Overall metrics and cost tracking
├── statement_results.json  # Per-statement results (created after evaluation)
├── example_id_1.json       # Individual result for each example
├── example_id_2.json
└── ...
```

### evaluation_summary.json

Contains aggregate metrics after evaluation completes:

```json
{
    "total_cost": 12.3456,
    "start_date": "2025-12-17T10:00:00",
    "end_date": "2025-12-17T14:30:00",
    "results": {
        "splits": {
            "test": {
                "TP": 45,
                "FP": 5,
                "FN": 8,
                "TN": 42,
                "Undetermined": 0,
                "accuracy": 87.0,
                "precision": 90.0,
                "recall": 84.91,
                "f1_score": 87.37,
                "undetermined_rate": 0.0
            }
        },
        "full": {
            "TP": 45,
            "FP": 5,
            "FN": 8,
            "TN": 42,
            "Undetermined": 0,
            "accuracy": 87.0,
            "precision": 90.0,
            "recall": 84.91,
            "f1_score": 87.37,
            "undetermined_rate": 0.0
        }
    }
}
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **TP (True Positives)** | Correctly identified true statements |
| **TN (True Negatives)** | Correctly identified false statements |
| **FP (False Positives)** | False statements incorrectly labeled as true |
| **FN (False Negatives)** | True statements incorrectly labeled as false |
| **Undetermined** | Statements for which the system couldn't get enough information to classify into True or False (You can force the system to always return a classification even if it doesn't have enough confidence by removing the `Undetermined` label from the Metadata Agent in in `veridika/src/agents/MetadataAgent.py`) |
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN) * 100` |
| **Precision** | `TP / (TP + FP) * 100` |
| **Recall** | `TP / (TP + FN) * 100` |
| **F1 Score** | `2 * (Precision * Recall) / (Precision + Recall)` |
| **Undetermined Rate** | `Undetermined / Total * 100` |

## Resuming Evaluations

One of the key features of the evaluator is the ability to resume from where it left off:

1. **Automatic checkpointing**: Each processed example is saved immediately to disk
2. **Configuration validation**: When resuming, the evaluator verifies the configuration matches the original run
3. **Progress tracking**: Already processed examples are skipped automatically

To resume an interrupted evaluation, simply run the same command again with `overwrite_output_dir: False`:

```bash
python -m veridika.src.evaluator.evaluator --config configs/evaluator_configs/pipeline_gemini_rag.yaml
```

> **Important**: If you want to start a fresh evaluation, set `overwrite_output_dir: True` in your config or the evaluator will try to resume from existing results.

## Available Evaluator Configurations

The repository includes several pre-configured evaluator setups in `configs/evaluator_configs/`:

| Configuration | Description |
|---------------|-------------|
| `pipeline_gemini_rag.yaml` | Google Gemini models with RAG enabled |
| `pipeline_gemini_full.yaml` | Google Gemini models with full context |
| `pipeline_gpt_rag.yaml` | OpenAI GPT models with RAG enabled |
| `pipeline_gpt_full.yaml` | OpenAI GPT models with full context |
| `pipeline_grok_rag.yaml` | Grok models with RAG enabled |
| `pipeline_latxa8b_rag.yaml` | Latxa 8B model with RAG |
| `pipeline_latxa8b-4bit_rag.yaml` | Latxa 8B 4-bit quantized with RAG |
| `pipeline_latxa70b-4bit_rag.yaml` | Latxa 70B 4-bit quantized with RAG |

## Domain Banning

The evaluator automatically bans domains from the evaluation dataset to prevent data leakage. This means:

- Source domains from the evaluation dataset are added to the web search ban list. This is, the sources from wich the statements are taken are banned as sources for the fact-checking process.
- Default banned domains (defined in `veridika/src/web_search/default_ban_domains.py`) are also included
- This ensures the model cannot simply retrieve the original fact-check articles

