# CheckApp Fact-Checking System

CheckApp is an AI-powered fact-checking system that analyzes claims and generates detailed fact-checking articles with citations from web sources.

## Features

- **Server Mode**: Deploy as a REST API server with job queue management
- **Offline Inference**: Run fact-checks directly from Python notebooks or scripts
- **Evaluation Framework**: Benchmark performance of LLMs using the [FactCheckingEval](https://huggingface.co/datasets/Iker/FactCheckingEval) dataset
- **Multiple Model Support**: Works with OpenAI, Google Gemini, OpenRouter, and self-hosted models
- **RAG Integration**: Retrieval-Augmented Generation for improved source analysis

## Example Notebooks

- [`notebooks/run_workflow.ipynb`](notebooks/run_workflow.ipynb) - Run a complete fact-check in one step
- [`notebooks/test_stepwiseworkflow.ipynb`](notebooks/test_stepwiseworkflow.ipynb) - Run the workflow step-by-step with customization

## Documentation

| Document | Description |
|----------|-------------|
| [Offline Fact-Checking](docs/offline-fact-checking.md) | Guide for running fact-checks locally using Python notebooks |
| [Build and Run Docker Server](docs/build-and-run-docker-server.md) | Deploy the fact-checking API server using Docker |
| [Server Specification](docs/server-specification.md) | API endpoints, request/response formats, and usage examples |
| [Fact-Checking Workflow](docs/fact-checking-workflow-description.md) | Detailed explanation of each step in the fact-checking pipeline |
| [Supported Models](docs/supported_models.md) | List of supported AI models and configuration options |
| [Running Evaluations](docs/run-evaluation.md) | Benchmark LLMs using the evaluation framework |
| [Evaluation Results](docs/evaluation-results.md) | Performance benchmarks and comparison of different models |
| [Model Quantization](docs/model-quantization.md) | Guide for model quantization setup |
