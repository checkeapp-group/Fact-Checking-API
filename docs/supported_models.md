# Supported Models

The fact-cheking pipeline support both, API models (via [OpenRouter](https://openrouter.ai)) and Local models (via any OpenAI compatible server such as [vLLM](https://github.com/vllm-project/vllm)),

We support the following models:
- Any OpenRouter model: Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro... (Requires an API Key with pre-paid credits).
- Any OpenAI compatible API server (Local vLLM models)
- Gemini Embeddings (via [google.genai](https://github.com/googleapis/python-genai) API, requires an API Key with pre-paid credits)
- OpenAI Embeddings (via [openai](https://platform.openai.com/docs/guides/embeddings) API, requires an API Key with pre-paid credits)
- Flux Image Generation Models via [Replicate API](https://replicate.com/collections/flux) (requires an API Key with pre-paid credits)
- Flux Image Generation Models via a local/runpod [ComfyUI](https://www.comfy.org/) server.


## Open Router Models

You can use any model from [OpenRouter](https://openrouter.ai/models) by specifying the model name in the pipeline configuration file or agent calls. Unless defined in the local model or embedding configuration, the pipeline will always attempt to use the OpenRouter API to access the model. Thefore you can use any model from OpenRouter without needing to do any change to the code. We will automatically use the OpenRouter API to gather the pricing information for the model and calculate the cost of the API calls. 

In order to use OpenRouter models, you need to set the `OPENROUTER_API_KEY` variable into your `.env` file. OpenRouter requires you to prepay for the credits you want to use. You can find more information about the pricing [here](https://openrouter.ai/pricing).

Example:
```python
critical_questions, cost = await critical_question_workflow.run(
    statement=statement,
    language=language,
    location=location,
    model="google/gemini-2.5-flash",
)

critical_questions, cost = await critical_question_workflow.run(
    statement=statement,
    language=language,
    location=location,
    model="openai/gpt-5.2-chat",
)

critical_questions, cost = await critical_question_workflow.run(
    statement=statement,
    language=language,
    location=location,
    model="anthropic/claude-sonnet-4.5",
)
```

## Embedding Models

We support the OpenAI and Gemini embedding models via their respective APIs (You can also use locally hosted embeddings, see the Local Model section for details). To use them you must set the following environment variables:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

You can use these model by using the following model names:
- `gemini-embedding-001`
- `text-embedding-3-large`

## Image Models

The default way to generate images for articles is using the [Replicate API](https://replicate.com/collections/flux). You will need to set the `REPLICATE_API_TOKEN` variable into your `.env` file, as well as adding pre-paid credits to your account. 

We also support local image generation with Flux using a local [ComfyUI](https://www.comfy.org/) server. You can use `flux-local` as the image model name and configure the local model in the `configs/local-model-config/flux-config.json` file (See the Local Model section for instructions). However we did not test this feature extensively and setting up a ComfyUI server requires some effort. We recommend using the Replicate API for image generation for a simple setup, it costs 3$ for 1000 images. 


## Self-Hosted / Local Models

We support any text generation and embedding generation model that is compatible with the OpenAI API. In this section, we will explain how to use [vLLM](https://github.com/vllm-project/vllm) to host a `Latxa8B` model locally and use it in the pipeline. You can use any other OpenAI compatible serving platform, such as [SGLang](https://github.com/sgl-project/sglang) or even hosting services such as [RunPod](https://runpod.io/). Any server that follows the OpenAI API specification will work as long as the IP/URL of the machine serving the model is accesible from the machine that is running the fact-checking pipeline.

First you need to install vLLM in the host machine with `pip install vllm`. Then you need to start the vLLM server with the following command (See the vLLM docs for more info: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/). 

### Launch a Latxa8B server 

```bash
vllm serve HiTZ/Latxa-Llama-3.1-8B-Instruct \
--api-key test \
--quantization fp8 \
--gpu-memory-utilization 0.95 \
--max_model_len 8192
```

### Launch a Latxa8B server + embedding server in the same GPU

If you want to launch two models in the same GPU, ajust the `gpu-memory-utilization` parameter to split the GPU memory between the two models. You should also use different ports for each model.

```bash
vllm serve HiTZ/Latxa-Llama-3.1-8B-Instruct \
--api-key test \
--quantization fp8 \
--gpu-memory-utilization 0.55 \
--max_model_len 8192 \
--port 8000

vllm serve Qwen/Qwen3-Embedding-0.6B \
--api-key test \
--quantization fp8 \
--gpu-memory-utilization 0.4 \
--max_model_len 8192 \
--port 8001
```

### Register the model

After you have started the vLLM server, you need to register the model. You can do this by adding the edditing the `configs/local-model-config/*.json` files.

- `configs/local-model-config/embedding-config.json`: Defines self-host embedding models
- `configs/local-model-config/model-config.json`: Defines self-host text generation models
- `configs/local-model-config/flux-config.json`: Defines self-host Flux image generation models


For example, if you have a vLLM server running at `http://localhost:8000` and you want to use the `Latxa8B` model, you can add the following to the `configs/local-model-config/model-config.json` file:

```json
{
    "models": {
        "Latxa8B-local": {
            "model_name": "HiTZ/Latxa-Llama-3.1-8B-Instruct",
            "api_key_name": "LocalModel_API_KEY",
            "vllm_url": "LocalModel_URL"
        }
    }
}
```

**Important**: The `api_key_name` and `vllm_url` parameters define the name of the environment variable that contains the API key and the URL of the vLLM server, respectively. We do this to prevent that the URL or the API keys to be exposed in the code and accidentally leaked. 

So if you have a vLLM server running at `http://localhost:8000` with the API key `test`, now you need to add the following to the `.env` file:

```bash
LocalModel_API_KEY=test
LocalModel_URL=http://0.0.0.0:8000/v1
```


We can do the same for the self-host embedding model. If you have a vLLM server running at `http://localhost:8001` and you want to use the `Qwen3-Embedding-0.6B` model, you can add the following to the `configs/local-model-config/embedding-config.json` file:

```json
{
    "models": {
        "Qwen3-Embedding-0.6B-local": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "api_key_name": "Local_API_KEY",
            "vllm_url": "LocalEmbedding_URL"
        }
    }
}
```

Then in our `.env` file we need to add the following:

```bash
LocalEmbedding_API_KEY=test
LocalEmbedding_URL=http://0.0.0.0:8001/v1
```


Now you can use the models in the pipeline by using the model name `Latxa8B-local` and `Qwen3-Embedding-0.6B-local`.

```python
result, cost = await article_fact_checking_workflow.run(
    question=new_critical_questions["questions"],
    statement=statement,
    language=language,
    location=location,
    sources=sources,
    model="Latxa8B-local",
    use_rag=True,
    embedding_model="Qwen3-Embedding-0.6B-local",
    article_writer_model="Latxa8B-local",
    question_answer_model="Latxa8B-local",
    metadata_model="Latxa8B-local",
)
```

You an also use these models in pipeline configuration `.yaml` files, see `configs/pipeline_configs/latxa8b_rag.yaml` for an example. 
