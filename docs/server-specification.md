# Veridika Fact-Checking Server API Documentation

A FastAPI server that processes fact-checking jobs. The server supports both full end-to-end fact-checking and step-by-step processing.

## Table of Contents
- [Authentication](#authentication)
- [Server Endpoints](#server-endpoints)
  - [Health Check](#health-check)
  - [Status](#status)
  - [Get Job Result](#get-job-result)
- [Full Fact-Checking Endpoints](#full-fact-checking-endpoints)
  - [Submit Fact-Check Job](#submit-fact-check-job)
- [Stepwise Workflow Endpoints](#stepwise-workflow-endpoints)
  - [1. Generate Critical Questions](#1-generate-critical-questions)
  - [2. Refine Critical Questions](#2-refine-critical-questions)
  - [3. Search Sources](#3-search-sources)
  - [4. Refine Sources](#4-refine-sources)
  - [5. Generate Final Article](#5-generate-final-article)
  - [6. Generate Image](#6-generate-image)

---

## Authentication

All endpoints (except `/health`) require API key authentication via the `X-API-Key` header.

```bash
export API_KEY="your-secret-api-key-here"
```

---

## Server Endpoints

### Health Check

Check if the server is running and get basic information.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "server_id": "veridika_server_abc123",
  "is_processing": false,
  "current_job_id": null,
  "workflow_config": "configs/pipeline_configs/gemini.yaml",
  "max_concurrent_jobs": 4,
  "active_jobs_count": 0
}
```

**Example:**
```bash
curl -X GET http://localhost:7860/health
```

---

### Status

Get detailed server status including queue information.

**Endpoint:** `GET /status`

**Authentication:** Required

**Response:**
```json
{
  "server_id": "veridika_server_abc123",
  "is_processing": false,
  "current_job_id": null,
  "workflow_config": "configs/pipeline_configs/gemini.yaml",
  "queue_length": 0,
  "redis_url": "redis://localhost:6379",
  "job_queue_name": "veridika_jobs",
  "result_queue_name": "veridika_results",
  "max_concurrent_jobs": 4,
  "active_jobs_count": 0
}
```

**Example:**
```bash
curl -X GET http://localhost:7860/status \
  -H "X-API-Key: your-secret-api-key-here"
```

---

### Get Job Result

Retrieve the result of a submitted job.

**Endpoint:** `GET /result/{job_id}`

**Authentication:** Required

**Response (In Progress):**
```json
{
  "status": "in_progress",
  "result": null
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "result": {
    "questions": ["Question 1", "Question 2"],
    "cost": 0.0019304
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X GET http://localhost:7860/result/123e4567-e89b-12d3-a456-426614174000 \
  -H "X-API-Key: your-secret-api-key-here"
```

---

## Full Fact-Checking Endpoints

### Submit Fact-Check Job (Veridika.ai current workflow)

Submit a complete fact-checking job that runs the full workflow.

**Endpoint:** `POST /fact-check`

**Authentication:** Required

**Request Body:**
```json
{
  "article_topic": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "language": "es",
  "location": "es",
  "config": "unused"
}
```

**Parameters:**
- `article_topic` (string, required): The statement to fact-check (10-1000 characters)
- `language` (string, required): Language code (e.g., "es", "en")
- `location` (string, required): Location code for web search (1-2 characters)
- `config` (string, required): Legacy parameter, kept for compatibility (can be any value)

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/fact-check \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "article_topic": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "language": "es",
    "location": "es",
    "config": "unused"
  }'
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "question": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "answer": "Full fact-checking article...",
    "metadata": {
      "title": "Los coches eléctricos son más contaminantes que los coches de gasolina",
      "label": "Fake",
      "categories": ["Automoción", "Medio ambiente", "Ciencia"],
      "main_claim": "..."
    },
    "sources": [...],
    "related_questions": [...],
    "image": {
      "url": "https://...",
      "description": "..."
    },
    "logs": {...}
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

## Stepwise Workflow Endpoints

These endpoints allow you to run the fact-checking process step-by-step with user interaction between each stage.

### 1. Generate Critical Questions

Generate critical research questions for a given statement.

**Endpoint:** `POST /stepwise/critical-questions`

**Authentication:** Required

**Request Body:**
```json
{
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "model": "google/gemini-2.5-flash",
  "language": "es",
  "location": "es"
}
```

**Parameters:**
- `input` (string, required): The statement to analyze (10-1000 characters)
- `model` (string, required): Model name to use (e.g., "google/gemini-2.5-flash")
- `language` (string, required): Language code (e.g., "es", "en"... google api lang codes)
- `location` (string, required): Location code (e.g, "es", "en"... google api lang codes)

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "questions": [
      "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico fabricado en 2025?",
      "¿Qué tipo de emisiones producen los coches de gasolina en comparación con los eléctricos?",
      "¿Qué impacto ambiental tiene la extracción de materias primas para las baterías?"
    ],
    "cost": 0.0019304
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/critical-questions \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "model": "google/gemini-2.5-flash",
    "language": "es",
    "location": "es"
  }'
```

---

### 2. Refine Critical Questions

Refine the generated questions based on user feedback. This endpoint returns the updated list of questions that replaces the original one. 

**Endpoint:** `POST /stepwise/refine-questions`

**Authentication:** Required

**Request Body:**
```json
{
  "questions": [
    "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
    "¿Qué tipo de emisiones producen los coches de gasolina?"
  ],
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "refinement": "También quiero que consideres la contaminación del proceso de reciclaje de la batería",
  "language": "es",
  "location": "es",
  "model": "google/gemini-2.5-flash"
}
```

**Parameters:**
- `questions` (array of strings, required): Current list of critical questions
- `input` (string, required): Original statement 
- `refinement` (string, required): User feedback for refinement
- `language` (string, required): Language code
- `location` (string, required): Location code
- `model` (string, required): Model name to use

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "questions": [
      "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
      "¿Qué tipo de emisiones producen los coches de gasolina?",
      "¿Cuál es el impacto ambiental del proceso de reciclaje de las baterías?"
    ],
    "cost": 0.0012145
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/refine-questions \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
      "¿Qué tipo de emisiones producen los coches de gasolina?"
    ],
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "refinement": "También quiero que consideres la contaminación del proceso de reciclaje de la batería",
    "language": "es",
    "location": "es",
    "model": "google/gemini-2.5-flash"
  }'
```

---

### 3. Search Sources

Generate search queries from questions and fetch web sources. This endpoint gets the final critical questions and returns a list of sources that answer those questions. The sources contain 'snippets' from the full text. The full-text is still not downloaded at this step. 

**Endpoint:** `POST /stepwise/search-sources`

**Authentication:** Required

**Request Body:**
```json
{
  "questions": [
    "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
    "¿Qué tipo de emisiones producen los coches de gasolina?"
  ],
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "language": "es",
  "location": "es",
  "model": "google/gemini-2.5-flash",
  "web_search_provider": "serper",
  "ban_domains": null
}
```

**Parameters:**
- `questions` (array of strings, required): List of critical questions
- `input` (string, required): Original statement
- `language` (string, required): Language code
- `location` (string, required): Location code
- `model` (string, required): Model name to use
- `web_search_provider` (string, optional): Web search provider (default: "serper")
- `ban_domains` (array of strings, optional): Domains to exclude from results

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "sources": [
      {
        "title": "Comparativa de emisiones de CO2 entre coche eléctrico y gasolina",
        "link": "https://effimove.com/emisiones-de-co2-en-la-vida-del-coche-electricos-vs-combustion/",
        "snippet": "En este estudio se considera que las emisiones se sitúan entre 61 y 106 kg de CO2 por kWh...",
        "favicon": "https://effimove.com/favicon.ico",
        "base_url": "https://effimove.com"
      }
    ],
    "searches": [
      "huella carbono ciclo vida coche eléctrico 2025 batería vs gasolina estudios",
      "emisiones gases efecto invernadero coches eléctricos vs gasolina 2025"
    ],
    "cost": 0.0056027
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/search-sources \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
      "¿Qué tipo de emisiones producen los coches de gasolina?"
    ],
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "language": "es",
    "location": "es",
    "model": "google/gemini-2.5-flash"
  }'
```

---

### 4. Refine Sources

Generate additional searches based on user feedback and fetch new sources. This endpoint returns new sources (5 by default) that should be *added* to the ones in the previos step.

**Endpoint:** `POST /stepwise/refine-sources`

**Authentication:** Required

**Request Body:**
```json
{
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "searches": [
    "huella carbono ciclo vida coche eléctrico 2025",
    "emisiones gases efecto invernadero coches eléctricos vs gasolina"
  ],
  "language": "es",
  "location": "es",
  "refinement": "Quiero información oficial de la Unión Europea",
  "model": "google/gemini-2.5-flash",
  "web_search_provider": "serper",
  "ban_domains": null
}
```

**Parameters:**
- `input` (string, required): Original statement (10-1000 characters)
- `searches` (array of strings, required): List of current search queries, they are returned by this and the find sources step. 
- `language` (string, required): Language code
- `location` (string, required): Location code
- `refinement` (string, required): User feedback for additional searches
- `model` (string, required): Model name to use
- `web_search_provider` (string, optional): Web search provider (default: "serper")
- `ban_domains` (array of strings, optional): Domains to exclude

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "sources": [
      {
        "title": "Reducción de las emisiones de CO2 - Agencia Europea",
        "link": "https://www.eca.europa.eu/es/publications?ref=sr-2024-01",
        "snippet": "En 2021, representaron el 23% de las emisiones totales de gases...",
        "favicon": "https://www.eca.europa.eu/favicon.ico",
        "base_url": "https://www.eca.europa.eu"
      }
    ],
    "searches": [
      "Unión Europea comunicado prensa impacto ambiental coches eléctricos"
    ],
    "cost": 0.0014266
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/refine-sources \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "searches": [
      "huella carbono ciclo vida coche eléctrico 2025",
      "emisiones gases efecto invernadero coches eléctricos vs gasolina"
    ],
    "language": "es",
    "location": "es",
    "refinement": "Quiero información oficial de la Unión Europea",
    "model": "google/gemini-2.5-flash"
  }'
```

---

### 5. Generate Final Article

Generate the final fact-checking article with Q&A, citations, and metadata. Same format as veridika.ai altough some fields are set to None as we don't use them. 

**Endpoint:** `POST /stepwise/generate-article`

**Authentication:** Required

**Request Body:**
```json
{
  "questions": [
    "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
    "¿Qué tipo de emisiones producen los coches de gasolina?"
  ],
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "language": "es",
  "location": "es",
  "sources": [
    {
      "title": "Electric Cars - Wikipedia",
      "link": "https://www.wikipedia.org/wiki/Electric_car",
      "snippet": "An electric car is an automobile powered by electricity...",
      "favicon": "https://www.wikipedia.org/favicon.ico",
      "base_url": "https://www.wikipedia.org"
    }
  ],
  "model": "google/gemini-2.5-flash",
  "use_rag": true,
  "embedding_model": "gemini-embedding-001",
  "article_writer_model": "google/gemini-2.5-flash",
  "question_answer_model": "google/gemini-2.5-flash",
  "metadata_model": "google/gemini-2.5-flash",
}
```

**Parameters:**
- `questions` (array of strings, required): List of critical questions
- `input` (string, required): Original statement (10-1000 characters)
- `language` (string, required): Language code
- `location` (string, required): Location code
- `sources` (array of objects, required): List of source objects with title, link, snippet, favicon, base_url
- `model` (string, required): Default model name to use
- `use_rag` (boolean, optional): Whether to use RAG (default: true)
- `embedding_model` (string, optional): Embedding model for RAG (default: "gemini-embedding-001")
- `article_writer_model` (string, optional): Model for article generation (defaults to `model`)
- `question_answer_model` (string, optional): Model for Q&A (defaults to `model`)
- `metadata_model` (string, optional): Model for metadata extraction (defaults to `model`)

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "question": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "answer": "Full fact-checking article with citations...",
    "metadata": {
      "title": "Los coches eléctricos son más contaminantes que los coches de gasolina",
      "categories": ["Automoción", "Medio ambiente", "Ciencia"],
      "label": "Fake",
      "main_claim": "Los coches eléctricos no son más contaminantes que los coches de gasolina",
      "config": null
    },
    "sources": [
      {
        "id": "1",
        "title": "Electric Cars - Wikipedia",
        "link": "https://www.wikipedia.org/wiki/Electric_car",
        "favicon": "https://www.wikipedia.org/favicon.ico",
        "base_url": "https://www.wikipedia.org"
      }
    ],
    "related_questions": [
      {
        "question": "¿Cuál es la huella de carbono del ciclo de vida completo?",
        "answer": "La huella de carbono...",
        "sources_id": ["1"]
      }
    ],
    "cost": 0.009145,
    "runtime": null,
    "agents": null,
    "config": null,
    "image": null,
    "logs": null
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/generate-article \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "¿Cuál es la huella de carbono promedio del ciclo de vida completo de un coche eléctrico?",
      "¿Qué tipo de emisiones producen los coches de gasolina?"
    ],
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "language": "es",
    "location": "es",
    "sources": [
      {
        "title": "Electric Cars - Wikipedia",
        "link": "https://www.wikipedia.org/wiki/Electric_car",
        "snippet": "An electric car is an automobile powered by electricity...",
        "favicon": "https://www.wikipedia.org/favicon.ico",
        "base_url": "https://www.wikipedia.org"
      }
    ],
    "model": "google/gemini-2.5-flash",
    "use_rag": true,
    "embedding_model": "gemini-embedding-001"
  }'
```



---

### 6. Generate Image

Create a header image for the article by first generating a concise description with an LLM and then calling the Flux image generator.

**Endpoint:** `POST /stepwise/generate-image`

**Authentication:** Required

**Request Body:**
```json
{
  "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
  "model": "flux",
  "size": "1280x256",
  "style": "The style is bold and energetic, using vibrant colors and dynamic compositions to create visually engaging scenes that emphasize activity, culture, and lively atmospheres. Digital art."
}
```

**Parameters:**
- `input` (string, required): Statement/title to illustrate (5-1000 characters)
- `model` (string, required): Model for prompt generation (e.g., "google/gemini-2.5-flash")
- `size` (string, optional): Image size in `WIDTHxHEIGHT` (default: "1280x256")
- `style` (string, optional): Style guidance appended to the description

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Result Format (use `/result/{job_id}`):**
```json
{
  "status": "completed",
  "result": {
    "image_description": "Dynamic city street at dusk with electric vehicles...",
    "image_url": "https://...",
    "size": "1280x256",
    "cost": 0.00642
  },
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Example:**
```bash
curl -X POST http://localhost:7860/stepwise/generate-image \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "model": "google/gemini-2.5-flash",
    "size": "1280x256"
  }'
```


---

## Complete Stepwise Workflow Example

Here's a complete example of using the stepwise workflow:

```bash
# 1. Generate critical questions
STEP1=$(curl -X POST http://localhost:7860/stepwise/critical-questions \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Los coches eléctricos son más contaminantes que los coches de gasolina",
    "model": "google/gemini-2.5-flash",
    "language": "es",
    "location": "es"
  }')

JOB_ID=$(echo $STEP1 | jq -r '.job_id')

# Wait for completion and get questions
sleep 5
QUESTIONS=$(curl -X GET http://localhost:7860/result/$JOB_ID \
  -H "X-API-Key: your-secret-api-key-here" | jq '.result.questions')

# 2. Search for sources
STEP2=$(curl -X POST http://localhost:7860/stepwise/search-sources \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d "{
    \"questions\": $QUESTIONS,
    \"input\": \"Los coches eléctricos son más contaminantes que los coches de gasolina\",
    \"language\": \"es\",
    \"location\": \"es\",
    \"model\": \"google/gemini-2.5-flash\"
  }")

JOB_ID=$(echo $STEP2 | jq -r '.job_id')

# Wait for completion and get sources
sleep 10
SOURCES=$(curl -X GET http://localhost:7860/result/$JOB_ID \
  -H "X-API-Key: your-secret-api-key-here" | jq '.result.sources')

# 3. Generate final article
STEP3=$(curl -X POST http://localhost:7860/stepwise/generate-article \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d "{
    \"questions\": $QUESTIONS,
    \"input\": \"Los coches eléctricos son más contaminantes que los coches de gasolina\",
    \"language\": \"es\",
    \"location\": \"es\",
    \"sources\": $SOURCES,
    \"model\": \"google/gemini-2.5-flash\",
    \"use_rag\": true,
    \"embedding_model\": \"gemini-embedding-001\"
  }")

JOB_ID=$(echo $STEP3 | jq -r '.job_id')

# Wait for completion and get final result
sleep 30
curl -X GET http://localhost:7860/result/$JOB_ID \
  -H "X-API-Key: your-secret-api-key-here" | jq '.result'
```

---

## Error Responses

### 401 Unauthorized
```json
{
  "detail": "Missing API key. Please provide X-API-Key header."
}
```

### 403 Forbidden
```json
{
  "detail": "Invalid API key"
}
```

### 404 Not Found
```json
{
  "detail": "Job not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "input"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```



