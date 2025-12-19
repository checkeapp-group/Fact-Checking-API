# Veridika Fact-Checking Server

The easiest and best way of running the fact-checking server is via Docker. You will need to first build the docker image and then run it together with a redist server. This document will guide you trough all the step required to setup the server. 

The server specification together with examples of how to send queries are available in [server-specification.md](server-specification.md).

## Prerequisites

- Docker 24+
- A Linux host for the server (or WSL2/macOS) with at least 2 CPU and 2GB RAM
- Redis (we will run it as a separate container)
- A `.env` file with required API keys and server configuration (see below)

## Prepare the Environment

1) Create a `.env` file. You will need the following variables.

```bash
# Server config
API_KEY=your-secret-api-key  # Api key for the server
REDIS_URL=redis://my-redis:6379
SERVER_ID=veridika_server_1
LOG_LEVEL=INFO
MAX_CONCURRENT_JOBS=4
HOST=0.0.0.0
PORT=7860

WORKFLOW_CONFIG_PATH=configs/pipeline_configs/gemini_rag.yaml # Workflow for the end-to-end endpoint

# Provider keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
SERPER_API_KEY=...
REPLICATE_API_TOKEN=r8_...
OPENROUTER_API_KEY=sk-or-...

# Self-hosted keys if you want to serve a custom model. Names are defined in configs/local-model-configs
RUNPOD_API_KEY=rpa...
RUNPOD_Latxa70B_URL=https://api.runpod.ai/v2/.../openai/v1
```

2) Optional: Run a quick pre-build verification:

```bash
python docker-pre-build-check.py
```

## Build the Docker Image

```bash
docker build --platform linux/amd64 -t factchecking-server:latest .
```

Copy the image to the host machine (If it is different from the one in which you build the image).

```bash
docker save factchecking-server:latest > factchecking-server.tar
rsync -avzh factchecking-server.tar user@your-vps:/home/user/ --progress
# On the host machine
docker load < /home/user/factchecking-server.tar
```

## Run the docker

### 1) Create a dedicated network and start Redis

```bash
docker network create factchecking-network || true

docker run -d --name my-redis \
  --network factchecking-network \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7-alpine
```

### 2) Start the Veridika server using your .env

```bash
docker run -d --name veridika-server \
  --network factchecking-network \
  --env-file ./.env \
  -p 7860:7860 \
  --restart unless-stopped \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 2g --cpus 2 \
  --tmpfs /tmp:rw,exec,size=1g \
  --tmpfs /var/tmp:rw,size=8192m \
  --shm-size 8g \
  -e TMPDIR=/tmp \
  factchecking-server:latest
```

## Verify Deployment

```bash
# Check container status
docker ps

# Tail server logs
docker logs -f veridika-server

# Health endpoint (no auth)
curl http://localhost:7860/health

# Status endpoint (auth required)
curl -H "X-API-Key: $API_KEY" http://localhost:7860/status
```

## Submit a Test Job

```bash
curl -X POST "http://localhost:7860/fact-check" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "article_topic": "Los coches eléctricos son más contaminantes que los coches de gasolina",
        "language": "es",
        "location": "es",
        "config": "unused"
      }'

# Replace JOB_ID with the value returned by the previous call
curl -H "X-API-Key: $API_KEY" http://localhost:7860/result/JOB_ID
```

You can also test the server using the [test_server.py](../test_server.py) script. You can replace `localhost` with the IP adress of your host machine. 

```bash 
python3 test_server.py  --url http://localhost:7860 --api-key $API_KEY
```

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | ✅ | Server authentication key |
| `WORKFLOW_CONFIG_PATH` | ✅ | Path to workflow config (mounted in image) |
| `REDIS_URL` | ✅ | Redis URL (e.g., `redis://my-redis:6379`) |
| `MAX_CONCURRENT_JOBS` | ❌ | Max worker concurrency (default: 4) |
| `HOST` | ❌ | Bind host (default: `0.0.0.0`) |
| `PORT` | ❌ | Bind port (default: `7860`) |
| `LOG_LEVEL` | ❌ | Logging level (e.g., `INFO`) |
| Provider keys | ✅ | Always required: `SERPER_API_KEY`. Requied unless you use a self-hosted model: `OPENROUTER_API_KEY`. Required for image generation (unless you use self-hosted image generation model) `REPLICATE_API_TOKEN`. At least one required for the embeddings (unless you use self-hosted embeddings): `OPENAI_API_KEY`, `GEMINI_API_KEY`. |
| Self-hosted keys | ❌ | Keys and urls defined in `configs/local-model-config` |


## Troubleshooting

- If the server can’t connect to Redis, verify `REDIS_URL` and that the `my-redis` container is running on the same network.
- Check logs with `docker logs -f veridika-server`.
- Verify environment is loaded: `docker exec veridika-server env | sort`.
