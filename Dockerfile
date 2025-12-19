FROM python:3.11.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    uv venv -p 3.11; \
    uv pip install --upgrade pip; \
    uv pip install --upgrade certifi setuptools wheel ninja; \
    uv pip install --no-compile --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

FROM python:3.11.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:/root/.local/bin:/home/appuser/.local/bin:${PATH}"

RUN useradd -m -u 1000 --no-log-init appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser . /app

EXPOSE 7860

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python docker-healthcheck.py

CMD ["uvicorn", "veridika_server:app", "--host", "0.0.0.0", "--port", "7860"]