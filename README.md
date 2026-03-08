# chat-inference-backend

TinyLlama-powered chat inference backend with FastAPI + Redis.

## What changed

This branch replaces the previous Dialo-style inference direction with a TinyLlama-first service architecture:

- FastAPI API server (`/health`, `/chat`)
- Redis-backed session history + response cache
- TinyLlama model loading and generation via Hugging Face Transformers
- `accelerate` runtime support for memory-efficient model loading (`low_cpu_mem_usage=True`)
- Dockerized local stack (`api` + `redis`)

## Quick start

### Local Python runtime

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Docker Compose

```bash
docker compose up --build
```

## Common startup issues

If startup fails with:

```text
ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>=0.26.0'`
```

Install dependencies again (or install accelerate directly):

```bash
pip install -r requirements.txt
# or
pip install 'accelerate>=0.26.0'
```

## API usage

### Health check

```bash
curl http://localhost:8000/health
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Hello TinyLlama"}'
```

## Environment variables

- `MODEL_NAME` (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `MAX_NEW_TOKENS` (default: `128`)
- `TEMPERATURE` (default: `0.7`)
- `REDIS_URL` (default: `redis://redis:6379/0`)
- `REDIS_HISTORY_TTL_SECONDS` (default: `86400`)
