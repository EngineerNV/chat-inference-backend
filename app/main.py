from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.model_service import TinyLlamaService
from app.redis_store import RedisChatStore
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.store = RedisChatStore(app.state.redis, settings.redis_history_ttl_seconds)
    app.state.model = TinyLlamaService(
        model_name=settings.model_name,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
    )
    try:
        yield
    finally:
        await app.state.redis.close()


app = FastAPI(title="TinyLlama Inference Backend", lifespan=lifespan)


def get_runtime_settings() -> Settings:
    return get_settings()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, settings: Settings = Depends(get_runtime_settings)) -> ChatResponse:
    store: RedisChatStore = app.state.store
    model: TinyLlamaService = app.state.model

    cached_response = await store.get_cached_response(payload.session_id, payload.message)
    if cached_response:
        return ChatResponse(session_id=payload.session_id, answer=cached_response, cached=True)

    history = await store.get_history(payload.session_id)
    answer = model.generate_response(payload.message, history=history)

    await store.append_history(
        payload.session_id,
        [
            {"role": "user", "content": payload.message},
            {"role": "assistant", "content": answer},
        ],
    )
    await store.cache_response(payload.session_id, payload.message, answer)

    return ChatResponse(session_id=payload.session_id, answer=answer, cached=False)
