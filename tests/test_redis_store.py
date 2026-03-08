import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.redis_store import RedisChatStore


@pytest.fixture
def mock_redis() -> AsyncMock:
    return AsyncMock()


@pytest.mark.asyncio
async def test_get_history_empty(mock_redis: AsyncMock) -> None:
    mock_redis.lrange.return_value = []
    store = RedisChatStore(mock_redis, ttl_seconds=3600)
    history = await store.get_history("session-1")
    assert history == []
    mock_redis.lrange.assert_called_once_with("history:session-1", 0, -1)


@pytest.mark.asyncio
async def test_get_history_with_messages(mock_redis: AsyncMock) -> None:
    raw = [
        json.dumps({"role": "user", "content": "hello"}),
        json.dumps({"role": "assistant", "content": "hi"}),
    ]
    mock_redis.lrange.return_value = raw
    store = RedisChatStore(mock_redis, ttl_seconds=3600)
    history = await store.get_history("session-1")

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "hi"


@pytest.mark.asyncio
async def test_append_history(mock_redis: AsyncMock) -> None:
    store = RedisChatStore(mock_redis, ttl_seconds=3600)
    messages = [{"role": "user", "content": "test"}]

    await store.append_history("session-1", messages)

    mock_redis.rpush.assert_called_once()
    mock_redis.expire.assert_called_once_with("history:session-1", 3600)


@pytest.mark.asyncio
async def test_cache_response(mock_redis: AsyncMock) -> None:
    store = RedisChatStore(mock_redis, ttl_seconds=3600)

    await store.cache_response("session-1", "hello", "response")

    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args
    cache_key = args[0][0]
    ttl = args[0][1]
    value = args[0][2]

    assert cache_key.startswith("cache:session-1:")
    assert ttl == 3600
    assert value == "response"


@pytest.mark.asyncio
async def test_get_cached_response_hit(mock_redis: AsyncMock) -> None:
    mock_redis.get.return_value = "cached_response"
    store = RedisChatStore(mock_redis, ttl_seconds=3600)

    result = await store.get_cached_response("session-1", "hello")

    assert result == "cached_response"


@pytest.mark.asyncio
async def test_get_cached_response_miss(mock_redis: AsyncMock) -> None:
    mock_redis.get.return_value = None
    store = RedisChatStore(mock_redis, ttl_seconds=3600)

    result = await store.get_cached_response("session-1", "hello")

    assert result is None
