import json
from collections.abc import Sequence

from redis.asyncio import Redis


class RedisChatStore:
    def __init__(self, redis_client: Redis, ttl_seconds: int = 86400) -> None:
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"history:{session_id}"

    @staticmethod
    def _cache_key(session_id: str, message: str) -> str:
        return f"cache:{session_id}:{message.strip().lower()}"

    async def get_cached_response(self, session_id: str, message: str) -> str | None:
        data = await self.redis_client.get(self._cache_key(session_id, message))
        if not data:
            return None
        return str(data)

    async def cache_response(self, session_id: str, message: str, answer: str) -> None:
        await self.redis_client.setex(self._cache_key(session_id, message), self.ttl_seconds, answer)

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        raw = await self.redis_client.lrange(self._history_key(session_id), 0, -1)
        history: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            history.append(json.loads(item))
        return history

    async def append_history(self, session_id: str, messages: Sequence[dict[str, str]]) -> None:
        if not messages:
            return

        key = self._history_key(session_id)
        encoded = [json.dumps(msg) for msg in messages]
        await self.redis_client.rpush(key, *encoded)
        await self.redis_client.expire(key, self.ttl_seconds)
