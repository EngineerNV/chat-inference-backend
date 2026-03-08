import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.model_service import TinyLlamaService
from app.redis_store import RedisChatStore


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_invalid_request(client: TestClient) -> None:
    response = client.post("/chat", json={"session_id": "", "message": "hello"})
    assert response.status_code == 422  # validation error
