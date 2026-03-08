import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import ChatRequest, ChatResponse


def test_chat_request_valid() -> None:
    req = ChatRequest(session_id="test", message="hello")
    assert req.session_id == "test"
    assert req.message == "hello"


def test_chat_request_empty_session_id() -> None:
    with pytest.raises(ValueError):
        ChatRequest(session_id="", message="hello")


def test_chat_request_empty_message() -> None:
    with pytest.raises(ValueError):
        ChatRequest(session_id="test", message="")


def test_chat_response_defaults() -> None:
    resp = ChatResponse(session_id="test", answer="hi")
    assert resp.session_id == "test"
    assert resp.answer == "hi"
    assert resp.cached is False


def test_chat_response_with_cached() -> None:
    resp = ChatResponse(session_id="test", answer="hi", cached=True)
    assert resp.cached is True
