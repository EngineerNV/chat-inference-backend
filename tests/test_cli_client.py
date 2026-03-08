"""Unit tests for cli.client.ChatClient."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from cli.client import BackendError, ChatClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resp(body: dict, status: int = 200) -> MagicMock:
    """Build a fake urllib response context manager."""
    raw = json.dumps(body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(body: dict | str, code: int) -> urllib.error.HTTPError:
    if isinstance(body, dict):
        raw = json.dumps(body).encode()
    else:
        raw = body.encode()
    fp = BytesIO(raw)
    return urllib.error.HTTPError(
        url="http://test",
        code=code,
        msg="err",
        hdrs={},  # type: ignore[arg-type]
        fp=fp,
    )


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_true_when_status_ok(self) -> None:
        client = ChatClient(base_url="http://test")
        with patch("urllib.request.urlopen", return_value=_make_resp({"status": "ok"})):
            assert client.health() is True

    def test_returns_false_when_status_not_ok(self) -> None:
        client = ChatClient(base_url="http://test")
        with patch("urllib.request.urlopen", return_value=_make_resp({"status": "degraded"})):
            assert client.health() is False

    def test_returns_false_on_network_error(self) -> None:
        client = ChatClient(base_url="http://test")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert client.health() is False

    def test_returns_false_on_backend_error(self) -> None:
        client = ChatClient(base_url="http://test")
        err = _make_http_error({"detail": "server error"}, 500)
        with patch("urllib.request.urlopen", side_effect=err):
            assert client.health() is False


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

class TestChat:
    def test_returns_answer_dict(self) -> None:
        client = ChatClient(base_url="http://test")
        payload = {"session_id": "s1", "answer": "hello", "cached": False}
        with patch("urllib.request.urlopen", return_value=_make_resp(payload)):
            result = client.chat("s1", "hi")
        assert result["answer"] == "hello"
        assert result["cached"] is False

    def test_raises_backend_error_on_4xx(self) -> None:
        client = ChatClient(base_url="http://test")
        err = _make_http_error({"detail": "bad request"}, 422)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BackendError) as exc_info:
                client.chat("s1", "hi")
        assert exc_info.value.status == 422
        assert "bad request" in exc_info.value.detail

    def test_raises_backend_error_on_500(self) -> None:
        client = ChatClient(base_url="http://test")
        err = _make_http_error({"detail": "internal error"}, 500)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BackendError) as exc_info:
                client.chat("s1", "hi")
        assert exc_info.value.status == 500

    def test_backend_error_with_non_json_body(self) -> None:
        client = ChatClient(base_url="http://test")
        err = _make_http_error("plain text error", 503)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(BackendError) as exc_info:
                client.chat("s1", "hi")
        assert exc_info.value.status == 503
        assert "plain text error" in exc_info.value.detail

    def test_raises_os_error_on_network_failure(self) -> None:
        client = ChatClient(base_url="http://test")
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(OSError):
                client.chat("s1", "hi")

    def test_base_url_trailing_slash_stripped(self) -> None:
        client = ChatClient(base_url="http://test/")
        assert client.base_url == "http://test"

    def test_cached_response_flag(self) -> None:
        client = ChatClient(base_url="http://test")
        payload = {"session_id": "s1", "answer": "cached answer", "cached": True}
        with patch("urllib.request.urlopen", return_value=_make_resp(payload)):
            result = client.chat("s1", "hi")
        assert result["cached"] is True
