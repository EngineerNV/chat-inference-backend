"""Integration-style tests for cli.chat (run_chat, list_sessions, export_session, main)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.chat import export_session, list_sessions, main, run_chat
from cli.client import BackendError


# ---------------------------------------------------------------------------
# Shared fixture: isolated sessions directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def sessions_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


# ---------------------------------------------------------------------------
# run_chat
# ---------------------------------------------------------------------------

class TestRunChat:
    def _chat_resp(self, answer: str, cached: bool = False) -> dict:
        return {"session_id": "s1", "answer": answer, "cached": cached}

    def test_new_session_and_single_message(
        self, sessions_dir: Path, capsys
    ) -> None:
        inputs = iter(["hello", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("cli.chat.ChatClient.chat", return_value=self._chat_resp("hi there")),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "hi there" in out

    def test_resume_existing_session(
        self, sessions_dir: Path, capsys
    ) -> None:
        from cli.session import SessionManager

        mgr = SessionManager(sessions_dir=sessions_dir)
        sid = mgr.new_session()
        inputs = iter(["/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=sid, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Resuming" in out

    def test_unknown_session_id_creates_local_record(
        self, sessions_dir: Path, capsys
    ) -> None:
        inputs = iter(["/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(
                session_id="ghost-123",
                backend_url="http://test",
                sessions_dir=str(sessions_dir),
            )

        assert (sessions_dir / "ghost-123.json").exists()

    def test_backend_unreachable_warning(
        self, sessions_dir: Path, capsys
    ) -> None:
        inputs = iter(["/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=False),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://dead", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "not reachable" in out

    def test_backend_error_is_shown(
        self, sessions_dir: Path, capsys
    ) -> None:
        inputs = iter(["what?", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("cli.chat.ChatClient.chat", side_effect=BackendError(422, "bad input")),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Error from backend" in out

    def test_network_error_is_shown(
        self, sessions_dir: Path, capsys
    ) -> None:
        inputs = iter(["what?", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("cli.chat.ChatClient.chat", side_effect=OSError("timeout")),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Network error" in out

    def test_slash_help(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/help", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "/quit" in out

    def test_slash_session(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/session", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Session ID" in out

    def test_slash_history_empty(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/history", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "no messages" in out

    def test_slash_history_with_messages(self, sessions_dir: Path, capsys) -> None:
        from cli.session import SessionManager

        mgr = SessionManager(sessions_dir=sessions_dir)
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "ping")
        mgr.save_message(sid, "assistant", "pong")

        inputs = iter(["/history", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=sid, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "ping" in out
        assert "pong" in out

    def test_slash_export(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/export", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        # /export prints valid JSON
        # Find the JSON block between other output lines
        lines = out.splitlines()
        json_lines = [l for l in lines if l.strip().startswith("{") or l.strip().startswith('"')]
        assert json_lines  # some JSON was printed

    def test_slash_new_creates_new_session(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/new", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "New session" in out

    def test_slash_save(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/save", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Session already persisted" in out

    def test_unknown_slash_command(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["/badcommand", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "Unknown command" in out

    def test_empty_input_is_skipped(self, sessions_dir: Path) -> None:
        """Empty input should not call the backend."""
        inputs = iter(["", "   ", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("cli.chat.ChatClient.chat") as mock_chat,
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        mock_chat.assert_not_called()

    def test_eof_exits_gracefully(self, sessions_dir: Path) -> None:
        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch("builtins.input", side_effect=EOFError),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

    def test_messages_persisted_locally(self, sessions_dir: Path) -> None:
        from cli.session import SessionManager

        inputs = iter(["hello world", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch(
                "cli.chat.ChatClient.chat",
                return_value={"session_id": "s", "answer": "hey", "cached": False},
            ),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        mgr = SessionManager(sessions_dir=sessions_dir)
        sessions = mgr.list_sessions()
        assert sessions  # at least one session saved
        sid = sessions[0]["session_id"]
        history = mgr.get_history(sid)
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    def test_cached_response_flag_shown(self, sessions_dir: Path, capsys) -> None:
        inputs = iter(["hi", "/quit"])

        with (
            patch("cli.chat.ChatClient.health", return_value=True),
            patch(
                "cli.chat.ChatClient.chat",
                return_value={"session_id": "s", "answer": "cached answer", "cached": True},
            ),
            patch("builtins.input", side_effect=inputs),
        ):
            run_chat(session_id=None, backend_url="http://test", sessions_dir=str(sessions_dir))

        out = capsys.readouterr().out
        assert "cached" in out.lower()


# ---------------------------------------------------------------------------
# list_sessions function
# ---------------------------------------------------------------------------

class TestListSessionsFunction:
    def test_no_sessions(self, sessions_dir: Path, capsys) -> None:
        list_sessions(sessions_dir=str(sessions_dir))
        out = capsys.readouterr().out
        assert "No saved sessions" in out

    def test_with_sessions(self, sessions_dir: Path, capsys) -> None:
        from cli.session import SessionManager

        mgr = SessionManager(sessions_dir=sessions_dir)
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "hello")

        list_sessions(sessions_dir=str(sessions_dir))
        out = capsys.readouterr().out
        assert sid in out


# ---------------------------------------------------------------------------
# export_session function
# ---------------------------------------------------------------------------

class TestExportSessionFunction:
    def test_valid_session(self, sessions_dir: Path, capsys) -> None:
        from cli.session import SessionManager

        mgr = SessionManager(sessions_dir=sessions_dir)
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "test")

        export_session(sid, sessions_dir=str(sessions_dir))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["session_id"] == sid

    def test_missing_session_exits(self, sessions_dir: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            export_session("no-such-id", sessions_dir=str(sessions_dir))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() argument parsing
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_list_flag(self, sessions_dir: Path, capsys) -> None:
        main(["--list", "--sessions-dir", str(sessions_dir)])
        out = capsys.readouterr().out
        assert "No saved sessions" in out

    def test_main_export_flag(self, sessions_dir: Path) -> None:
        from cli.session import SessionManager

        mgr = SessionManager(sessions_dir=sessions_dir)
        sid = mgr.new_session()

        main(["--export", sid, "--sessions-dir", str(sessions_dir)])

    def test_main_export_missing_session_exits(self, sessions_dir: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--export", "ghost", "--sessions-dir", str(sessions_dir)])
        assert exc_info.value.code == 1

    def test_main_starts_chat(self, sessions_dir: Path) -> None:
        with (
            patch("cli.chat.run_chat") as mock_run,
        ):
            main(["--sessions-dir", str(sessions_dir)])
        mock_run.assert_called_once()
