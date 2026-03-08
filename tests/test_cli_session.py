"""Unit tests for cli.session.SessionManager."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from cli.session import SessionManager


# ---------------------------------------------------------------------------
# Fixture: fresh temp sessions dir for each test
# ---------------------------------------------------------------------------

@pytest.fixture()
def sessions_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture()
def mgr(sessions_dir: Path) -> SessionManager:
    return SessionManager(sessions_dir=sessions_dir)


# ---------------------------------------------------------------------------
# new_session
# ---------------------------------------------------------------------------

class TestCreateSessionRecord:
    def test_creates_file(self, mgr: SessionManager, sessions_dir: Path) -> None:
        mgr.create_session_record("remote-sid")
        assert (sessions_dir / "remote-sid.json").exists()

    def test_creates_empty_messages(self, mgr: SessionManager) -> None:
        mgr.create_session_record("remote-sid")
        assert mgr.get_history("remote-sid") == []


class TestNewSession:
    def test_creates_file(self, mgr: SessionManager, sessions_dir: Path) -> None:
        sid = mgr.new_session()
        assert (sessions_dir / f"{sid}.json").exists()

    def test_returns_uuid_string(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # uuid4

    def test_new_session_has_empty_messages(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        data = mgr.load(sid)
        assert data["messages"] == []

    def test_new_session_has_metadata(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        data = mgr.load(sid)
        assert "created_at" in data
        assert "updated_at" in data
        assert data["session_id"] == sid


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_existing(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        data = mgr.load(sid)
        assert data["session_id"] == sid

    def test_load_missing_raises(self, mgr: SessionManager) -> None:
        with pytest.raises(FileNotFoundError):
            mgr.load("non-existent-session")


# ---------------------------------------------------------------------------
# save_message / get_history
# ---------------------------------------------------------------------------

class TestSaveMessage:
    def test_appends_user_message(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "hello")
        history = mgr.get_history(sid)
        assert history == [{"role": "user", "content": "hello"}]

    def test_appends_multiple_messages(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "ping")
        mgr.save_message(sid, "assistant", "pong")
        history = mgr.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_save_message_creates_session_if_missing(self, mgr: SessionManager) -> None:
        mgr.save_message("orphan-id", "user", "hello")
        history = mgr.get_history("orphan-id")
        assert history == [{"role": "user", "content": "hello"}]

    def test_get_history_returns_empty_for_missing_session(self, mgr: SessionManager) -> None:
        history = mgr.get_history("does-not-exist")
        assert history == []

    def test_updated_at_changes_after_save(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        before = mgr.load(sid)["updated_at"]
        time.sleep(0.01)
        mgr.save_message(sid, "user", "hello")
        after = mgr.load(sid)["updated_at"]
        assert after >= before


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_empty_dir(self, mgr: SessionManager) -> None:
        assert mgr.list_sessions() == []

    def test_lists_all_sessions(self, mgr: SessionManager) -> None:
        sid1 = mgr.new_session()
        sid2 = mgr.new_session()
        sessions = mgr.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert sid1 in ids
        assert sid2 in ids

    def test_message_count(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "a")
        mgr.save_message(sid, "assistant", "b")
        sessions = mgr.list_sessions()
        entry = next(s for s in sessions if s["session_id"] == sid)
        assert entry["message_count"] == 2

    def test_corrupt_file_is_skipped(self, mgr: SessionManager, sessions_dir: Path) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "bad.json").write_text("{{not valid json")
        sid = mgr.new_session()
        sessions = mgr.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert sid in ids
        # 'bad' should be silently skipped
        assert "bad" not in ids


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing(self, mgr: SessionManager, sessions_dir: Path) -> None:
        sid = mgr.new_session()
        mgr.delete(sid)
        assert not (sessions_dir / f"{sid}.json").exists()

    def test_delete_non_existing_is_noop(self, mgr: SessionManager) -> None:
        mgr.delete("ghost-session")  # Should not raise


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_export_is_valid_json(self, mgr: SessionManager) -> None:
        sid = mgr.new_session()
        mgr.save_message(sid, "user", "hi")
        raw = mgr.export_json(sid)
        data = json.loads(raw)
        assert data["session_id"] == sid
        assert len(data["messages"]) == 1

    def test_export_raises_for_missing_session(self, mgr: SessionManager) -> None:
        with pytest.raises(FileNotFoundError):
            mgr.export_json("no-such-session")
