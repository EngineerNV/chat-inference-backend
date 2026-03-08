"""Session management: persist and resume chat sessions as JSON files."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_SESSIONS_DIR = Path.home() / ".chat-inference" / "sessions"


class SessionManager:
    """Load, save and list chat sessions stored as JSON files on disk.

    Each session is stored as a single JSON file named ``<session_id>.json``
    inside *sessions_dir*.  The file schema is::

        {
            "session_id": "<uuid>",
            "created_at": "<iso8601>",
            "updated_at": "<iso8601>",
            "messages": [
                {"role": "user"|"assistant", "content": "…"},
                …
            ]
        }
    """

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        self.sessions_dir = Path(sessions_dir) if sessions_dir else _DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session_record(self, session_id: str) -> None:
        """Create a local session file for a known remote *session_id*.

        Use this when you want to track a session that already exists in the
        backend (e.g. Redis) but has no local JSON record yet.
        """
        self._write(session_id, [])

    def new_session(self) -> str:
        """Create a fresh session ID and persist an empty session file."""
        session_id = str(uuid.uuid4())
        self._write(session_id, messages=[])
        return session_id

    def load(self, session_id: str) -> dict[str, Any]:
        """Load an existing session.  Returns the full session dict.

        Raises:
            FileNotFoundError: If no session file exists for *session_id*.
        """
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found in {self.sessions_dir}")
        with path.open() as fh:
            return json.load(fh)

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Append a single message to the session and persist to disk."""
        try:
            data = self.load(session_id)
        except FileNotFoundError:
            data = self._blank(session_id)

        data["messages"].append({"role": role, "content": content})
        data["updated_at"] = _now_iso()
        self._write_raw(session_id, data)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the message list for *session_id*, or [] if not found."""
        try:
            return self.load(session_id)["messages"]
        except FileNotFoundError:
            return []

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return summary info for all saved sessions, newest first."""
        sessions: list[dict[str, Any]] = []
        for p in sorted(self.sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with p.open() as fh:
                    data = json.load(fh)
                sessions.append(
                    {
                        "session_id": data.get("session_id", p.stem),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete(self, session_id: str) -> None:
        """Delete a session file if it exists."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()

    def export_json(self, session_id: str) -> str:
        """Return the full session as a pretty-printed JSON string."""
        data = self.load(session_id)
        return json.dumps(data, indent=2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _blank(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "messages": [],
        }

    def _write(self, session_id: str, messages: list[dict[str, str]]) -> None:
        data = self._blank(session_id)
        data["messages"] = messages
        self._write_raw(session_id, data)

    def _write_raw(self, session_id: str, data: dict[str, Any]) -> None:
        with self._path(session_id).open("w") as fh:
            json.dump(data, fh, indent=2)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
