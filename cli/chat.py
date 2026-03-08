"""Interactive CLI chat interface for the TinyLlama inference backend.

Usage
-----
Start a new session::

    python -m cli.chat

Resume an existing session::

    python -m cli.chat --session <session-id>

Use a different backend URL::

    python -m cli.chat --url http://localhost:8000

List saved sessions::

    python -m cli.chat --list

Export a session to JSON (stdout)::

    python -m cli.chat --export <session-id>

In-session commands
-------------------
Type ``/help`` for a list of available slash-commands.
Type ``/quit`` or ``/exit`` to leave the chat (Ctrl-C also works).
"""

from __future__ import annotations

import argparse
import sys

from cli.client import BackendError, ChatClient
from cli.session import SessionManager


# ANSI colour helpers (disabled automatically when stdout is not a TTY)
def _colour(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _colour("32", t)


def _yellow(t: str) -> str:
    return _colour("33", t)


def _cyan(t: str) -> str:
    return _colour("36", t)


def _dim(t: str) -> str:
    return _colour("2", t)


def _bold(t: str) -> str:
    return _colour("1", t)


# ---------------------------------------------------------------------------
# Slash-command handlers
# ---------------------------------------------------------------------------

_SLASH_HELP = """\
Available commands:
  /help          Show this help message
  /session       Print the current session ID
  /history       Print the full conversation history
  /save          Force-save the session to disk
  /export        Print the session JSON to stdout
  /new           Start a new session (current session is saved)
  /quit  /exit   Exit the chat
"""


def _cmd_history(session_mgr: SessionManager, session_id: str) -> None:
    history = session_mgr.get_history(session_id)
    if not history:
        print(_dim("  (no messages yet)"))
        return
    for msg in history:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        label = _green("You") if role == "user" else _cyan("Bot")
        print(f"  {label}: {content}")


def _cmd_export(session_mgr: SessionManager, session_id: str) -> None:
    try:
        print(session_mgr.export_json(session_id))
    except FileNotFoundError as exc:
        print(_yellow(str(exc)))


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def run_chat(
    session_id: str | None,
    backend_url: str,
    sessions_dir: str | None = None,
) -> None:
    """Start (or resume) an interactive chat session."""
    client = ChatClient(base_url=backend_url)
    session_mgr = SessionManager(sessions_dir=sessions_dir)

    # Health-check the backend before starting
    if not client.health():
        print(
            _yellow(
                f"Warning: backend at {backend_url} is not reachable.\n"
                "Messages will fail until the server is running."
            )
        )

    # Resolve session
    if session_id:
        try:
            session_mgr.load(session_id)
            print(_dim(f"Resuming session {session_id}"))
        except FileNotFoundError:
            # Session exists in Redis/backend but not locally – create local record
            session_mgr._write(session_id, [])  # noqa: SLF001
            print(_dim(f"Starting local record for session {session_id}"))
    else:
        session_id = session_mgr.new_session()
        print(_dim(f"New session started: {session_id}"))

    print(_bold("TinyLlama Chat") + _dim("  (type /help for commands, /quit to exit)\n"))

    while True:
        try:
            raw = input(_green("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # Slash commands
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(_SLASH_HELP)
            elif cmd == "/session":
                print(_dim(f"  Session ID: {session_id}"))
            elif cmd == "/history":
                _cmd_history(session_mgr, session_id)
            elif cmd == "/save":
                print(_dim("  Session already persisted to disk automatically."))
            elif cmd == "/export":
                _cmd_export(session_mgr, session_id)
            elif cmd == "/new":
                session_id = session_mgr.new_session()
                print(_dim(f"  New session: {session_id}"))
            else:
                print(_yellow(f"  Unknown command: {cmd}  (type /help for help)"))
            continue

        # Normal message – send to backend
        try:
            resp = client.chat(session_id=session_id, message=raw)
        except BackendError as exc:
            print(_yellow(f"  Error from backend: {exc}"))
            continue
        except OSError as exc:
            print(_yellow(f"  Network error: {exc}"))
            continue

        answer = str(resp.get("answer", ""))
        cached = bool(resp.get("cached", False))

        # Persist both turns locally
        session_mgr.save_message(session_id, "user", raw)
        session_mgr.save_message(session_id, "assistant", answer)

        suffix = _dim("  [cached]") if cached else ""
        print(f"{_cyan('Bot')}: {answer}{suffix}\n")

    print(_dim("Session saved.  Goodbye!"))


# ---------------------------------------------------------------------------
# List / export sub-commands (non-interactive)
# ---------------------------------------------------------------------------

def list_sessions(sessions_dir: str | None = None) -> None:
    """Print a table of all saved sessions."""
    session_mgr = SessionManager(sessions_dir=sessions_dir)
    sessions = session_mgr.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    print(_bold(f"{'Session ID':<38}  {'Messages':>8}  {'Updated'}"))
    print("-" * 72)
    for s in sessions:
        print(
            f"{s['session_id']:<38}  {s['message_count']:>8}  {s['updated_at'][:19]}"
        )


def export_session(session_id: str, sessions_dir: str | None = None) -> None:
    """Print a single session as JSON."""
    session_mgr = SessionManager(sessions_dir=sessions_dir)
    try:
        print(session_mgr.export_json(session_id))
    except FileNotFoundError as exc:
        print(_yellow(str(exc)), file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli.chat",
        description="Interactive CLI chat interface for the TinyLlama inference backend.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        metavar="URL",
        help="Backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="Resume an existing session by ID",
    )
    parser.add_argument(
        "--sessions-dir",
        metavar="DIR",
        default=None,
        help="Directory where session JSON files are stored",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all saved sessions and exit",
    )
    parser.add_argument(
        "--export",
        metavar="SESSION_ID",
        default=None,
        help="Export a session as JSON to stdout and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_sessions(sessions_dir=args.sessions_dir)
        return

    if args.export:
        export_session(args.export, sessions_dir=args.sessions_dir)
        return

    run_chat(
        session_id=args.session,
        backend_url=args.url,
        sessions_dir=args.sessions_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
