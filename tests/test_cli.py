import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import cli


def test_run_chat_saves_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "send_message",
        lambda api_url, session_id, message: {"answer": "hello back", "cached": False},
    )
    prompts = iter(["hello", "/exit"])
    output: list[str] = []

    cli.run_chat(
        api_url="http://localhost:8000/chat",
        session_id="session-a",
        output_dir=tmp_path,
        input_fn=lambda _: next(prompts),
        output_fn=output.append,
    )

    saved = json.loads((tmp_path / "session-a.json").read_text(encoding="utf-8"))
    assert saved[0]["role"] == "user"
    assert saved[0]["content"] == "hello"
    assert saved[1]["role"] == "assistant"
    assert "assistant> hello back" in output


def test_run_chat_loads_existing_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = [{"role": "user", "content": "prior"}]
    (tmp_path / "session-b.json").write_text(json.dumps(existing), encoding="utf-8")
    prompts = iter(["/save", "/exit"])
    output: list[str] = []

    monkeypatch.setattr(cli, "send_message", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unused")))

    cli.run_chat(
        api_url="http://localhost:8000/chat",
        session_id="session-b",
        output_dir=tmp_path,
        input_fn=lambda _: next(prompts),
        output_fn=output.append,
    )

    assert any("loaded 1 messages" in line for line in output)
    assert any("saved:" in line for line in output)


def test_main_prompts_for_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run_chat(api_url: str, session_id: str, output_dir: Path, input_fn=input, output_fn=print) -> None:
        called["api_url"] = api_url
        called["session_id"] = session_id
        called["output_dir"] = output_dir

    monkeypatch.setattr(cli, "run_chat", fake_run_chat)
    monkeypatch.setattr("builtins.input", lambda _: "session-c")

    exit_code = cli.main(["--api-url", "http://example.test/chat", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert called["session_id"] == "session-c"
    assert called["api_url"] == "http://example.test/chat"
