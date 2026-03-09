import argparse
import json
import os
from pathlib import Path
from urllib import error, request


def send_message(api_url: str, session_id: str, message: str, timeout: int = 30) -> dict[str, object]:
    payload = json.dumps({"session_id": session_id, "message": message}).encode("utf-8")
    req = request.Request(api_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"API error {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc


def _session_file(output_dir: Path, session_id: str) -> Path:
    return output_dir / f"{session_id}.json"


def _load_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _save_history(path: Path, history: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_chat(
    api_url: str,
    session_id: str,
    output_dir: Path,
    input_fn=input,
    output_fn=print,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = _session_file(output_dir, session_id)
    history = _load_history(transcript_path)

    output_fn(f"session: {session_id}")
    if history:
        output_fn(f"loaded {len(history)} messages from {transcript_path}")

    while True:
        prompt = input_fn("you> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            _save_history(transcript_path, history)
            output_fn("bye")
            break
        if prompt == "/save":
            _save_history(transcript_path, history)
            output_fn(f"saved: {transcript_path}")
            continue
        if prompt == "/history":
            output_fn(json.dumps(history, indent=2))
            continue

        reply = send_message(api_url=api_url, session_id=session_id, message=prompt)
        answer = str(reply.get("answer", ""))
        cached = bool(reply.get("cached", False))
        output_fn(f"assistant{' (cached)' if cached else ''}> {answer}")

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": answer, "cached": cached})
        _save_history(transcript_path, history)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TinyLlama terminal chat client")
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat"))
    parser.add_argument("--session-id")
    parser.add_argument("--output-dir", default="sessions")
    args = parser.parse_args(argv)

    session_id = args.session_id or input("session id> ").strip()
    if not session_id:
        raise SystemExit("session id is required")

    run_chat(api_url=args.api_url, session_id=session_id, output_dir=Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
