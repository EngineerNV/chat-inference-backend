"""HTTP client for the TinyLlama chat inference backend."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import json


class BackendError(Exception):
    """Raised when the backend returns an error response."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Backend error {status}: {detail}")
        self.status = status
        self.detail = detail


class ChatClient:
    """Thin HTTP client that wraps the /chat and /health endpoints.

    Uses only stdlib ``urllib`` so that the CLI has no extra runtime
    dependencies beyond what is already listed in ``requirements.txt``.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Return True if the backend is reachable and healthy."""
        try:
            data = self._get("/health")
            return data.get("status") == "ok"
        except (BackendError, OSError):
            return False

    def chat(self, session_id: str, message: str) -> dict[str, object]:
        """Send *message* to the backend and return the full response dict.

        Returns a dict with keys ``session_id``, ``answer``, ``cached``.

        Raises:
            BackendError: On non-2xx responses.
            OSError: On network errors.
        """
        payload = {"session_id": session_id, "message": message}
        return self._post("/chat", payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        url = self.base_url + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(req)

    def _get(self, path: str) -> dict[str, object]:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        return self._send(req)

    @staticmethod
    def _send(req: urllib.request.Request) -> dict[str, object]:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read().decode()
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            try:
                detail = json.loads(raw).get("detail", raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw
            raise BackendError(exc.code, detail) from exc
