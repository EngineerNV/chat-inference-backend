"""Pytest configuration: stub out heavy ML dependencies so unit tests
can run in environments without GPU/model weights installed."""
import sys
from unittest.mock import MagicMock


def pytest_configure() -> None:
    """Register lightweight stubs for torch and transformers before any import."""
    if "torch" not in sys.modules:
        _torch = MagicMock(name="torch")
        _torch.cuda.is_available.return_value = False
        _torch.float32 = "float32"
        _torch.float16 = "float16"
        sys.modules["torch"] = _torch

    for _mod in ("transformers", "accelerate"):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock(name=_mod)
