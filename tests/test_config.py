import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "tiny-llama-chat-inference"
    assert settings.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert settings.max_new_tokens == 128
    assert settings.temperature == 0.7
    assert settings.redis_url == "redis://redis:6379/0"
    assert settings.redis_history_ttl_seconds == 86400


def test_settings_validation_max_new_tokens() -> None:
    with pytest.raises(Exception):  # Pydantic validation error
        Settings(max_new_tokens=0)
    with pytest.raises(Exception):
        Settings(max_new_tokens=2000)


def test_settings_validation_temperature() -> None:
    with pytest.raises(Exception):  # Pydantic validation error
        Settings(temperature=-0.1)
    with pytest.raises(Exception):
        Settings(temperature=2.1)


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MAX_NEW_TOKENS", "256")
    monkeypatch.setenv("TEMPERATURE", "0.5")

    settings = Settings()
    assert settings.model_name == "test-model"
    assert settings.max_new_tokens == 256
    assert settings.temperature == 0.5


def test_get_settings_caches() -> None:
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2
