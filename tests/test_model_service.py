import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_service import TinyLlamaService


@patch("app.model_service.AutoTokenizer.from_pretrained")
@patch("app.model_service.AutoModelForCausalLM.from_pretrained")
def test_init_loads_model(mock_model_cls, mock_tokenizer_cls) -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model_cls.return_value = mock_model

    service = TinyLlamaService(model_name="test-model")

    mock_tokenizer_cls.assert_called_once_with("test-model")
    mock_model_cls.assert_called_once()
    assert service.tokenizer == mock_tokenizer
    assert service.model == mock_model


@patch("app.model_service.AutoTokenizer.from_pretrained")
@patch("app.model_service.AutoModelForCausalLM.from_pretrained")
def test_init_stores_config(mock_model_cls, mock_tokenizer_cls) -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model_cls.return_value = mock_model

    service = TinyLlamaService(
        model_name="custom-model",
        max_new_tokens=256,
        temperature=0.9,
    )

    assert service.model_name == "custom-model"
    assert service.max_new_tokens == 256
    assert service.temperature == 0.9


@patch("app.model_service.AutoTokenizer.from_pretrained")
@patch("app.model_service.AutoModelForCausalLM.from_pretrained")
def test_generate_response_with_history(mock_model_cls, mock_tokenizer_cls) -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model_cls.return_value = mock_model

    mock_tokenizer.apply_chat_template.return_value = "prompt text"
    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    mock_tokenizer.eos_token_id = 2
    mock_model.generate.return_value = MagicMock(shape=(1, 20))
    mock_model.generate.return_value.__getitem__.side_effect = lambda x: MagicMock(shape=(1, 10))
    mock_tokenizer.decode.return_value = "response text"

    service = TinyLlamaService(model_name="test")
    history = [{"role": "user", "content": "previous"}, {"role": "assistant", "content": "reply"}]

    result = service.generate_response("current", history=history)

    mock_tokenizer.apply_chat_template.assert_called_once()
    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[-1]["content"] == "current"


@patch("app.model_service.AutoTokenizer.from_pretrained")
@patch("app.model_service.AutoModelForCausalLM.from_pretrained")
def test_generate_response_empty_history(mock_model_cls, mock_tokenizer_cls) -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model_cls.return_value = mock_model

    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    mock_tokenizer.eos_token_id = 2
    mock_model.generate.return_value = MagicMock(shape=(1, 20))
    mock_model.generate.return_value.__getitem__.side_effect = lambda x: MagicMock(shape=(1, 10))
    mock_tokenizer.decode.return_value = "response"

    service = TinyLlamaService(model_name="test")

    result = service.generate_response("message")

    mock_tokenizer.apply_chat_template.assert_called_once()
    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
