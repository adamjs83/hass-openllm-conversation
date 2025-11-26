"""Fixtures for OpenLLM Conversation tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from aioresponses import aioresponses

from custom_components.openllm_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
)


@pytest.fixture
def mock_models_response() -> list[dict[str, Any]]:
    """Return mock models response."""
    return [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
        {"id": "claude-3-opus", "object": "model", "owned_by": "anthropic"},
    ]


@pytest.fixture
def mock_chat_response() -> dict[str, Any]:
    """Return mock chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }


@pytest.fixture
def mock_aioresponse() -> Generator[aioresponses, None, None]:
    """Return aioresponses mock."""
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_config_entry_data() -> dict[str, Any]:
    """Return mock config entry data."""
    return {
        CONF_BASE_URL: "http://localhost:4000/v1",
        CONF_API_KEY: "test-api-key",
        CONF_MODEL: "gpt-4o",
    }


@pytest.fixture
def mock_config_entry_options() -> dict[str, Any]:
    """Return mock config entry options."""
    return {
        "prompt_template": "You are a helpful assistant.",
        "max_tokens": 1024,
        "temperature": 0.7,
        "context_messages": 5,
        "timeout": 30,
    }
