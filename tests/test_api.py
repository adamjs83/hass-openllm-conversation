"""Tests for the API client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.openllm_conversation.api import (
    OpenLLMApiClient,
    OpenLLMApiError,
    OpenLLMAuthError,
)


def test_normalize_base_url() -> None:
    """Test URL normalization."""
    # Without /v1
    client = OpenLLMApiClient("http://localhost:4000")
    assert client.base_url == "http://localhost:4000/v1"

    # With /v1
    client = OpenLLMApiClient("http://localhost:4000/v1")
    assert client.base_url == "http://localhost:4000/v1"

    # With trailing slash
    client = OpenLLMApiClient("http://localhost:4000/v1/")
    assert client.base_url == "http://localhost:4000/v1"


def test_sanitize_url_for_logging() -> None:
    """Test URL sanitization removes credentials."""
    # URL with credentials
    result = OpenLLMApiClient._sanitize_url_for_logging(
        "http://user:pass@localhost:4000/v1"
    )
    assert "user" not in result
    assert "pass" not in result
    assert "localhost:4000" in result

    # URL without credentials
    result = OpenLLMApiClient._sanitize_url_for_logging("http://localhost:4000/v1")
    assert result == "http://localhost:4000/v1"


def test_headers_with_api_key() -> None:
    """Test headers include API key when provided."""
    client = OpenLLMApiClient(
        "http://localhost:4000/v1",
        api_key="test-key",
    )
    headers = client._get_headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_headers_without_api_key() -> None:
    """Test headers without API key."""
    client = OpenLLMApiClient("http://localhost:4000/v1")
    headers = client._get_headers()

    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


async def test_list_models_success() -> None:
    """Test successful model listing."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
            ]
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.closed = False
    mock_session.close = AsyncMock()

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False  # Don't try to close our mock

    models = await client.list_models()

    assert len(models) == 2
    assert models[0]["id"] == "gpt-4o"


async def test_list_models_auth_error() -> None:
    """Test auth error on model listing."""
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1", api_key="bad-key")
    client._session = mock_session
    client._owns_session = False

    with pytest.raises(OpenLLMAuthError):
        await client.list_models()


async def test_list_models_forbidden() -> None:
    """Test forbidden error on model listing."""
    mock_response = AsyncMock()
    mock_response.status = 403
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False

    with pytest.raises(OpenLLMAuthError):
        await client.list_models()


async def test_chat_completion_success() -> None:
    """Test successful chat completion."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    }
                }
            ]
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False

    response = await client.chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response == "Hello! How can I help you today?"


async def test_chat_completion_no_choices() -> None:
    """Test chat completion with no choices."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"choices": []})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False

    with pytest.raises(OpenLLMApiError, match="No response choices"):
        await client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )


async def test_chat_completion_auth_error() -> None:
    """Test auth error on chat completion."""
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False

    with pytest.raises(OpenLLMAuthError):
        await client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )


async def test_test_connection() -> None:
    """Test connection test."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": [{"id": "gpt-4o"}]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.closed = False

    client = OpenLLMApiClient("http://localhost:4000/v1")
    client._session = mock_session
    client._owns_session = False

    result = await client.test_connection()

    assert result is True
