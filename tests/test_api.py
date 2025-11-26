"""Tests for the API client."""

from __future__ import annotations

from typing import Any

import pytest
from aioresponses import aioresponses

from custom_components.openllm_conversation.api import (
    OpenLLMApiClient,
    OpenLLMApiError,
    OpenLLMAuthError,
)


class TestOpenLLMApiClient:
    """Tests for OpenLLMApiClient."""

    def test_normalize_base_url(self) -> None:
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

    def test_sanitize_url_for_logging(self) -> None:
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

    @pytest.mark.asyncio
    async def test_list_models_success(
        self,
        mock_aioresponse: aioresponses,
        mock_models_response: list[dict[str, Any]],
    ) -> None:
        """Test successful model listing."""
        mock_aioresponse.get(
            "http://localhost:4000/v1/models",
            payload={"data": mock_models_response},
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")
        models = await client.list_models()
        await client.close()

        assert len(models) == 3
        assert models[0]["id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_list_models_auth_error(
        self,
        mock_aioresponse: aioresponses,
    ) -> None:
        """Test auth error on model listing."""
        mock_aioresponse.get(
            "http://localhost:4000/v1/models",
            status=401,
        )

        client = OpenLLMApiClient("http://localhost:4000/v1", api_key="bad-key")

        with pytest.raises(OpenLLMAuthError):
            await client.list_models()

        await client.close()

    @pytest.mark.asyncio
    async def test_list_models_forbidden(
        self,
        mock_aioresponse: aioresponses,
    ) -> None:
        """Test forbidden error on model listing."""
        mock_aioresponse.get(
            "http://localhost:4000/v1/models",
            status=403,
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")

        with pytest.raises(OpenLLMAuthError):
            await client.list_models()

        await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_success(
        self,
        mock_aioresponse: aioresponses,
        mock_chat_response: dict[str, Any],
    ) -> None:
        """Test successful chat completion."""
        mock_aioresponse.post(
            "http://localhost:4000/v1/chat/completions",
            payload=mock_chat_response,
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")
        response = await client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        await client.close()

        assert response == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_chat_completion_no_choices(
        self,
        mock_aioresponse: aioresponses,
    ) -> None:
        """Test chat completion with no choices."""
        mock_aioresponse.post(
            "http://localhost:4000/v1/chat/completions",
            payload={"choices": []},
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")

        with pytest.raises(OpenLLMApiError, match="No response choices"):
            await client.chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_auth_error(
        self,
        mock_aioresponse: aioresponses,
    ) -> None:
        """Test auth error on chat completion."""
        mock_aioresponse.post(
            "http://localhost:4000/v1/chat/completions",
            status=401,
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")

        with pytest.raises(OpenLLMAuthError):
            await client.chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        await client.close()

    @pytest.mark.asyncio
    async def test_test_connection(
        self,
        mock_aioresponse: aioresponses,
        mock_models_response: list[dict[str, Any]],
    ) -> None:
        """Test connection test."""
        mock_aioresponse.get(
            "http://localhost:4000/v1/models",
            payload={"data": mock_models_response},
        )

        client = OpenLLMApiClient("http://localhost:4000/v1")
        result = await client.test_connection()
        await client.close()

        assert result is True

    @pytest.mark.asyncio
    async def test_headers_with_api_key(self) -> None:
        """Test headers include API key when provided."""
        client = OpenLLMApiClient(
            "http://localhost:4000/v1",
            api_key="test-key",
        )
        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

        await client.close()

    @pytest.mark.asyncio
    async def test_headers_without_api_key(self) -> None:
        """Test headers without API key."""
        client = OpenLLMApiClient("http://localhost:4000/v1")
        headers = client._get_headers()

        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

        await client.close()
