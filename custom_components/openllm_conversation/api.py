"""API client for OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
from typing import Any, Final
from urllib.parse import urlparse, urlunparse

import aiohttp

from .const import (
    DEFAULT_TIMEOUT,
    ENDPOINT_CHAT_COMPLETIONS,
    ENDPOINT_MODELS,
)

_LOGGER = logging.getLogger(__name__)

# HTTP status codes
HTTP_UNAUTHORIZED: Final = 401
HTTP_FORBIDDEN: Final = 403
HTTP_OK: Final = 200


class OpenLLMApiError(Exception):
    """Base exception for API errors."""


class OpenLLMAuthError(OpenLLMApiError):
    """Authentication error."""


class OpenLLMConnectionError(OpenLLMApiError):
    """Connection error."""


class OpenLLMApiClient:
    """Async client for OpenAI-compatible APIs.

    This client handles communication with OpenAI-compatible API endpoints,
    supporting model listing and chat completions.

    Attributes:
        base_url: The normalized base URL for API requests.
        timeout: The aiohttp client timeout configuration.
    """

    base_url: str
    api_key: str | None
    timeout: aiohttp.ClientTimeout
    _session: aiohttp.ClientSession | None
    _owns_session: bool

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the API client.

        Args:
            base_url: Base URL of the API (e.g., http://litellm:4000/v1).
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            session: Optional shared aiohttp session. If not provided,
                     a new session will be created per request.
        """
        self.base_url = self._normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session = session
        self._owns_session = session is None

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Normalize the base URL to ensure consistent format.

        Args:
            base_url: The raw base URL input.

        Returns:
            Normalized URL with trailing slash removed and /v1 path ensured.
        """
        url = base_url.rstrip("/")
        if not url.endswith("/v1") and "/v1" not in url:
            url = f"{url}/v1"
        return url

    @staticmethod
    def _sanitize_url_for_logging(url: str) -> str:
        """Remove credentials from URL for safe logging.

        Args:
            url: The URL that may contain credentials.

        Returns:
            URL with username and password removed.
        """
        try:
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                # Reconstruct netloc without credentials
                netloc = parsed.hostname or ""
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                sanitized = parsed._replace(netloc=netloc)
                return urlunparse(sanitized)
        except Exception:  # noqa: BLE001
            # If parsing fails, return a generic placeholder
            return "<url>"
        return url

    def _get_headers(self) -> dict[str, str]:
        """Get request headers.

        Returns:
            Dictionary of HTTP headers for API requests.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session.

        Returns:
            An aiohttp ClientSession for making requests.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from /v1/models.

        Returns:
            List of model objects with 'id' field.

        Raises:
            OpenLLMAuthError: If authentication fails.
            OpenLLMConnectionError: If connection fails or times out.
            OpenLLMApiError: If the API returns an error.
        """
        url = f"{self.base_url}{ENDPOINT_MODELS}"
        safe_url = self._sanitize_url_for_logging(self.base_url)
        _LOGGER.debug("Fetching models from %s", safe_url)

        try:
            session = await self._get_session()
            async with session.get(url, headers=self._get_headers()) as response:
                if response.status == HTTP_UNAUTHORIZED:
                    raise OpenLLMAuthError("Invalid API key")
                if response.status == HTTP_FORBIDDEN:
                    raise OpenLLMAuthError("API key not authorized")
                if response.status != HTTP_OK:
                    text = await response.text()
                    raise OpenLLMApiError(
                        f"Failed to fetch models: {response.status} - {text}"
                    )

                data = await response.json()
                models: list[dict[str, Any]] = data.get("data", [])
                _LOGGER.debug("Found %d models", len(models))
                return models

        except aiohttp.ClientConnectorError as err:
            raise OpenLLMConnectionError(
                f"Failed to connect to {safe_url}: {err}"
            ) from err
        except TimeoutError as err:
            raise OpenLLMConnectionError(
                f"Timeout connecting to {safe_url}"
            ) from err

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Send chat completion request.

        Args:
            model: Model ID to use.
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens in response.
            temperature: Response creativity (0-2).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Assistant message content.

        Raises:
            OpenLLMAuthError: If authentication fails.
            OpenLLMConnectionError: If connection fails or times out.
            OpenLLMApiError: If the API returns an error or no response.
        """
        url = f"{self.base_url}{ENDPOINT_CHAT_COMPLETIONS}"
        safe_url = self._sanitize_url_for_logging(self.base_url)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        _LOGGER.debug("Sending chat completion request to %s", safe_url)

        try:
            session = await self._get_session()
            async with session.post(
                url, headers=self._get_headers(), json=payload
            ) as response:
                if response.status == HTTP_UNAUTHORIZED:
                    raise OpenLLMAuthError("Invalid API key")
                if response.status == HTTP_FORBIDDEN:
                    raise OpenLLMAuthError("API key not authorized")
                if response.status != HTTP_OK:
                    text = await response.text()
                    raise OpenLLMApiError(
                        f"Chat completion failed: {response.status} - {text}"
                    )

                data = await response.json()
                choices: list[dict[str, Any]] = data.get("choices", [])
                if not choices:
                    raise OpenLLMApiError("No response choices returned")

                message: dict[str, Any] = choices[0].get("message", {})
                content: str = message.get("content", "")
                _LOGGER.debug("Received response with %d characters", len(content))
                return content

        except aiohttp.ClientConnectorError as err:
            raise OpenLLMConnectionError(
                f"Failed to connect to {safe_url}: {err}"
            ) from err
        except TimeoutError as err:
            raise OpenLLMConnectionError(
                f"Timeout waiting for response from {safe_url}"
            ) from err

    async def test_connection(self) -> bool:
        """Test the connection to the API.

        Returns:
            True if connection is successful.

        Raises:
            OpenLLMApiError: If the connection fails.
        """
        await self.list_models()
        return True
