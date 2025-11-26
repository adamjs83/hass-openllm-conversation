"""Tests for the config flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from aioresponses import aioresponses
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.openllm_conversation.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    DOMAIN,
)


@pytest.mark.asyncio
async def test_form_user_step(
    hass: HomeAssistant,
    mock_aioresponse: aioresponses,
    mock_models_response: list[dict[str, Any]],
) -> None:
    """Test we get the user form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {}


@pytest.mark.asyncio
async def test_form_invalid_url(hass: HomeAssistant) -> None:
    """Test we handle invalid URL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_BASE_URL: "not-a-valid-url",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {CONF_BASE_URL: "invalid_url_format"}


@pytest.mark.asyncio
async def test_form_invalid_url_scheme(hass: HomeAssistant) -> None:
    """Test we handle invalid URL scheme."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_BASE_URL: "ftp://localhost:4000",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {CONF_BASE_URL: "invalid_url_scheme"}


@pytest.mark.asyncio
async def test_form_cannot_connect(
    hass: HomeAssistant,
    mock_aioresponse: aioresponses,
) -> None:
    """Test we handle cannot connect error."""
    # Don't mock the URL - will cause connection error
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.openllm_conversation.config_flow.OpenLLMApiClient.list_models",
        side_effect=Exception("Connection refused"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_BASE_URL: "http://localhost:4000/v1",
            },
        )

    # Should proceed to model step with manual entry option
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"


@pytest.mark.asyncio
async def test_full_flow(
    hass: HomeAssistant,
    mock_aioresponse: aioresponses,
    mock_models_response: list[dict[str, Any]],
) -> None:
    """Test complete config flow."""
    mock_aioresponse.get(
        "http://localhost:4000/v1/models",
        payload={"data": mock_models_response},
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Step 1: Enter URL
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_BASE_URL: "http://localhost:4000/v1",
            CONF_API_KEY: "test-key",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"

    # Step 2: Select model
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: "gpt-4o",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure"

    # Step 3: Configure agent
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "prompt_template": "You are a helpful assistant.",
            "max_tokens": 1024,
            "temperature": 0.7,
            "context_messages": 5,
            "timeout": 30,
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "OpenLLM Conversation (gpt-4o)"
    assert result["data"][CONF_BASE_URL] == "http://localhost:4000/v1"
    assert result["data"][CONF_MODEL] == "gpt-4o"
    assert result["options"]["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_manual_model_entry(
    hass: HomeAssistant,
    mock_aioresponse: aioresponses,
) -> None:
    """Test manual model entry when API doesn't return models."""
    mock_aioresponse.get(
        "http://localhost:4000/v1/models",
        payload={"data": []},
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_BASE_URL: "http://localhost:4000/v1",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"

    # Use manual entry
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "manual_model": "custom-model",
        },
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "configure"
