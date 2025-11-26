"""OpenLLM Conversation integration for Home Assistant.

This integration provides a conversation agent that connects to any
OpenAI-compatible API endpoint, such as LiteLLM, Ollama, LocalAI, etc.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .api import OpenLLMApiClient
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_TIMEOUT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenLLM Conversation from a config entry.

    Args:
        hass: Home Assistant instance.
        entry: The config entry to set up.

    Returns:
        True if setup was successful.
    """
    hass.data.setdefault(DOMAIN, {})

    # Create shared API client
    client = OpenLLMApiClient(
        base_url=entry.data[CONF_BASE_URL],
        api_key=entry.data.get(CONF_API_KEY),
        timeout=entry.options.get(
            CONF_TIMEOUT, entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        ),
    )

    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry.

    Args:
        hass: Home Assistant instance.
        entry: The config entry to unload.

    Returns:
        True if unload was successful.
    """
    unload_ok: bool = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        data = hass.data[DOMAIN].pop(entry.entry_id, None)
        if data and "client" in data:
            client: OpenLLMApiClient = data["client"]
            await client.close()

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update.

    Args:
        hass: Home Assistant instance.
        entry: The config entry that was updated.
    """
    await hass.config_entries.async_reload(entry.entry_id)
