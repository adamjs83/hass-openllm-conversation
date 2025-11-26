"""AI Task entity for OpenLLM Conversation integration."""

from __future__ import annotations

import json
import logging
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task import AITaskEntity, AITaskEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import OpenLLMApiClient, OpenLLMApiError
from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Default system prompt for AI tasks
AI_TASK_SYSTEM_PROMPT = (
    "You are a Home Assistant AI assistant that helps users with tasks. "
    "Follow the user's instructions precisely. "
    "When asked to generate structured data, respond with valid JSON only."
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI Task entities.

    Args:
        hass: Home Assistant instance.
        config_entry: The config entry for this integration.
        async_add_entities: Callback to add entities to Home Assistant.
    """
    client: OpenLLMApiClient = hass.data[DOMAIN][config_entry.entry_id]["client"]
    entity = OpenLLMAITaskEntity(config_entry, client)
    async_add_entities([entity])


class OpenLLMAITaskEntity(AITaskEntity):
    """OpenLLM AI Task entity.

    This entity provides AI task capabilities using OpenAI-compatible APIs.
    It supports the GENERATE_DATA feature for generating text or structured data.
    """

    _attr_has_entity_name: bool = True
    _attr_name: str = "AI Task"
    _attr_supported_features: AITaskEntityFeature = AITaskEntityFeature.GENERATE_DATA

    def __init__(self, config_entry: ConfigEntry, client: OpenLLMApiClient) -> None:
        """Initialize the AI Task entity.

        Args:
            config_entry: The config entry for this entity.
            client: Shared API client instance.
        """
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_ai_task"
        self._client = client

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for this entity.

        Returns:
            Device information for the device registry.
        """
        return DeviceInfo(
            identifiers={(DOMAIN, self._config_entry.entry_id)},
            name=self._config_entry.title,
            manufacturer="OpenLLM",
            model=self._config_entry.data.get(CONF_MODEL, "Unknown"),
            entry_type=DeviceEntryType.SERVICE,
        )

    def _get_option(self, key: str, default: Any = None) -> Any:
        """Get configuration value from options with fallback to data.

        Args:
            key: The configuration key to retrieve.
            default: Default value if key not found.

        Returns:
            The configuration value.
        """
        return self._config_entry.options.get(
            key, self._config_entry.data.get(key, default)
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task.

        Args:
            task: The task containing instructions and optional structure.
            chat_log: The chat log for this task.

        Returns:
            The result containing generated data.
        """
        model = self._config_entry.data.get(CONF_MODEL)
        max_tokens = self._get_option(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)

        # Build the system prompt
        system_prompt = AI_TASK_SYSTEM_PROMPT
        if task.structure:
            system_prompt += (
                "\n\nYou must respond with valid JSON matching the requested structure. "
                "Do not include any text outside of the JSON object."
            )

        # Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.instructions},
        ]

        try:
            response_text = await self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )

            # If no structure requested, return raw text
            if not task.structure:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=response_text,
                )

            # Parse JSON response for structured output
            try:
                # Try to extract JSON from the response
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                _LOGGER.warning(
                    "Failed to parse JSON response, returning raw text: %s",
                    response_text[:100],
                )
                data = {"text": response_text}

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )

        except OpenLLMApiError as err:
            _LOGGER.error("Error calling OpenLLM API for AI task: %s", err)
            raise
