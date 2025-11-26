"""Conversation agent for OpenLLM Conversation integration."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Final, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid

from .api import OpenLLMApiClient, OpenLLMApiError
from .const import (
    CONF_CONTEXT_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT_TEMPLATE,
    CONF_TEMPERATURE,
    DEFAULT_CONTEXT_MESSAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Memory management constants
CONVERSATION_TIMEOUT: Final = timedelta(hours=24)
MAX_CONVERSATIONS: Final = 100


class ConversationData:
    """Container for conversation history data.

    Attributes:
        messages: List of message dictionaries with role and content.
        last_used: Timestamp of last interaction.
    """

    messages: list[dict[str, str]]
    last_used: datetime

    __slots__ = ("messages", "last_used")

    def __init__(self) -> None:
        """Initialize conversation data."""
        self.messages = []
        self.last_used = datetime.now()

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = datetime.now()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities.

    Args:
        hass: Home Assistant instance.
        config_entry: The config entry for this integration.
        async_add_entities: Callback to add entities to Home Assistant.
    """
    client: OpenLLMApiClient = hass.data[DOMAIN][config_entry.entry_id]["client"]
    agent = OpenLLMConversationEntity(config_entry, client)
    async_add_entities([agent])


class OpenLLMConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenLLM conversation agent entity.

    This entity provides a conversation agent that communicates with
    OpenAI-compatible API endpoints for natural language processing.

    Attributes:
        config_entry: The associated config entry.
    """

    _attr_has_entity_name: bool = True
    _attr_name: str | None = None

    config_entry: ConfigEntry
    _client: OpenLLMApiClient
    _conversation_history: dict[str, ConversationData]

    def __init__(self, config_entry: ConfigEntry, client: OpenLLMApiClient) -> None:
        """Initialize the conversation entity.

        Args:
            config_entry: The config entry for this entity.
            client: Shared API client instance.
        """
        self.config_entry = config_entry
        self._attr_unique_id = config_entry.entry_id
        self._client = client
        self._conversation_history = {}

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for this entity.

        Returns:
            Device information for the device registry.
        """
        return DeviceInfo(
            identifiers={(DOMAIN, self.config_entry.entry_id)},
            name=self.config_entry.title,
            manufacturer="OpenLLM",
            model=self.config_entry.data.get(CONF_MODEL, "Unknown"),
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages.

        Returns:
            "*" indicating all languages are supported.
        """
        return "*"

    def _get_option(self, key: str, default: Any = None) -> Any:
        """Get configuration value from options with fallback to data.

        Args:
            key: The configuration key to retrieve.
            default: Default value if key not found.

        Returns:
            The configuration value.
        """
        return self.config_entry.options.get(
            key, self.config_entry.data.get(key, default)
        )

    def _cleanup_old_conversations(self) -> None:
        """Remove old conversation history to prevent memory leaks.

        Removes conversations that haven't been used within CONVERSATION_TIMEOUT,
        and also limits total conversations to MAX_CONVERSATIONS.
        """
        now = datetime.now()

        # Remove timed-out conversations
        to_remove = [
            conv_id
            for conv_id, data in self._conversation_history.items()
            if now - data.last_used > CONVERSATION_TIMEOUT
        ]
        for conv_id in to_remove:
            del self._conversation_history[conv_id]
            _LOGGER.debug("Removed expired conversation %s", conv_id)

        # Limit total conversations
        if len(self._conversation_history) > MAX_CONVERSATIONS:
            # Sort by last used and keep only the most recent
            sorted_convs = sorted(
                self._conversation_history.items(),
                key=lambda x: x[1].last_used,
            )
            for conv_id, _ in sorted_convs[:-MAX_CONVERSATIONS]:
                del self._conversation_history[conv_id]
            _LOGGER.debug(
                "Trimmed conversation history to %d entries", MAX_CONVERSATIONS
            )

    async def async_added_to_hass(self) -> None:
        """Handle entity being added to hass.

        Registers this entity as a conversation agent.
        """
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.config_entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity being removed from hass.

        Unregisters this entity as a conversation agent.
        """
        conversation.async_unset_agent(self.hass, self.config_entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence and return the response.

        Args:
            user_input: The user's input to process.

        Returns:
            The conversation result with the assistant's response.
        """
        # Cleanup old conversations periodically
        self._cleanup_old_conversations()

        # Get configuration
        model = self.config_entry.data.get(CONF_MODEL)
        prompt_template = self._get_option(
            CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE
        )
        max_tokens = self._get_option(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature = self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        context_messages = self._get_option(
            CONF_CONTEXT_MESSAGES, DEFAULT_CONTEXT_MESSAGES
        )

        # Get or create conversation ID
        conversation_id = user_input.conversation_id or ulid.ulid_now()

        # Get or create conversation history
        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = ConversationData()

        conv_data = self._conversation_history[conversation_id]
        conv_data.touch()

        # Build messages list
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt_template}
        ]

        # Add conversation history (limited by context_messages)
        context_messages = int(context_messages)
        if context_messages > 0 and conv_data.messages:
            # Each turn has 2 messages (user + assistant), so multiply by 2
            max_history = context_messages * 2
            messages.extend(conv_data.messages[-max_history:])

        # Add current user message
        messages.append({"role": "user", "content": user_input.text})

        # Trace the request
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "model": model},
        )

        try:
            # Call the API
            response_text = await self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )

            # Update conversation history
            conv_data.messages.append({"role": "user", "content": user_input.text})
            conv_data.messages.append({"role": "assistant", "content": response_text})

            # Limit history size within conversation
            max_history_size = (context_messages + 5) * 2
            if len(conv_data.messages) > max_history_size:
                conv_data.messages = conv_data.messages[-max_history_size:]

            # Create response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response_text)

            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

        except OpenLLMApiError as err:
            _LOGGER.error("Error calling OpenLLM API: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error communicating with AI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )
