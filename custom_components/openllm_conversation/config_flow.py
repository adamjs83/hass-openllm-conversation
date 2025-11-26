"""Config flow for OpenLLM Conversation integration."""

from __future__ import annotations

import logging
from typing import Any, Final
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api import (
    OpenLLMApiClient,
    OpenLLMApiError,
    OpenLLMAuthError,
    OpenLLMConnectionError,
)
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CONTEXT_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT_TEMPLATE,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_CONTEXT_MESSAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

MANUAL_MODEL_ENTRY: Final = "__manual__"
VALID_URL_SCHEMES: Final = frozenset({"http", "https"})


def _build_options_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    """Build the options schema for agent configuration.

    This helper function creates the voluptuous schema for configuring
    the conversation agent options, avoiding duplication between the
    initial config flow and the options flow.

    Args:
        defaults: Optional dictionary of default values for the form fields.

    Returns:
        A voluptuous Schema for the options form.
    """
    defaults = defaults or {}

    return vol.Schema(
        {
            vol.Optional(
                CONF_PROMPT_TEMPLATE,
                default=defaults.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE),
            ): TemplateSelector(),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=defaults.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=1,
                    max=32768,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=defaults.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=2,
                    step=0.1,
                    mode=NumberSelectorMode.SLIDER,
                )
            ),
            vol.Optional(
                CONF_CONTEXT_MESSAGES,
                default=defaults.get(CONF_CONTEXT_MESSAGES, DEFAULT_CONTEXT_MESSAGES),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TIMEOUT,
                default=defaults.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=5,
                    max=300,
                    step=5,
                    mode=NumberSelectorMode.BOX,
                )
            ),
        }
    )


def _validate_url(url: str) -> str | None:
    """Validate a URL and return an error key if invalid.

    Args:
        url: The URL to validate.

    Returns:
        An error key string if validation fails, None if valid.
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return "invalid_url_format"
        if parsed.scheme not in VALID_URL_SCHEMES:
            return "invalid_url_scheme"
    except Exception:  # noqa: BLE001
        return "invalid_url_format"
    return None


class OpenLLMConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenLLM Conversation.

    This config flow guides users through setting up an OpenAI-compatible
    API connection in three steps:
    1. Provider configuration (URL and API key)
    2. Model selection (from API or manual entry)
    3. Agent configuration (prompt, temperature, etc.)
    """

    VERSION: int = 1

    _base_url: str | None
    _api_key: str | None
    _models: list[dict[str, Any]]
    _model_fetch_failed: bool
    _selected_model: str

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._base_url = None
        self._api_key = None
        self._models = []
        self._model_fetch_failed = False
        self._selected_model = ""

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OpenLLMOptionsFlow:
        """Get the options flow for this handler.

        Args:
            config_entry: The config entry to configure options for.

        Returns:
            An options flow handler instance.
        """
        return OpenLLMOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - provider configuration.

        Args:
            user_input: Form data submitted by the user.

        Returns:
            The next step in the flow or an error form.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_BASE_URL]

            # Validate URL format
            url_error = _validate_url(base_url)
            if url_error:
                errors[CONF_BASE_URL] = url_error
            else:
                # Check for duplicate entry
                await self.async_set_unique_id(base_url.lower().rstrip("/"))
                self._abort_if_unique_id_configured()

                self._base_url = base_url
                self._api_key = user_input.get(CONF_API_KEY)

                # Warn about insecure HTTP (but don't block)
                parsed = urlparse(base_url)
                if parsed.scheme == "http":
                    _LOGGER.warning(
                        "Using insecure HTTP connection to %s", parsed.netloc
                    )

                # Test connection and fetch models
                client = OpenLLMApiClient(
                    base_url=self._base_url,
                    api_key=self._api_key,
                )

                try:
                    self._models = await client.list_models()
                    self._model_fetch_failed = False
                except OpenLLMAuthError:
                    errors["base"] = "invalid_auth"
                except OpenLLMConnectionError:
                    errors["base"] = "cannot_connect"
                except OpenLLMApiError as err:
                    _LOGGER.warning("Failed to fetch models: %s", err)
                    # Allow proceeding with manual model entry
                    self._models = []
                    self._model_fetch_failed = True
                finally:
                    await client.close()

            if not errors:
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.URL)
                    ),
                    vol.Optional(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
            description_placeholders={
                "example_url": "http://litellm:4000/v1",
            },
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle model selection step.

        Args:
            user_input: Form data submitted by the user.

        Returns:
            The next step in the flow or an error form.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            model = user_input.get(CONF_MODEL)
            manual_model = user_input.get("manual_model")

            # Use manual entry if selected or if it's provided
            if model == MANUAL_MODEL_ENTRY or (not model and manual_model):
                model = manual_model

            if not model:
                errors["base"] = "no_model_selected"
            else:
                self._selected_model = model
                return await self.async_step_configure()

        # Build model options
        model_options: list[dict[str, str]] = []
        if self._models:
            model_options = [
                {"value": m.get("id", ""), "label": m.get("id", "Unknown")}
                for m in self._models
                if m.get("id")
            ]

        # Always add manual entry option
        model_options.append(
            {"value": MANUAL_MODEL_ENTRY, "label": "Enter model manually..."}
        )

        schema_dict: dict[vol.Marker, Any] = {
            vol.Optional(CONF_MODEL): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }

        # Show manual entry field if model fetch failed or no models found
        if self._model_fetch_failed or not self._models:
            schema_dict[vol.Optional("manual_model")] = TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT)
            )

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={
                "model_count": str(len(self._models)),
                "fetch_failed": str(self._model_fetch_failed),
            },
        )

    async def async_step_configure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle agent configuration step.

        Args:
            user_input: Form data submitted by the user.

        Returns:
            A config entry creation or the configuration form.
        """
        if user_input is not None:
            # Create the config entry with connection data
            data: dict[str, Any] = {
                CONF_BASE_URL: self._base_url,
                CONF_API_KEY: self._api_key,
                CONF_MODEL: self._selected_model,
            }

            # Options go in the options dict
            options: dict[str, Any] = {
                CONF_PROMPT_TEMPLATE: user_input.get(
                    CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE
                ),
                CONF_MAX_TOKENS: user_input.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                CONF_TEMPERATURE: user_input.get(
                    CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                ),
                CONF_CONTEXT_MESSAGES: user_input.get(
                    CONF_CONTEXT_MESSAGES, DEFAULT_CONTEXT_MESSAGES
                ),
                CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
            }

            # Use model name as title
            title = f"{DEFAULT_NAME} ({self._selected_model})"

            return self.async_create_entry(title=title, data=data, options=options)

        return self.async_show_form(
            step_id="configure",
            data_schema=_build_options_schema(),
            description_placeholders={
                "model": self._selected_model,
            },
        )


class OpenLLMOptionsFlow(OptionsFlow):
    """Handle options flow for OpenLLM Conversation.

    This allows users to reconfigure the conversation agent settings
    after initial setup.
    """

    config_entry: ConfigEntry

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow.

        Args:
            config_entry: The config entry being configured.
        """
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options.

        Args:
            user_input: Form data submitted by the user.

        Returns:
            An entry creation or the options form.
        """
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current values from options (with fallback to data for migration)
        current_options = self.config_entry.options
        current_data = self.config_entry.data

        defaults = {
            CONF_PROMPT_TEMPLATE: current_options.get(
                CONF_PROMPT_TEMPLATE,
                current_data.get(CONF_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE),
            ),
            CONF_MAX_TOKENS: current_options.get(
                CONF_MAX_TOKENS,
                current_data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ),
            CONF_TEMPERATURE: current_options.get(
                CONF_TEMPERATURE,
                current_data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ),
            CONF_CONTEXT_MESSAGES: current_options.get(
                CONF_CONTEXT_MESSAGES,
                current_data.get(CONF_CONTEXT_MESSAGES, DEFAULT_CONTEXT_MESSAGES),
            ),
            CONF_TIMEOUT: current_options.get(
                CONF_TIMEOUT,
                current_data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
            ),
        }

        return self.async_show_form(
            step_id="init",
            data_schema=_build_options_schema(defaults),
        )
