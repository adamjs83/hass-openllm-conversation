"""Constants for OpenLLM Conversation integration."""

DOMAIN = "openllm_conversation"

# Configuration keys
CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_PROMPT_TEMPLATE = "prompt_template"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_CONTEXT_MESSAGES = "context_messages"
CONF_TIMEOUT = "timeout"

# Defaults
DEFAULT_NAME = "OpenLLM Conversation"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_MESSAGES = 5
DEFAULT_TIMEOUT = 30
DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant for Home Assistant.
Answer questions about the smart home and help with automations.
Be concise and helpful."""

# API endpoints
ENDPOINT_MODELS = "/models"
ENDPOINT_CHAT_COMPLETIONS = "/chat/completions"
