# CLAUDE.md - Home Assistant OpenAI-Compatible LLM Integration

## Project Overview

This project creates a Home Assistant custom integration that connects to any OpenAI-compatible API provider (such as LiteLLM, Ollama, LocalAI, text-generation-webui, etc.). It provides a unified interface for conversation agents and LLM services within Home Assistant.

**Repository Name:** `hass-openai-compatible` (or `ha-litellm-conversation`)

## Development Resources (MCP)

MCP servers are available for development assistance:

### Available MCP Servers
- **gitea** - Repository access, commits, issues, PRs, branch management
- **ref** - Reference documentation lookup
- **vibe-check** - Code review and quality assessment
- **semgrep** - Static analysis and security scanning

### Usage Guidelines
- Use **gitea** to check existing code, create branches, commit changes, manage issues/PRs
- Use **ref** to look up Home Assistant integration patterns, API docs, and best practices
- Use **vibe-check** before committing to validate code quality and style
- Use **semgrep** to scan for security issues and common bugs before finalizing code

## Goals

- Connect Home Assistant to any OpenAI-compatible API endpoint
- Mimic functionality of native `openai_conversation` and `google_generative_ai_conversation` integrations
- Support dynamic model discovery via API polling
- Flexible authentication: global token per provider OR per-model tokens
- HACS-ready from day one

## Architecture

### Integration Type
- **Config Flow Integration** - Full UI-based configuration
- **Conversation Agent** - Implements `conversation` platform for Assist
- **Service Provider** - Exposes services for automations

### Core Components

```
custom_components/openai_compatible/
├── __init__.py           # Integration setup, entry management
├── manifest.json         # HACS/HA metadata
├── config_flow.py        # UI configuration wizard
├── const.py              # Constants, defaults
├── conversation.py       # Conversation agent implementation
├── api.py                # OpenAI-compatible API client
├── strings.json          # UI strings (English)
├── translations/
│   └── en.json           # Translations
└── services.yaml         # Service definitions
```

## Technical Requirements

### Home Assistant
- Minimum HA version: 2024.1.0
- Python 3.11+
- Uses `aiohttp` for async HTTP (already in HA)

### Dependencies
- `openai` Python package (optional, for compatibility)
- Or direct `aiohttp` implementation for lighter footprint

### API Compatibility
Must support standard OpenAI endpoints:
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (streaming optional)

## Configuration Flow

### Step 1: Provider Setup
```yaml
fields:
  - name: Provider Name (display name)
  - base_url: API Base URL (e.g., http://litellm:4000/v1)
  - api_key: API Key (optional for some providers)
  - auth_mode: 
      - "global" (one key for all models)
      - "per_model" (configure key per model entry)
```

### Step 2: Model Selection
- Fetch available models from `GET /v1/models`
- Display selectable list
- Allow manual model ID entry (fallback)
- If `per_model` auth: prompt for model-specific key

### Step 3: Agent Configuration
```yaml
fields:
  - prompt_template: System prompt (default provided)
  - max_tokens: Max response tokens (default: 1024)
  - temperature: Response creativity (default: 0.7)
  - context_messages: Number of conversation turns to include (default: 5)
```

### Options Flow
- Reconfigure all settings post-setup
- Add/remove models without recreating entry
- Update API keys

## Implementation Details

### API Client (`api.py`)

```python
class OpenAICompatibleClient:
    """Async client for OpenAI-compatible APIs."""
    
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    async def list_models(self) -> list[dict]:
        """Fetch available models from /v1/models."""
        # Returns list of model objects with 'id' field
        
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Send chat completion request."""
        # POST to /v1/chat/completions
        # Handle streaming if enabled
        # Return assistant message content
```

### Conversation Agent (`conversation.py`)

```python
class OpenAICompatibleConversationAgent(ConversationEntity):
    """Conversation agent for OpenAI-compatible providers."""
    
    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process user input and return response."""
        # Build message history
        # Include system prompt
        # Call API
        # Return ConversationResult with response
```

### Entry Data Structure

```python
# Global auth mode
entry.data = {
    "provider_name": "LiteLLM",
    "base_url": "http://litellm:4000/v1",
    "api_key": "sk-xxx",  # Global key
    "auth_mode": "global",
    "models": [
        {
            "model_id": "gpt-4o",
            "display_name": "GPT-4o",
            "max_tokens": 4096,
            "temperature": 0.7,
        }
    ]
}

# Per-model auth mode
entry.data = {
    "provider_name": "Multi-Provider",
    "base_url": "http://litellm:4000/v1",
    "auth_mode": "per_model",
    "models": [
        {
            "model_id": "gpt-4o",
            "display_name": "GPT-4o",
            "api_key": "sk-openai-xxx",  # Model-specific
            "max_tokens": 4096,
        },
        {
            "model_id": "claude-3-opus",
            "display_name": "Claude 3 Opus",
            "api_key": "sk-anthropic-xxx",  # Different key
            "max_tokens": 4096,
        }
    ]
}
```

## HACS Requirements

### manifest.json
```json
{
  "domain": "openai_compatible",
  "name": "OpenAI Compatible Conversation",
  "codeowners": ["@yourusername"],
  "config_flow": true,
  "dependencies": [],
  "documentation": "https://github.com/yourusername/hass-openai-compatible",
  "integration_type": "service",
  "iot_class": "cloud_polling",
  "issue_tracker": "https://github.com/yourusername/hass-openai-compatible/issues",
  "requirements": [],
  "version": "1.0.0"
}
```

### hacs.json
```json
{
  "name": "OpenAI Compatible Conversation",
  "render_readme": true,
  "homeassistant": "2024.1.0"
}
```

### Repository Structure
```
/
├── .github/
│   └── workflows/
│       ├── validate.yaml      # HACS validation
│       └── release.yaml       # Release automation
├── custom_components/
│   └── openai_compatible/
│       └── ... (integration files)
├── hacs.json
├── README.md
├── LICENSE                    # Required for HACS
└── CLAUDE.md                  # This file
```

## Services

### `openai_compatible.generate_response`
Direct API call for automations:
```yaml
service: openai_compatible.generate_response
data:
  config_entry_id: "xxx"
  model: "gpt-4o"
  prompt: "Summarize the weather forecast"
  max_tokens: 500
response_variable: llm_response
```

### `openai_compatible.reload_models`
Refresh model list from provider:
```yaml
service: openai_compatible.reload_models
data:
  config_entry_id: "xxx"
```

## Error Handling

- Connection errors: Graceful degradation, retry with backoff
- Auth errors: Clear error message, prompt reconfiguration
- Model not found: Fallback or clear error
- Rate limiting: Respect `Retry-After` headers
- Timeout: Configurable timeout (default 30s)

## Testing Strategy

### Unit Tests
- API client mock responses
- Config flow validation
- Message formatting

### Integration Tests
- Real LiteLLM instance (Docker)
- Model listing
- Chat completions

### Test Fixtures
```python
# conftest.py
@pytest.fixture
def mock_openai_api():
    """Mock OpenAI-compatible API responses."""
    with aioresponses() as m:
        m.get(
            "http://test:4000/v1/models",
            payload={"data": [{"id": "gpt-4o"}, {"id": "claude-3"}]}
        )
        yield m
```

## Development Commands

```bash
# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt

# Run tests
pytest tests/ -v

# Lint
ruff check custom_components/
black custom_components/

# Type check
mypy custom_components/

# Validate for HACS
docker run --rm -v $(pwd):/repo ghcr.io/hacs/action:main
```

## Reference Implementations

Study these for patterns:
- `homeassistant/components/openai_conversation/` - Native OpenAI integration
- `homeassistant/components/google_generative_ai_conversation/` - Native Gemini
- `homeassistant/components/ollama/` - Native Ollama (local LLM)

## Milestones

### v0.1.0 - MVP
- [ ] Basic config flow (URL, API key, model selection)
- [ ] Model polling from `/v1/models`
- [ ] Conversation agent working
- [ ] Single model per entry

### v0.2.0 - Enhanced Auth
- [ ] Global vs per-model authentication
- [ ] Multiple models per provider entry
- [ ] Options flow for reconfiguration

### v0.3.0 - Production Ready
- [ ] Streaming responses (if HA supports)
- [ ] Service calls for automations
- [ ] Comprehensive error handling
- [ ] HACS submission

### v1.0.0 - Feature Complete
- [ ] Function calling support
- [ ] Vision/image input support
- [ ] Token usage tracking/sensors
- [ ] Full test coverage

## Notes

- LiteLLM proxy URL typically: `http://<host>:4000/v1`
- Test with LiteLLM using `--detailed_debug` for troubleshooting
- Some providers may not implement `/v1/models` - provide manual entry fallback
- Consider caching model list to reduce API calls

## Quick Start for Development

1. Clone to `config/custom_components/openai_compatible/`
2. Restart Home Assistant
3. Add integration via UI
4. Enter LiteLLM URL: `http://litellm:4000/v1`
5. Select model from discovered list
6. Use as conversation agent in Assist
