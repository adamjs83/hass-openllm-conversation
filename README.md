# OpenLLM Conversation for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/release/adamjs83/hass-openllm-conversation.svg)](https://github.com/adamjs83/hass-openllm-conversation/releases)

A Home Assistant custom integration that connects to any OpenAI-compatible API provider, enabling conversation agents powered by your choice of LLM backend.

## Why I Made This

I've been experimenting with AI agents inside Home Assistant and wanted more flexibility than the built-in LLM integrations offer. Specifically, I wanted to:

- **Centralize my LLM access** - Use a single proxy (LiteLLM) to manage multiple AI providers
- **Have more choice** - Not be limited to just the providers Home Assistant natively supports
- **Access advanced features** - Connect to LLMs that have access to vector stores, MCP (Model Context Protocol), and other tools through LiteLLM

This is a work in progress. Feedback, suggestions, and contributions are welcome!

## Supported Providers

This integration works with any OpenAI-compatible API, including:

- **[LiteLLM](https://github.com/BerriAI/litellm)** - Unified proxy for 100+ LLM providers
- **[Ollama](https://ollama.ai/)** - Run LLMs locally
- **[LocalAI](https://localai.io/)** - Self-hosted OpenAI alternative
- **[text-generation-webui](https://github.com/oobabooga/text-generation-webui)** - Web UI with API support
- **[vLLM](https://vllm.ai/)** - High-throughput LLM serving
- **[OpenRouter](https://openrouter.ai/)** - Multi-provider API gateway
- Any other service implementing the OpenAI API specification

## Features

- **Dynamic Model Discovery** - Automatically fetches available models from your provider
- **Manual Model Entry** - Fallback for providers that don't support model listing
- **Configurable System Prompt** - Customize the assistant's personality and behavior
- **Conversation History** - Maintains context across conversation turns
- **Full UI Configuration** - No YAML required, configure everything through the Home Assistant interface

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots menu → Custom repositories
3. Add `https://github.com/adamjs83/hass-openllm-conversation` as an Integration
4. Search for "OpenLLM Conversation" and install it
5. Restart Home Assistant

### Manual Installation

1. Download the latest release from GitHub
2. Copy the `custom_components/openllm_conversation` folder to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for "OpenLLM Conversation"
4. Enter your API endpoint URL (e.g., `http://litellm:4000/v1`)
5. Enter your API key (if required)
6. Select a model from the discovered list or enter one manually
7. Configure the system prompt and other settings

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| API Base URL | Your OpenAI-compatible API endpoint | Required |
| API Key | Authentication key (if required) | Optional |
| Model | The model ID to use | Required |
| System Prompt | Instructions for the assistant | Helpful assistant prompt |
| Max Tokens | Maximum response length | 1024 |
| Temperature | Response creativity (0-2) | 0.7 |
| Context Messages | Conversation turns to remember | 5 |
| Timeout | Request timeout in seconds | 30 |

## Usage

Once configured, the integration creates a conversation agent that you can:

- Use in **Assist** - Set it as your default conversation agent
- Call via **automations** - Send prompts and receive responses
- Access through the **conversation.process** service

### Example: Using in Assist

1. Go to **Settings** → **Voice assistants**
2. Create or edit an assistant
3. Select your OpenLLM agent under "Conversation agent"

### Example: Automation

```yaml
service: conversation.process
data:
  agent_id: conversation.openllm_conversation_gpt_4o
  text: "What's the weather like today?"
```

## Provider Setup Examples

### LiteLLM

```
Base URL: http://your-litellm-host:4000/v1
API Key: your-litellm-key (if configured)
```

### Ollama

```
Base URL: http://your-ollama-host:11434/v1
API Key: (leave empty)
```

### LocalAI

```
Base URL: http://your-localai-host:8080/v1
API Key: (leave empty)
```

## Troubleshooting

### Connection Failed

- Verify your API endpoint is accessible from Home Assistant
- Check that the URL includes `/v1` if required
- Ensure any required API key is correct

### No Models Found

- Some providers don't implement the `/v1/models` endpoint
- Use the manual model entry field to specify your model ID
- Check your provider's documentation for available model names

### Slow Responses

- Increase the timeout setting in options
- Consider using a faster model or provider
- Check your provider's load and capacity

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
