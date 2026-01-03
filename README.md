# Qwen3 Coder 32B Local Server

Local vLLM inference server for Qwen3 Coder 32B with OpenAI-compatible API.

## Quick Start

### 1. Set Hugging Face Token

```bash
export HF_TOKEN="your_huggingface_token_here"
```

### 2. Download Model

```bash
python scripts/download_model.py
```

This downloads Qwen3-Coder-32B-Instruct (~65GB) to `models/Qwen3-Coder-32B-Instruct/`.

### 3. Start Server

```bash
./scripts/start_server.sh
```

Server starts on `http://localhost:12345/v1` with OpenAI-compatible endpoints.

## IDE Configuration

### Claude Code
Use `configs/claude-code.json`:
```json
{
  "api_type": "openai",
  "base_url": "http://localhost:12345/v1",
  "model": "Qwen/Qwen3-Coder-32B-Instruct",
  "api_key": "not-needed-for-local"
}
```

### Cursor
Use `configs/cursor.json`:
```json
{
  "api": "OpenAI Compatible",
  "baseUrl": "http://localhost:12345/v1",
  "model": "Qwen/Qwen3-Coder-32B-Instruct",
  "Key": "not-needed-for-local"
}
```

### VS Code Copilot
Configure in settings.json:
```json
{
  "github.copilot.chat.assistant.baseUrl": "http://localhost:12345/v1"
}
```

### OpenCode
Configure endpoint: `http://localhost:12345/v1`

## Available Endpoints

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List models
- `GET /health` - Health check

## Model Details

- **Model**: Qwen3-Coder-32B-Instruct
- **Context Length**: 131,072 tokens
- **Parameters**: 32B
- **Quantization**: A3B (Activation-aware Quantized Mixture-of-Experts)
