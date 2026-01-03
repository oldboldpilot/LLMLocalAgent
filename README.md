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

**Option 1: Settings JSON (Recommended)**

Add to `.vscode/settings.json`:
```json
{
  "github.copilot.chat.assistant.enabled": true,
  "github.copilot.chat.assistant.name": "Qwen3 Coder",
  "github.copilot.chat.assistant.baseUrl": "http://localhost:12345/v1",
  "github.copilot.chat.assistant.model": "Qwen/Qwen3-Coder-32B-Instruct"
}
```

**Option 2: VS Code Settings UI**

1. Open VS Code Settings (`Ctrl+,`)
2. Search for "Copilot Chat"
3. Set:
   - **Chat > Assistant > Enabled**: Checked
   - **Chat > Assistant > Name**: `Qwen3 Coder`
   - **Chat > Assistant > Base Url**: `http://localhost:12345/v1`
   - **Chat > Assistant > Model**: `Qwen/Qwen3-Coder-32B-Instruct`

### Claude Code (VS Code Extension)

**Install Extension:**
1. Open VS Code Extensions (`Ctrl+Shift+X`)
2. Search for "Claude Code"
3. Install the official Anthropic extension

**Configure:**

Add to `.vscode/settings.json`:
```json
{
  "claude-code.apiProvider": "custom",
  "claude-code.customEndpoint": "http://localhost:12345/v1",
  "claude-code.model": "Qwen/Qwen3-Coder-32B-Instruct",
  "claude-code.apiKey": "not-needed-for-local"
}
```

**Or via Settings UI:**

1. Extensions > Claude Code > Configuration
2. Set:
   - **Api Provider**: `Custom`
   - **Custom Endpoint**: `http://localhost:12345/v1`
   - **Model**: `Qwen/Qwen3-Coder-32B-Instruct`
   - **Api Key**: `not-needed-for-local`

### OpenCode

**CLI Configuration:**

```bash
export OPENCODE_API_BASE="http://localhost:12345/v1"
export OPENCODE_MODEL="Qwen/Qwen3-Coder-32B-Instruct"
```

**Or create a config file at `~/.opencode/config.json`:**
```json
{
  "apiBase": "http://localhost:12345/v1",
  "model": "Qwen/Qwen3-Coder-32B-Instruct"
}
```

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

## Troubleshooting

**Server won't start:**
- Check if port 12345 is in use: `lsof -i :12345`
- Ensure model is downloaded: `ls models/Qwen3-Coder-32B-Instruct/`
- Verify vLLM is installed: `pip show vllm`

**Connection refused:**
- Server running? Check process: `ps aux | grep vllm`
- Correct port? Default is `12345`
- Firewall blocking? Allow port `12345`

**Out of memory:**
- Reduce `gpu-memory-utilization` in `start_server.sh`
- Enable tensor parallelism for multi-GPU
