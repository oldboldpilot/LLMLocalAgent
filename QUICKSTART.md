# Local LLM Coding Assistant

Local coding agent with **vLLM** (GPU) + **MCP Protocol** (CPU/JAX) for IDE integration.

## Quick Start

### 1. Start vLLM (GPU - Port 12345)

```bash
cd /home/muyiwa/Development/LLMLocalAgent

nohup uv run vllm serve models/gemma-3-12b-it-quantized \
  --host 127.0.0.1 --port 12345 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 > /tmp/vllm.log 2>&1 &
```

### 2. Start MCP Server (CPU - Port 12346)

```bash
export VLLM_API_URL="http://localhost:12345/v1"
export MODEL_NAME="gemma-3-12b-it-quantized"

nohup uv run python scripts/mcp_server.py > /tmp/mcp.log 2>&1 &
```

### 3. Verify

```bash
# vLLM health
curl http://localhost:12345/health

# MCP health  
curl http://localhost:12346/health

# MCP tools
curl http://localhost:12346/mcp/tools
```

### Or Use Startup Script

```bash
./scripts/start_server.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Coding Agent                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────┐       ┌─────────────────────┐      │
│   │   Port 12345        │       │   Port 12346        │      │
│   │   vLLM Server       │       │   MCP Server        │      │
│   │   GPU CUDA          │       │   CPU JAX           │      │
│   │   - Transformers    │       │   - 13 Tools        │      │
│   │   - PagedAttention  │       │   - JIT compiled    │      │
│   │   - Continuous      │       │   - Text similarity │      │
│   │     Batching        │       │                     │      │
│   └─────────────────────┘       └─────────────────────┘      │
│            ▲                               ▲                 │
│            │ GPU │ CPU                     │                 │
│            └─────┴─────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Why vLLM?

| Feature | vLLM | Transformers |
|---------|------|--------------|
| Throughput | 2-4x higher | Baseline |
| Memory | PagedAttention | Standard |
| Batching | Continuous auto | Static |

## MCP Tools (13 total)

| Tool | Description | Backend |
|------|-------------|---------|
| `complete_code` | Code completion | vLLM |
| `explain_code` | Explain code | vLLM |
| `refactor_code` | Refactor code | vLLM |
| `debug_code` | Debug and fix | vLLM |
| `write_tests` | Generate tests | vLLM |
| `generate_docs` | Create docs | vLLM |
| `write_code` | Write new code | vLLM |
| `find_bugs` | Bug analysis | vLLM |
| `optimize_code` | Performance | vLLM |
| `convert_code` | Language trans. | vLLM |
| `find_similar_code` | Similarity | JAX CPU |
| `chat` | General chat | vLLM |

## IDE Configuration

### Claude Code

Claude Code supports OpenAI-compatible APIs. Import or configure manually:

**Import config:**
```bash
# In Claude Code, use the Model Configuration panel
# Import configs/claude-code.json
```

**Manual configuration:**
```json
{
  "name": "Gemma 3 12B Local",
  "model": "gemma-3-12b-it-quantized",
  "api_type": "openai",
  "base_url": "http://localhost:12345/v1",
  "api_key": "not-needed-for-local",
  "context_length": 16384,
  "temperature": 0.0,
  "max_tokens": 16384
}
```

**Verify:**
```bash
curl http://localhost:12345/v1/models
```

### Cursor

**Settings > AI Features > Models:**
```json
{
  "name": "Gemma 3 12B Local",
  "model": "gemma-3-12b-it-quantized",
  "api": "OpenAI Compatible",
  "baseUrl": "http://localhost:12345/v1",
  "Key": "not-needed-for-local",
  "context_length": 16384,
  "temperature": 0.0,
  "max_tokens": 16384
}
```

### OpenCode

**Import config:**
```bash
# In OpenCode, import configs/opencode.json
```

**Manual configuration:**
```json
[
  {
    "name": "Gemma 3 12B Local vLLM",
    "model": "gemma-3-12b-it-quantized",
    "api_type": "openai",
    "base_url": "http://localhost:12345/v1",
    "api_key": "not-needed-for-local",
    "context_length": 16384,
    "temperature": 0.0,
    "max_tokens": 16384
  },
  {
    "name": "Local Agent MCP",
    "model": "gemma-3-12b-it-quantized",
    "api_type": "openai",
    "base_url": "http://localhost:12346/v1",
    "api_key": "not-needed-for-local",
    "context_length": 16384,
    "temperature": 0.0,
    "max_tokens": 4096
  }
]
```

### GitHub Copilot (VS Code)

Copilot doesn't support custom endpoints, but use VS Code extensions like Continue:

**Continue (`~/.continue/config.json`):**
```json
{
  "models": [
    {
      "title": "Gemma 3 12B Local",
      "provider": "openai",
      "model": "gemma-3-12b-it-quantized",
      "apiKey": "not-needed-for-local",
      "baseUrl": "http://localhost:12345/v1"
    }
  ]
}
```

### Aider

```bash
aider --model gemma-3-12b-it-quantized --api-base http://localhost:12345/v1/
```

## MCP Tools (13 total)

| Tool | Description | Backend |
|------|-------------|---------|
| `complete_code` | Code completion | vLLM |
| `explain_code` | Explain code | vLLM |
| `refactor_code` | Refactor code | vLLM |
| `debug_code` | Debug and fix | vLLM |
| `write_tests` | Generate tests | vLLM |
| `generate_docs` | Create docs | vLLM |
| `write_code` | Write new code | vLLM |
| `find_bugs` | Bug analysis | vLLM |
| `optimize_code` | Performance | vLLM |
| `convert_code` | Language trans. | vLLM |
| `find_similar_code` | Similarity | JAX CPU |
| `chat` | General chat | vLLM |

## Files

```
LLMLocalAgent/
├── scripts/
│   ├── mcp_server.py      # MCP server with JAX CPU
│   ├── start_server.sh    # Startup script
│   └── download_model.py  # Model download
├── configs/
│   ├── claude-code.json   # Claude Code config
│   ├── cursor.json        # Cursor config
│   └── opencode.json      # OpenCode config
├── models/                # Model storage (gemma-3-12b-it-quantized)
├── README.md              # Full documentation
└── QUICKSTART.md          # This file
```
