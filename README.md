# Local LLM Coding Assistant

Local inference server with **vLLM** (GPU) + **MCP Protocol** (CPU/JAX) for code assistance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Coding Agent Stack                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   IDE / Client                          │   │
│   │   (Claude Code, Cursor, VS Code, OpenCode, Aider, etc.) │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│            ┌───────────────┴───────────────┐                    │
│            ▼                               ▼                    │
│   ┌─────────────────────┐       ┌─────────────────────┐         │
│   │   Port 12345        │       │   Port 12346        │         │
│   │   vLLM Server       │       │   MCP Server        │         │
│   │   - OpenAI REST     │       │   - 13 Tools        │         │
│   │   - GPU CUDA        │       │   - JAX CPU (JIT)   │         │
│   │   - Continuous      │       │   - HTTP REST       │         │
│   │     Batching        │       │                     │         │
│   │   - PagedAttention  │       │                     │         │
│   └─────────────────────┘       └─────────────────────┘         │
│            ▲                               ▲                    │
│            │                               │                    │
│            │          GPU │ CPU           │                    │
│            └──────────────┼───────────────┘                    │
│                           │                                     │
│                    ┌──────┴──────┐                             │
│                    │   Hardware  │                             │
│                    │   GPU+CPU   │                             │
│                    └─────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
LLMLocalAgent/                          # Project root directory
├── scripts/                            # Server scripts
│   ├── mcp_server.py                   # MCP server with JAX CPU tools
│   ├── start_server.sh                 # Startup script for both servers
│   └── download_model.py               # Model download utility
├── configs/                            # IDE configuration files
│   ├── claude-code.json                # Claude Code config
│   ├── cursor.json                     # Cursor config
│   └── opencode.json                   # OpenCode config
├── models/                             # Model storage directory
│   └── gemma-3-12b-it-quantized/       # Quantized model (7.9GB)
├── configs/                            # Additional config files
├── README.md                           # Full documentation
├── QUICKSTART.md                       # Quick start guide
└── pyproject.toml                      # Python dependencies
```

## Required Files & Locations

| File/Folder | Location | Purpose |
|-------------|----------|---------|
| `models/gemma-3-12b-it-quantized/` | Project root | Quantized model (7.9GB) |
| `scripts/mcp_server.py` | Project root | MCP server for coding tools |
| `scripts/start_server.sh` | Project root | Startup script |
| `configs/claude-code.json` | Project root | Claude Code configuration |
| `configs/cursor.json` | Project root | Cursor configuration |
| `configs/opencode.json` | Project root | OpenCode configuration |

### Model Location
- **Required:** `LLMLocalAgent/models/gemma-3-12b-it-quantized/`
- **Size:** ~7.9GB (quantized from 23GB)
- **Download:** Run `python scripts/download_model.py` or download from HuggingFace

### IDE Config Files
Copy or reference these config files from `LLMLocalAgent/configs/`:

| IDE | Config File | Import/Setup Location |
|-----|-------------|----------------------|
| Claude Code | `configs/claude-code.json` | Model Configuration panel |
| Cursor | `configs/cursor.json` | Settings > AI Features > Models |
| OpenCode | `configs/opencode.json` | Model settings import |
| Continue (VS Code) | `~/.continue/config.json` | Home directory |
| CodeGPT (VS Code) | Extension settings | VS Code Extensions |

## Quick Start

### 1. Start vLLM Server (GPU - Port 12345)

```bash
cd /home/muyiwa/Development/LLMLocalAgent

# Start vLLM with Gemma 3 12B quantized model
nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model models/gemma-3-12b-it-quantized \
  --host 127.0.0.1 --port 12345 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 > /tmp/vllm.log 2>&1 &

# Or use the startup script
./scripts/start_server.sh

# Verify
curl http://localhost:12345/health
```
  --host 127.0.0.1 --port 12345 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > /tmp/vllm.log 2>&1 &

# Verify
curl http://localhost:12345/health
```

### 2. Start MCP Server (CPU - Port 12346)

```bash
cd /home/muyiwa/Development/LLMLocalAgent

export VLLM_API_URL="http://localhost:12345/v1"
export MODEL_NAME="gemma-3-12b-it"

nohup .venv/bin/python scripts/mcp_server.py > /tmp/mcp.log 2>&1 &

# Verify
curl http://localhost:12346/health
```

### 3. Or Use Startup Script

```bash
./scripts/start_server.sh
```

## Why vLLM?

| Feature | vLLM | Vanilla Transformers |
|---------|------|---------------------|
| **Throughput** | 2-4x higher | Baseline |
| **Memory** | PagedAttention, 2x more efficient | Standard caching |
| **Batching** | Continuous automatic | Static, manual |
| **Latency** | Optimized decode | Standard |
| **Setup** | Production-ready | Research-focused |

## Verification

```bash
# vLLM health
curl http://localhost:12345/health

# vLLM models
curl http://localhost:12345/v1/models

# MCP health
curl http://localhost:12346/health

# MCP tools
curl http://localhost:12346/mcp/tools
```

## Usage Examples

### Direct vLLM API

```bash
curl -X POST http://localhost:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it",
    "messages": [{"role": "user", "content": "Write a Python factorial"}],
    "max_tokens": 200,
    "temperature": 0.0
  }'
```

### MCP Tool Call

```bash
curl -X POST http://localhost:12346/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "explain_code",
    "arguments": {
      "code": "def quicksort(arr): ...",
      "detail_level": "detailed"
    }
  }'
```

### JAX Similarity Search

```bash
curl -X POST http://localhost:12346/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "find_similar_code",
    "arguments": {
      "code": "def add(a, b): return a + b",
      "candidates": [
        "def sum_list(l): return sum(l)",
        "def mul(x, y): return x * y"
      ],
      "top_k": 2
    }
  }'
```

## IDE Configuration

### Claude Code

Claude Code supports OpenAI-compatible APIs. Import or configure manually:

**Option 1: Import config file**
```bash
# In Claude Code, use the Model Configuration panel
# Import configs/claude-code.json
```

**Option 2: Manual configuration**
```json
{
  "name": "Gemma 3 12B Local vLLM",
  "model": "gemma-3-12b-it-quantized",
  "api_type": "openai",
  "base_url": "http://localhost:12345/v1",
  "api_key": "not-needed-for-local",
  "context_length": 16384,
  "temperature": 0.0,
  "max_tokens": 16384
}
```

**Verify connection:**
```bash
curl http://localhost:12345/v1/models
```

### Cursor

Cursor supports OpenAI-compatible APIs for its AI features:

**Configuration:**
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

**Setup steps:**
1. Open Cursor Settings (Ctrl+,)
2. Go to AI Features > Models
3. Add new model with the configuration above

### GitHub Copilot (VS Code)

GitHub Copilot doesn't support custom endpoints directly, but you can use the **Copilot Free** with this local server as an alternative:

**Setup VS Code with local LLM:**

1. Install VS Code extension: "Continue" or "CodeGPT" or "Tabby ML"

2. Configure Continue extension (`~/.continue/config.json`):
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

3. Or use CodeGPT extension:
   - Settings > Extensions > CodeGPT
   - Provider: OpenAI
   - API Key: not-needed-for-local
   - Base URL: http://localhost:12345/v1
   - Model: gemma-3-12b-it-quantized

### OpenCode

OpenCode supports multiple model providers. Import or configure manually:

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

### Aider

Aider works with OpenAI-compatible APIs:

```bash
# Start aider with local model
aider --model gemma-3-12b-it-quantized --api-base http://localhost:12345/v1/

# Or set environment variables
export OPENAI_API_KEY="not-needed-for-local"
export OPENAI_API_BASE="http://localhost:12345/v1"
aider --model gemma-3-12b-it-quantized
```

## MCP Tools Reference

| Tool | Description | Location |
|------|-------------|----------|
| `complete_code` | Code completion | CPU (JAX→vLLM) |
| `explain_code` | Explain code | CPU→GPU |
| `refactor_code` | Refactor code | CPU→GPU |
| `debug_code` | Debug and fix | CPU→GPU |
| `write_tests` | Generate tests | CPU→GPU |
| `generate_docs` | Create docs | CPU→GPU |
| `write_code` | Write new code | CPU→GPU |
| `find_bugs` | Bug analysis | CPU→GPU |
| `optimize_code` | Performance | CPU→GPU |
| `convert_code` | Language trans. | CPU→GPU |
| `find_similar_code` | JAX similarity | CPU (JAX) |
| `list_files` | File listing | CPU |
| `read_file` | Read file | CPU |
| `chat` | General chat | CPU→GPU |

## vLLM vs MCP: What Runs Where

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM (Port 12345)                    │
│  GPU CUDA                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  - Transformer forward pass                          │   │
│  │  - Attention computation (PagedAttention)           │   │
│  │  - KV cache management                               │   │
│  │  - Continuous batching                               │   │
│  │  - Token generation ( autoregressive )              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTP/gRPC
                              │
┌─────────────────────────────────────────────────────────────┐
│                       MCP Server (Port 12346)                │
│  CPU JAX                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  - Tool dispatch and orchestration                   │   │
│  │  - Text encoding (JAX JIT)                           │   │
│  │  - Similarity computation (JAX JIT)                  │   │
│  │  - Batch similarity (JAX JIT)                        │   │
│  │  - HTTP request/response handling                    │   │
│  │  - File I/O operations                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Process Management

```bash
# Check vLLM
ps aux | grep vllm
lsof -i :12345

# Check MCP
ps aux | grep mcp_server
lsof -i :12346

# vLLM logs
tail -f /tmp/vllm.log

# MCP logs
tail -f /tmp/mcp.log

# Kill all
pkill -9 -f vllm
pkill -9 -f mcp_server
```

## Model Specifications

| Property | Value |
|----------|-------|
| Model | gemma-3-12b-it-quantized |
| Parameters | 12B |
| Context | 16,384 tokens |
| Quantization | Compressed (7.9GB) |
| Inference | vLLM (GPU) |
| Tools | MCP + JAX CPU |

## Project Files Summary

```
LLMLocalAgent/                          # Project root: /home/muyiwa/Development/LLMLocalAgent/
├── scripts/
│   ├── mcp_server.py                   # MCP server (Python 3.11+)
│   ├── start_server.sh                 # Start both servers
│   └── download_model.py               # Download model from HuggingFace
├── configs/
│   ├── claude-code.json                # Claude Code import
│   ├── cursor.json                     # Cursor import
│   └── opencode.json                   # OpenCode import
├── models/
│   └── gemma-3-12b-it-quantized/       # Model: ~7.9GB (required)
├── README.md                           # This file
├── QUICKSTART.md                       # Quick start guide
└── pyproject.toml                      # Dependencies: vllm, jax, httpx, mcp
```

### Required vs Optional Files

| Required | Description | Size |
|----------|-------------|------|
| `models/gemma-3-12b-it-quantized/` | Quantized model | ~7.9GB |
| `scripts/mcp_server.py` | MCP tools server | - |
| `scripts/start_server.sh` | Startup script | - |

| Optional | Description |
|----------|-------------|
| `configs/*.json` | IDE configuration files (import to IDE) |
| `pyproject.toml` | Python dependencies |

## Troubleshooting

**vLLM won't start:**
```bash
lsof -i :12345
tail /tmp/vllm.log
nvidia-smi
```

**MCP can't connect:**
```bash
curl http://localhost:12345/health
curl http://localhost:12346/health
tail /tmp/mcp.log
```

**Out of memory:**
```bash
nvidia-smi
# Reduce --gpu-memory-utilization in start command
```
