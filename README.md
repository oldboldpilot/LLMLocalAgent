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

## Quick Start

### 1. Start vLLM Server (GPU - Port 12345)

```bash
cd /home/muyiwa/Development/LLMLocalAgent

# Start vLLM with Gemma 3 12B
nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model models/gemma-3-12b-it \
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

### Claude Code (`configs/claude-code.json`)

```json
{
  "name": "Gemma 3 12B vLLM",
  "model": "gemma-3-12b-it-quantized",
  "api_type": "openai",
  "base_url": "http://localhost:12345/v1",
  "api_key": "not-needed-for-local",
  "context_length": 16384,
  "temperature": 0.0,
  "max_tokens": 8192
}
```

### Cursor (`configs/cursor.json`)

```json
{
  "name": "Gemma 3 12B vLLM",
  "model": "gemma-3-12b-it-quantized",
  "api": "OpenAI Compatible",
  "baseUrl": "http://localhost:12345/v1",
  "Key": "not-needed-for-local",
  "context_length": 16384,
  "temperature": 0.0,
  "max_tokens": 16384
}
```

### OpenCode (`configs/opencode.json`)

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
| Model | gemma-3-12b-it |
| Parameters | 12B |
| Context | 32,768 tokens |
| Quantization | FP16 |
| Inference | vLLM (GPU) |
| Tools | MCP + JAX CPU |

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
