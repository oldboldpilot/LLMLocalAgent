#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server with JAX CPU Acceleration

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    MCP Server (Port 12346)                   │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  HTTP REST API Layer                                 │   │
    │  │  - GET  /health        → Health check                │   │
    │  │  - GET  /mcp/tools    → List available tools        │   │
    │  │  - POST /mcp/call     → Execute tool                │   │
    │  │  - POST /v1/chat/completions → Chat completion      │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                           │                                 │
    │                           ▼                                 │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  JAX Processor (CPU, JIT-compiled)                   │   │
    │  │                                                      │   │
    │  │  Text Processing (JAX CPU):                          │   │
    │  │  - encode()        → Text to fixed vector            │   │
    │  │  - similarity()    → Cosine similarity (JIT)         │   │
    │  │  - find_similar()  → Batch similarity search (JIT)   │   │
    │  │                                                      │   │
    │  │  All functions use @jax.jit for XLA optimization     │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                           │                                 │
    │                           ▼                                 │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  httpx AsyncClient                                   │   │
    │  │  → Calls vLLM at localhost:12345/v1 (GPU)            │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 vLLM Server (Port 12345)                     │
    │                                                              │
    │  GPU: PyTorch CUDA with:                                     │
    │  - Continuous batching for high throughput                   │
    │  - PagedAttention for efficient KV cache                     │
    │  - Tensor parallelism ready                                  │
    │  - Dynamic batching                                          │
    └─────────────────────────────────────────────────────────────┘

Usage:
    export VLLM_API_URL="http://localhost:12345/v1"
    export MODEL_NAME="gemma-3-12b-it"
    python scripts/mcp_server.py
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import jax
import jax.numpy as jnp
import httpx


# =============================================================================
# Configuration
# =============================================================================

VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:12345/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-3-12b-it")
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "12346"))


# =============================================================================
# JAX Processor (CPU)
# =============================================================================


class JAXProcessor:
    """
    JAX-accelerated text processing for CPU operations.

    All methods use @jax.jit for JIT compilation to XLA,
    providing significant speedup for repeated operations.
    """

    def __init__(self, dim: int = 256) -> None:
        """
        Initialize JAX processor.

        Args:
            dim: Vector dimension for text encoding (default: 256)
        """
        self.dim = dim
        self._init_jit_functions()

    def _init_jit_functions(self) -> None:
        """Initialize JIT-compiled functions for maximum performance."""

        # JIT-compiled fixed-length encoding
        @jax.jit
        def _encode_fixed(chars: jax.Array, dim: int) -> jax.Array:
            """Encode character array to fixed-length float vector."""
            padded = jnp.zeros(dim, dtype=jnp.float32)
            valid_len = jnp.minimum(dim, len(chars))
            padded = padded.at[:valid_len].set(
                jnp.array(chars[:valid_len], dtype=jnp.float32)
            )
            return padded

        # JIT-compiled cosine similarity
        @jax.jit
        def _cosine_sim(v1: jax.Array, v2: jax.Array) -> float:
            """Compute cosine similarity between two vectors."""
            dot = jnp.dot(v1, v2)
            n1 = jnp.linalg.norm(v1)
            n2 = jnp.linalg.norm(v2)
            denom = n1 * n2 + 1e-8
            return float(dot / denom) if denom > 0 else 0.0

        # JIT-compiled batch similarity
        @jax.jit
        def _batch_similarity(
            query_v: jax.Array, candidate_vecs: jax.Array
        ) -> jax.Array:
            """Compute similarities between query and all candidates."""
            dots = jnp.dot(candidate_vecs, query_v)
            norms = jnp.linalg.norm(candidate_vecs, axis=1)
            query_norm = jnp.linalg.norm(query_v)
            return dots / (norms * query_norm + 1e-8)

        self._encode_fixed = _encode_fixed
        self._cosine_sim = _cosine_sim
        self._batch_similarity = _batch_similarity

    def encode(self, text: str) -> jax.Array:
        """
        Encode text to fixed-length JAX array.

        Args:
            text: Input text string

        Returns:
            JAX array of shape (dim,) with float32 values
        """
        chars = list(text.encode("utf-8")[: self.dim])
        chars_array = jnp.array(chars, dtype=jnp.float32)
        return self._encode_fixed(chars_array, self.dim)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Cosine similarity score between 0 and 1
        """
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)
        return self._cosine_sim(vec1, vec2)

    def find_similar(
        self, query: str, candidates: list[str], top_k: int = 5
    ) -> list[dict]:
        """
        Find most similar candidates using batched JAX computation.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of dicts with index, text, and similarity score
        """
        query_vec = self.encode(query)
        candidate_vecs = jnp.array([self.encode(c) for c in candidates])
        similarities = self._batch_similarity(query_vec, candidate_vecs)
        top_indices = jnp.argsort(similarities)[::-1][:top_k]

        return [
            {
                "index": int(idx),
                "text": candidates[int(idx)][:100]
                + ("..." if len(candidates[int(idx)]) > 100 else ""),
                "similarity": float(similarities[idx]),
            }
            for idx in top_indices
        ]


jax_processor = JAXProcessor()


# =============================================================================
# LLM Client (calls vLLM on GPU)
# =============================================================================


async def call_llm(messages: list[dict], **kwargs) -> str:
    """
    Call LLM via vLLM API.

    Args:
        messages: List of message dicts with 'role' and 'content'
        **kwargs: Additional API parameters (temperature, max_tokens, etc.)

    Returns:
        Generated response text from LLM
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.0,
                **kwargs,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# =============================================================================
# MCP Tools Definition
# =============================================================================

MCP_TOOLS = [
    {
        "name": "complete_code",
        "description": "Complete code at cursor position",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code_context": {"type": "string", "description": "Code before cursor"},
                "code_after": {"type": "string", "description": "Code after cursor"},
                "language": {"type": "string", "default": "python"},
            },
            "required": ["code_context"],
        },
    },
    {
        "name": "explain_code",
        "description": "Explain what code does",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "detail_level": {
                    "type": "string",
                    "enum": ["brief", "medium", "detailed"],
                },
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "refactor_code",
        "description": "Refactor or improve code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "goal": {"type": "string"},
                "constraints": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["code", "goal"],
        },
    },
    {
        "name": "debug_code",
        "description": "Debug and fix issues in code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "error_message": {"type": "string"},
                "expected_behavior": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_tests",
        "description": "Write comprehensive unit tests",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "framework": {"type": "string", "default": "pytest"},
                "test_types": {"type": "array", "items": {"type": "string"}},
                "coverage_target": {"type": "integer", "default": 80},
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "generate_docs",
        "description": "Generate documentation for code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "doc_format": {
                    "type": "string",
                    "enum": ["markdown", "google", "javadoc"],
                },
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_code",
        "description": "Write new code from requirements",
        "inputSchema": {
            "type": "object",
            "properties": {
                "requirements": {"type": "string"},
                "language": {"type": "string"},
                "framework": {"type": "string"},
                "constraints": {"type": "string"},
            },
            "required": ["requirements"],
        },
    },
    {
        "name": "find_bugs",
        "description": "Static analysis to find potential bugs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string"},
                "severity_level": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "optimize_code",
        "description": "Optimize code for performance",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "goal": {"type": "string", "enum": ["speed", "memory", "both"]},
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "convert_code",
        "description": "Convert code from one language to another",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "source_language": {"type": "string"},
                "target_language": {"type": "string"},
                "preserve_comments": {"type": "boolean"},
            },
            "required": ["code", "source_language", "target_language"],
        },
    },
    {
        "name": "find_similar_code",
        "description": "Find similar code patterns using JAX vectorization (CPU)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "candidates": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["code", "candidates"],
        },
    },
    {
        "name": "chat",
        "description": "General purpose chat with coding assistant",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "system_prompt": {"type": "string"},
                "temperature": {"type": "number"},
            },
            "required": ["message"],
        },
    },
]


def build_llm_messages(tool_name: str, args: dict) -> list[dict]:
    """Build LLM messages based on tool name and arguments."""
    lang = args.get("language", "python")
    code = args.get("code", "") or args.get("code_context", "")
    error_msg = args.get("error_message", "")

    prompts = {
        "complete_code": (
            f"expert {lang} programmer",
            f"Complete this {lang} code:\n\n```{lang}\n{code}\n```",
        ),
        "explain_code": (
            "code documentation expert",
            f"Explain this code {'in detail' if args.get('detail_level') == 'detailed' else 'briefly'}:\n\n```{lang}\n{code}\n```",
        ),
        "refactor_code": (
            "software architect",
            f"Refactor this code to improve {args.get('goal', 'readability')}:\n\n```{lang}\n{code}\n```",
        ),
        "debug_code": (
            "expert debugger",
            f"Debug this code. Error: {error_msg}\n\n```{lang}\n{code}\n```",
        ),
        "write_tests": (
            f"testing expert ({args.get('framework', 'pytest')})",
            f"Write tests for:\n\n```{lang}\n{code}\n```",
        ),
        "generate_docs": (
            "documentation expert",
            f"Generate docs for:\n\n```{lang}\n{code}\n```",
        ),
        "write_code": (
            "expert programmer",
            f"Write code:\n{args.get('requirements', '')}\n\nLanguage: {lang}\nFramework: {args.get('framework', '')}",
        ),
        "find_bugs": (
            "security expert",
            f"Analyze for bugs:\n\n```{lang}\n{code}\n```",
        ),
        "optimize_code": (
            "performance expert",
            f"Optimize for {args.get('goal', 'speed')}:\n\n```{lang}\n{code}\n```",
        ),
        "convert_code": (
            "polyglot programmer",
            f"Convert from {args.get('source_language', '')} to {args.get('target_language', '')}:\n\n```{args.get('source_language', '')}\n{code}\n```",
        ),
        "chat": (
            args.get("system_prompt", "coding assistant"),
            args.get("message", ""),
        ),
    }

    role, content = prompts.get(tool_name, ("assistant", ""))
    return [
        {"role": "system", "content": f"You are an expert {role}."},
        {"role": "user", "content": content},
    ]


# =============================================================================
# HTTP Request Handler
# =============================================================================


class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP protocol."""

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, data: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path == "/health":
            self.send_json(
                {
                    "status": "ok",
                    "jax_version": jax.__version__,
                    "jax_devices": str(jax.devices()),
                }
            )
        elif path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": 0,
                    "owned_by": "vllm"
                }]
            })
        elif path == "/mcp/tools":
            self.send_json({"tools": MCP_TOOLS})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if path == "/v1/chat/completions":
            self.handle_chat_v1(body)
        elif path == "/mcp/call":
            self.handle_mcp_call(body)
        else:
            self.send_json({"error": "Not found"}, 404)

    def handle_chat_v1(self, body: dict) -> None:
        """Handle OpenAI-compatible chat completions."""
        try:
            import asyncio
            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.0)
            max_tokens = body.get("max_tokens", 1024)
            
            result = asyncio.run(call_llm(messages, temperature=temperature, max_tokens=max_tokens))
            
            self.send_json({
                "id": f"chatcmpl-{abs(hash(result)) % 1000000}",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            })
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_mcp_call(self, body: dict) -> None:
        """Handle MCP tool call."""
        try:
            name = body.get("name", "")
            args = body.get("arguments", {})

            if name == "find_similar_code":
                result = json.dumps(
                    jax_processor.find_similar(
                        args.get("code", ""),
                        args.get("candidates", []),
                        args.get("top_k", 5),
                    )
                )
            elif name == "jax_similarity":
                sim = jax_processor.similarity(
                    args.get("text1", ""), args.get("text2", "")
                )
                result = json.dumps({"similarity": sim})
            elif any(name == t["name"] for t in MCP_TOOLS):
                import asyncio

                messages = build_llm_messages(name, args)
                result = asyncio.run(call_llm(messages))
            else:
                result = f"Unknown tool: {name}"

            self.send_json({"result": result})
        except Exception as e:
            import traceback

            self.send_json({"error": f"{e}\n{traceback.format_exc()}"}, 500)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Start the MCP server."""
    print(f"""
    ═══════════════════════════════════════════════════════════
      MCP Server with JAX CPU Acceleration
    ═══════════════════════════════════════════════════════════
      Backend (GPU):  {VLLM_API_URL}
      Model:          {MODEL_NAME}
      Server:         http://{SERVER_HOST}:{SERVER_PORT}
      JAX:            {jax.__version__} on {jax.devices()}
    ═══════════════════════════════════════════════════════════
      Endpoints:
        GET  /health        → Health check
        GET  /mcp/tools     → List tools
        POST /mcp/call      → Call tool
    ═══════════════════════════════════════════════════════════
    """)

    server = HTTPServer((SERVER_HOST, SERVER_PORT), MCPHandler)
    print(f"Serving on http://{SERVER_HOST}:{SERVER_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
