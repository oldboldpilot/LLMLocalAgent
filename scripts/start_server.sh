#!/bin/bash
#
# Gemma 3 12B Server with Transformers CUDA + JAX CPU
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$PROJECT_ROOT/models/gemma-3-12b-it-quantized"
SERVER_HOST="127.0.0.1"
SERVER_PORT="12345"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_model() {
    log_info "Checking model at $MODEL_PATH..."
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model not found at $MODEL_PATH"
        exit 1
    fi
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    log_info "Model found (size: $MODEL_SIZE)"
}

start_vllm() {
    log_info "Starting vLLM server..."
    log_info "Model: $MODEL_PATH"
    log_info "API: http://$SERVER_HOST:$SERVER_PORT/v1"
    echo ""

    nohup uv run vllm serve "$MODEL_PATH" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 > /tmp/vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm.pid

    log_info "vLLM server started (PID: $VLLM_PID)"

    for i in {1..120}; do
        if curl -s "http://$SERVER_HOST:$SERVER_PORT/health" > /dev/null 2>&1; then
            log_info "vLLM server is healthy!"
            break
        fi
        sleep 2
    done
}

start_mcp() {
    log_info "Starting MCP server on port 12346..."

    export VLLM_API_URL="http://localhost:12345/v1"
    export MODEL_NAME="gemma-3-12b-it"
    export SERVER_PORT="12346"

    nohup uv run "$SCRIPT_DIR/mcp_server.py" > /tmp/mcp.log 2>&1 &
    MCP_PID=$!
    echo $MCP_PID > /tmp/mcp.pid

    log_info "MCP started (PID: $MCP_PID)"

    for i in {1..10}; do
        if curl -s "http://localhost:12346/health" > /dev/null 2>&1; then
            log_info "MCP is healthy!"
            break
        fi
        sleep 1
    done
}

main() {
    echo "=============================================="
    echo "  Gemma 3 12B Local Server (uv)"
    echo "  (vLLM GPU + MCP JAX CPU)"
    echo "=============================================="
    echo ""

    MODEL_PATH="$PROJECT_ROOT/models/gemma-3-12b-it-quantized"
    check_model
    echo ""
    start_vllm
    echo ""
    start_mcp

    echo ""
    echo "=============================================="
    echo "  Servers Ready!"
    echo "=============================================="
    echo ""
    echo "vLLM API:      http://localhost:12345/v1"
    echo "MCP Tools:     http://localhost:12346/mcp/tools"
    echo "Health:        http://localhost:12345/health"
    echo ""
}

main "$@"
