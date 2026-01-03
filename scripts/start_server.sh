#!/bin/bash
#
# Qwen3 Coder 32B vLLM Server Launcher
# Single script to start the local code server with OpenAI-compatible API
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-Coder-30B-A3B-Instruct"
SERVER_HOST="0.0.0.0"
SERVER_PORT="12345"
GPU_MEMORY_UTILIZATION="0.90"
MAX_MODEL_LEN="131072"
TENSOR_PARALLEL_SIZE="1"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed"
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - NVIDIA drivers may not be installed"
        exit 1
    fi
    
    if ! python -c "import vllm" 2>/dev/null; then
        log_warn "vllm not found. Installing..."
        pip install vllm
    fi
    
    log_info "Dependencies OK"
}

check_model() {
    log_info "Checking model at $MODEL_PATH..."
    
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model not found at $MODEL_PATH"
        log_info "Please run: python scripts/download_model.py"
        exit 1
    fi
    
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    log_info "Model found (size: $MODEL_SIZE)"
}

check_gpu() {
    log_info "Checking GPU availability..."
    
    if ! nvidia-smi &> /dev/null; then
        log_error "No NVIDIA GPU detected"
        exit 1
    fi
    
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    log_info "GPU: $GPU_INFO"
}

start_server() {
    log_info "Starting vLLM server..."
    log_info "Model: Qwen3-Coder-30B-A3B-Instruct"
    log_info "API Endpoint: http://$SERVER_HOST:$SERVER_PORT/v1"
    log_info "Compatible with: OpenAI, Claude Code, OpenCode, Cursor, Copilot"
    echo ""
    
    export CUDA_VISIBLE_DEVICES=0
    
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --trust-remote-code \
        --dtype auto \
        --enforce-eager \
        --disable-log-requests \
        "$@"
}

health_check() {
    log_info "Checking server health..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            log_info "Server is healthy!"
            log_info "API ready at http://localhost:$SERVER_PORT/v1"
            return 0
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 2
    done
    
    log_error "Server health check failed"
    return 1
}

main() {
    echo "=============================================="
    echo "  Qwen3 Coder 32B Local Code Server"
    echo "=============================================="
    echo ""
    
    check_dependencies
    check_gpu
    check_model
    
    echo ""
    start_server "$@"
    
    health_check
}

main "$@"
