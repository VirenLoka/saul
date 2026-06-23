#!/usr/bin/env bash
# =============================================================================
# Launch the local LLM engine for the Financial Advisor agent.
#
# Choose the engine with SPARKS_ENGINE (default: vllm):
#   SPARKS_ENGINE=vllm    bash serve.sh   # vLLM + server-side MCP tools (--tool-server)
#   SPARKS_ENGINE=ollama  bash serve.sh   # Ollama daemon (OpenAI-compatible at /v1)
#
# Keep config.yaml's model_selection.provider in sync with SPARKS_ENGINE.
#
# vLLM mode (full MCP tool calling):
#   Start the MCP tool server FIRST, in a separate terminal:
#       python mcp_server.py          # serves SSE on 127.0.0.1:8001 (per config.yaml)
#   Then:
#       export SPARKS_API_KEY="$(openssl rand -hex 32)"
#       SPARKS_ENGINE=vllm bash serve.sh
#       python cli.py                 # Terminal 3
#   Prereqs (serving host): pip install "vllm>=0.6.0" fastmcp
#
# Ollama mode:
#   NOTE: Ollama has no --tool-server, so the MCP tools are NOT executed
#   server-side; the CLI runs tool-free for this engine (it still gets the
#   portfolio analysis in its system context). The mcp_server is not needed.
#       SPARKS_ENGINE=ollama bash serve.sh
#       python cli.py                 # with model_selection.provider: ollama
#   Prereqs (serving host): install Ollama (https://ollama.com).
#
# Variables (mirror config.yaml):
#   vLLM   : SPARKS_HOST / SPARKS_PORT / SPARKS_MODEL / SPARKS_API_KEY (required)
#            SPARKS_TOOL_SERVER_URL / SPARKS_MAX_MODEL_LEN / _GPU_UTIL / _TOOL_PARSER
#   Ollama : OLLAMA_MODEL / OLLAMA_BIND (host:port, default 127.0.0.1:11434)
# =============================================================================
set -euo pipefail

ENGINE="${SPARKS_ENGINE:-vllm}"   # vllm | ollama

# --------------------------------------------------------------------------- #
launch_vllm() {
  local MODEL HOST PORT MAX_MODEL_LEN GPU_UTIL TOOL_PARSER TOOL_SERVER_URL
  MODEL="${SPARKS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  HOST="${SPARKS_HOST:-127.0.0.1}"
  PORT="${SPARKS_PORT:-8000}"
  MAX_MODEL_LEN="${SPARKS_MAX_MODEL_LEN:-32768}"
  GPU_UTIL="${SPARKS_GPU_UTIL:-0.92}"                  # ~129 GB of an H200's 140 GB
  TOOL_PARSER="${SPARKS_TOOL_PARSER:-hermes}"          # Qwen2.5-family tool-call parser
  TOOL_SERVER_URL="${SPARKS_TOOL_SERVER_URL:-http://127.0.0.1:8001/sse}"

  if [[ -z "${SPARKS_API_KEY:-}" ]]; then
    echo "ERROR: SPARKS_API_KEY is not set (required for vLLM mode)." >&2
    echo "Generate one with: export SPARKS_API_KEY=\"\$(openssl rand -hex 32)\"" >&2
    exit 1
  fi

  echo "Starting vLLM:"
  echo "  model      : ${MODEL}"
  echo "  endpoint   : http://${HOST}:${PORT}/v1"
  echo "  tool-server: ${TOOL_SERVER_URL}  (MCP: indian-market-data)"
  echo "  ctx / gpu  : ${MAX_MODEL_LEN} tokens / ${GPU_UTIL}"
  echo "NOTE: start 'python mcp_server.py' BEFORE this so --tool-server can connect."

  # --enable-auto-tool-choice + --tool-call-parser: parse the model's tool calls.
  # --tool-server: connect to the MCP server and execute tools server-side.
  exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER}" \
    --tool-server "${TOOL_SERVER_URL}" \
    --api-key "${SPARKS_API_KEY}"
}

# --------------------------------------------------------------------------- #
launch_ollama() {
  local MODEL BIND
  MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
  BIND="${OLLAMA_BIND:-127.0.0.1:11434}"

  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: 'ollama' not found. Install it from https://ollama.com" >&2
    exit 1
  fi

  echo "Starting Ollama:"
  echo "  model    : ${MODEL}"
  echo "  endpoint : http://${BIND}/v1  (OpenAI-compatible)"
  echo "NOTE: Ollama has no --tool-server; MCP tools are NOT executed server-side."
  echo "      Set model_selection.provider: ollama in config.yaml."

  # Pull the model if missing (no-op if already present).
  echo "Pulling ${MODEL} (no-op if already present)..."
  ollama pull "${MODEL}"

  # If the Ollama background service already holds the port, 'ollama serve' will
  # exit with 'address already in use' — that's fine, it's already running.
  echo "Launching daemon on ${BIND} (Ctrl-C to stop)..."
  exec env OLLAMA_HOST="${BIND}" ollama serve
}

# --------------------------------------------------------------------------- #
case "${ENGINE}" in
  vllm)   launch_vllm ;;
  ollama) launch_ollama ;;
  *)
    echo "ERROR: unknown SPARKS_ENGINE='${ENGINE}'. Use 'vllm' or 'ollama'." >&2
    exit 1
    ;;
esac

# --- One-liner equivalents (copy/paste) -------------------------------------
# vLLM:
#   Terminal 1:  python mcp_server.py
#   Terminal 2:
#     SPARKS_API_KEY="$(openssl rand -hex 32)" \
#     vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
#       --gpu-memory-utilization 0.92 --max-model-len 32768 \
#       --enable-auto-tool-choice --tool-call-parser hermes \
#       --tool-server http://127.0.0.1:8001/sse \
#       --api-key "${SPARKS_API_KEY}"
#   Terminal 3:  SPARKS_API_KEY="<same key>" python cli.py
#
# Ollama:
#   ollama pull qwen2.5:7b-instruct
#   ollama serve                      # daemon on 127.0.0.1:11434
#   python cli.py                     # config: model_selection.provider: ollama
