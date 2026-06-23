#!/usr/bin/env bash
# =============================================================================
# Launch the local vLLM engine for Qwen/Qwen2.5-7B-Instruct with the Indian
# market-data MCP server ATTACHED via --tool-server, exposing an
# OpenAI-compatible endpoint with native (server-side) tool calling.
#
# This script starts vLLM ONLY. Start the MCP tool server first, in a separate
# terminal:
#     python mcp_server.py          # serves SSE on 127.0.0.1:8001 (per config.yaml)
#
# Then run this script, then start the CLI:
#     bash serve.sh                 # this file (vLLM on :8000, attaches MCP)
#     python cli.py                 # interactive agent
#
# Prereqs on the serving host (NOT needed by the CLI client):
#     pip install "vllm>=0.6.0" fastmcp
#
# Security defaults:
#   - Binds to 127.0.0.1 only (not 0.0.0.0) to prevent external exposure.
#   - Requires a bearer token via SPARKS_API_KEY:
#       export SPARKS_API_KEY="$(openssl rand -hex 32)"
#     The CLI reads the same value from env SPARKS_API_KEY (preferred) or
#     local_inference_settings.api_key in config.yaml.
#
# Variables (mirror config.yaml):
#   SPARKS_HOST / SPARKS_PORT       -> local_inference_settings.host / .port
#   SPARKS_API_KEY                  -> local_inference_settings.api_key (required)
#   SPARKS_MODEL                    -> model_selection.model
#   SPARKS_TOOL_SERVER_URL          -> mcp.tool_server_url
#   SPARKS_MAX_MODEL_LEN / _GPU_UTIL / _TOOL_PARSER
# =============================================================================
set -euo pipefail

MODEL="${SPARKS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${SPARKS_HOST:-127.0.0.1}"
PORT="${SPARKS_PORT:-8000}"
MAX_MODEL_LEN="${SPARKS_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${SPARKS_GPU_UTIL:-0.92}"                  # ~129 GB of an H200's 140 GB
TOOL_PARSER="${SPARKS_TOOL_PARSER:-hermes}"          # Qwen2.5-family tool-call parser
TOOL_SERVER_URL="${SPARKS_TOOL_SERVER_URL:-http://127.0.0.1:8001/sse}"

if [[ -z "${SPARKS_API_KEY:-}" ]]; then
  echo "ERROR: SPARKS_API_KEY is not set." >&2
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

# --- One-liner equivalent (copy/paste) --------------------------------------
# Terminal 1:  python mcp_server.py
# Terminal 2:
#   SPARKS_API_KEY="$(openssl rand -hex 32)" \
#   vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
#     --gpu-memory-utilization 0.92 --max-model-len 32768 \
#     --enable-auto-tool-choice --tool-call-parser hermes \
#     --tool-server http://127.0.0.1:8001/sse \
#     --api-key "${SPARKS_API_KEY}"
# Terminal 3:  SPARKS_API_KEY="<same key>" python cli.py
