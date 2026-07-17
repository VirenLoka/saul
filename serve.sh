#!/usr/bin/env bash
# =============================================================================
# Launch the self-hosted vLLM engine for the Financial Advisor agent.
#
# vLLM is the ONLY self-hosting path (Ollama support was removed). It is also
# how DeepSeek models are served: point SPARKS_MODEL (or config.yaml's
# local_inference_settings.vllm.model) at a DeepSeek HF repo id, e.g.
#   SPARKS_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash serve.sh
#
# Keep config.yaml's model_selection.provider: vllm in sync with this script.
#
# REQUIRES A CUDA GPU. This script loads model weights and will fail on a
# CPU-only host — run it on the serving box, not a laptop.
#
# Full MCP tool calling:
#   Start the MCP tool server FIRST, in a separate terminal:
#       python mcp_server.py          # serves SSE on 127.0.0.1:8001 (per config.yaml)
#   Then:
#       export SPARKS_API_KEY="$(openssl rand -hex 32)"
#       bash serve.sh
#       python cli.py                 # Terminal 3
#   Prereqs (serving host): pip install "vllm>=0.6.0" fastmcp
#
# Variables (mirror config.yaml):
#   SPARKS_HOST / SPARKS_PORT / SPARKS_MODEL / SPARKS_API_KEY (required)
#   SPARKS_TOOL_SERVER_URL / SPARKS_MAX_MODEL_LEN / _GPU_UTIL / _TOOL_PARSER
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
launch_vllm() {
  local MODEL HOST PORT MAX_MODEL_LEN GPU_UTIL TOOL_PARSER TOOL_SERVER_URL
  MODEL="${SPARKS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  HOST="${SPARKS_HOST:-0.0.0.0}"
  PORT="${SPARKS_PORT:-8000}"
  MAX_MODEL_LEN="${SPARKS_MAX_MODEL_LEN:-32768}"
  GPU_UTIL="${SPARKS_GPU_UTIL:-0.92}"                  # ~129 GB of an H200's 140 GB
  TOOL_PARSER="${SPARKS_TOOL_PARSER:-hermes}"          # Qwen2.5-family tool-call parser
  TOOL_SERVER_URL="${SPARKS_TOOL_SERVER_URL:-172.23.0.5:8001}"

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
  # NOTE: for a DeepSeek model, set SPARKS_TOOL_PARSER=deepseek_v3 (the hermes
  # default is for the Qwen2.5 family).
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

launch_vllm

# --- One-liner equivalent (copy/paste) --------------------------------------
#   Terminal 1:  python mcp_server.py
#   Terminal 2:
#     SPARKS_API_KEY="$(openssl rand -hex 32)" \
#     vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
#       --gpu-memory-utilization 0.92 --max-model-len 32768 \
#       --enable-auto-tool-choice --tool-call-parser hermes \
#       --tool-server http://127.0.0.1:8001/sse \
#       --api-key "${SPARKS_API_KEY}"
#   Terminal 3:  SPARKS_API_KEY="<same key>" python cli.py
