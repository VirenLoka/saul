#!/usr/bin/env bash
# =============================================================================
# Serve Qwen/Qwen2.5-7B-Instruct on a single NVIDIA H200 (140 GB VRAM) with
# vLLM, exposing an OpenAI-compatible endpoint with NATIVE TOOL CALLING.
#
# Prereqs (on the serving host only — NOT needed by the Python agent):
#   pip install "vllm>=0.6.0"
#
# Security defaults:
#   - Binds to 127.0.0.1 only (not 0.0.0.0) to prevent external exposure.
#   - Requires a bearer token via SPARKS_API_KEY. Generate one with:
#       export SPARKS_API_KEY="$(openssl rand -hex 32)"
#     Agent client must pass: Authorization: Bearer <key>
#     The Python agent reads this from the SPARKS_API_KEY env var
#     (preferred) or local_inference_settings.api_key in config.yaml.
#
# Configuration:
#   All variables can be overridden by exporting them before running this
#   script. They mirror the names in config.yaml local_inference_settings:
#     SPARKS_HOST          -> config: host          (default 127.0.0.1)
#     SPARKS_PORT          -> config: port          (default 8000)
#     SPARKS_API_KEY       -> config: api_key       (required)
#     SPARKS_MODEL         -> model_selection.local_model
#     SPARKS_MAX_MODEL_LEN -> context window        (default 32768)
#     SPARKS_GPU_UTIL      -> GPU memory fraction   (default 0.92)
#     SPARKS_TOOL_PARSER   -> tool-call parser      (default hermes)
# =============================================================================
set -euo pipefail

MODEL="${SPARKS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # HF repo id or local path
PORT="${SPARKS_PORT:-8000}"
HOST="${SPARKS_HOST:-127.0.0.1}"                    # loopback only by default
MAX_MODEL_LEN="${SPARKS_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${SPARKS_GPU_UTIL:-0.92}"                 # ~129 GB of the H200's 140 GB
TOOL_PARSER="${SPARKS_TOOL_PARSER:-hermes}"         # Qwen2.5-family tool-call parser

# SPARKS_API_KEY must be set — refuse to start without it.
if [[ -z "${SPARKS_API_KEY:-}" ]]; then
  echo "ERROR: SPARKS_API_KEY is not set." >&2
  echo "Generate one with: export SPARKS_API_KEY=\"\$(openssl rand -hex 32)\"" >&2
  exit 1
fi

echo "Starting vLLM server:"
echo "  model   : ${MODEL}"
echo "  endpoint: http://${HOST}:${PORT}"
echo "  ctx len : ${MAX_MODEL_LEN} tokens"
echo "  GPU util: ${GPU_UTIL}"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${MODEL}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER}" \
  --api-key "${SPARKS_API_KEY}"

# --- One-liner equivalent (copy/paste) --------------------------------------
# SPARKS_API_KEY="$(openssl rand -hex 32)" \
# vllm serve Qwen/Qwen2.5-7B-Instruct \
#   --host 127.0.0.1 --port 8000 \
#   --gpu-memory-utilization 0.92 --max-model-len 32768 \
#   --enable-auto-tool-choice --tool-call-parser hermes \
#   --api-key "${SPARKS_API_KEY}"
