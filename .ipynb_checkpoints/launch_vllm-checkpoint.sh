#!/usr/bin/env bash
# =============================================================================
# Serve a Qwen model on a single NVIDIA H200 (140 GB VRAM) with vLLM, exposing
# an OpenAI-compatible endpoint with NATIVE TOOL CALLING.
#
# Prereqs on the serving host (NOT needed by the agent client):
#   pip install "vllm>=0.6.0"
#
# Security defaults:
#   - Binds to 127.0.0.1 only (not 0.0.0.0) to prevent external exposure.
#   - Requires a bearer token via SPARKS_API_KEY (generate one with:
#       export SPARKS_API_KEY="$(openssl rand -hex 32)"
#     Clients must pass: Authorization: Bearer <key>
#
# Adjust MODEL / TOOL_PARSER / MAX_MODEL_LEN to match your exact build.
# =============================================================================
set -euo pipefail

MODEL="${SPARKS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # HF repo id or local path
PORT="${SPARKS_PORT:-8000}"
HOST="${SPARKS_HOST:-127.0.0.1}"                    # Loopback only by default
MAX_MODEL_LEN="${SPARKS_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${SPARKS_GPU_UTIL:-0.92}"                 # ~129 GB of the H200's 140 GB
TOOL_PARSER="${SPARKS_TOOL_PARSER:-hermes}"         # Qwen3-family tool-call parser

# SPARKS_API_KEY must be set — refuse to start without it.
if [[ -z "${SPARKS_API_KEY:-}" ]]; then
  echo "ERROR: SPARKS_API_KEY is not set." >&2
  echo "Generate one with: export SPARKS_API_KEY=\"\$(openssl rand -hex 32)\"" >&2
  exit 1a
fi

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
# vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
#   --gpu-memory-utilization 0.92 --max-model-len 32768 \
#   --enable-auto-tool-choice --tool-call-parser hermes \
#   --api-key "${SPARKS_API_KEY}"