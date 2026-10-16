#!/usr/bin/env bash
# Run this ON the Lambda instance, not on the laptop.
#
# Creates a venv, installs a pinned vLLM, and serves one open-weight judge model with an
# OpenAI-compatible server on 127.0.0.1 only (reach it through an ssh tunnel; never open the port).
# The client (run_judge.py) fixes temperature 0 and the seed on every request.
#
# Usage:
#   bash serve_vllm.sh                      # first judge, default Qwen/Qwen3.5-4B
#   bash serve_vllm.sh Qwen/Qwen3-32B       # second, larger judge
#   PORT=8000 MAX_MODEL_LEN=4096 bash serve_vllm.sh <model>
#
# UNVERIFIED, check at launch and write the result in docs/build-log.md:
#   - the model names (Qwen/Qwen3.5-4B, Qwen/Qwen3-32B) exist on Hugging Face under these ids,
#     and their licences are still Apache-2.0;
#   - VLLM_VERSION below is a real release that supports the chosen model (Qwen3.5 needs a recent
#     vLLM); change the pin rather than installing an unpinned latest;
#   - the GPU has enough memory: a 4B model fits a 24 GB A10; a 32B model in bf16 needs about
#     64 GB of weights alone, so it needs an 80 GB A100 or H100 (or a quantized checkpoint).
#
# An HF token is not needed for these models. If one is ever needed it lives only in the HF_TOKEN
# env var on the instance, never in a file in either repo.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-4B}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"   # UNVERIFIED pin, see above
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VENV="${VENV:-$HOME/ati-judge-venv}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "vllm==${VLLM_VERSION}"
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

echo "serving ${MODEL} on 127.0.0.1:${PORT} (vllm ${VLLM_VERSION})"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --seed 0
