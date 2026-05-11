#!/usr/bin/env bash
# Run all three Qwen3-8B taboo ELK baselines in closed (1-of-20) mode.
# SAE and transcoder run for layers 9, 18, 27 (matching the open-mode sweep).
# Stops on the first failure.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYERS=(${LAYERS:-9 27})
LOGDIR="${LOGDIR:-$HERE/logs_closed}"
mkdir -p "$LOGDIR"

# echo "=== AO (closed) ==="
# python "$HERE/elk_using_qwen3_8b_ao.py" --mode closed \
#   2>&1 | tee "$LOGDIR/ao_closed.log"

for L in "${LAYERS[@]}"; do
  echo "=== SAE layer=$L (closed, tfidf) ==="
  python "$HERE/elk_using_qwen3_8b_sae.py" --layer "$L" --use_tfidf --mode closed \
    2>&1 | tee "$LOGDIR/sae_layer${L}_closed.log"
done

for L in "${LAYERS[@]}"; do
  echo "=== Transcoder layer=$L (closed, tfidf) ==="
  python "$HERE/elk_using_qwen3_8b_transcoder.py" --layer "$L" --use_tfidf --mode closed \
    2>&1 | tee "$LOGDIR/transcoder_layer${L}_closed.log"
done

echo "=== Done. Results: ==="
ls -1 "$HERE"/*_closed.json
