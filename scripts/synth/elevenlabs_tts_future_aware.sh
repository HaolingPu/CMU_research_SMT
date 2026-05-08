#!/usr/bin/env bash
# ============================================================
# ElevenLabs TTS for the future-aware test set.
# Synthesizes wavs/<id>.wav and writes source.txt/target.txt manifests.
#
# Set ELEVENLABS_API_KEY in the environment before invoking.
# Override paths/voice/model/speed via env vars:
#   INPUT_TSV=/path/to.tsv OUTPUT_DIR=/path SPEED=0.9 \
#     bash scripts/synth/elevenlabs_tts_future_aware.sh
# Extra CLI flags are passed through to the Python script.
# ============================================================

# SPEED=1.0 MODEL_ID=eleven_multilingual_v2 bash elevenlabs_tts_future_aware.sh
# SPEED=0.7 MODEL_ID=eleven_multilingual_v2 bash elevenlabs_tts_future_aware.sh

set -euo pipefail

source /home/siqiouya/miniconda3/etc/profile.d/conda.sh
conda activate consensus

export ELEVENLABS_API_KEY=$(cat /home/siqiouya/.keys/elevenlabs_sst_data)

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="${REPO_DIR}/data_synthesis/codes-refactored/test_data/elevenlabs_tts_future_aware.py"

INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/future_aware_test/future_aware_testset_v2.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/group_data/li_lab/siqiouya/datasets/future_aware_test}"
VOICE_ID="${VOICE_ID:-BIvP0GN1cAtSRTxNHnWS}" # Ellen - Serious, Direct and Confident
MODEL_ID="${MODEL_ID:-eleven_v3}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-wav_16000}"
SPEED="${SPEED:-1.0}"

exec python "${SCRIPT}" \
  --input-tsv "${INPUT_TSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --voice-id "${VOICE_ID}" \
  --model-id "${MODEL_ID}" \
  --output-format "${OUTPUT_FORMAT}" \
  --speed "${SPEED}" \
  "$@"
