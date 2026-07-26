#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
#SBATCH --exclude=babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-n5-32,babel-o5-16,babel-n5-28,babel-q5-32,babel-s5-24,babel-q5-24
#SBATCH --time=1-00:00:00
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH -o slurm_logs/%A_%a.out

# Simul-tst-COMMON (monotonic references) variant of infer_slurm.sh.
# Usage: sbatch infer_slurm_simultst.sh <MODEL_PATH> <PROMPT_TYPE>
#   PROMPT_TYPE: Standard | EAST
# 27 TED wavs (~5.5h audio, ~5x ACL dev) -> longer attempt timeout.

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation

MODEL_PATH=$1
PROMPT_TYPE=$2
SOURCE_SEGMENT_SIZE=$((960 * $SLURM_ARRAY_TASK_ID))

OUTPUT_PATH=${MODEL_PATH}/evaluation/simul_tst_common/en-zh/seg${SOURCE_SEGMENT_SIZE}
if [ "$PROMPT_TYPE" == "EAST" ]; then
    OUTPUT_PATH=${OUTPUT_PATH}_low
fi

MAX_RETRIES=3
RETRY_COUNT=0
ATTEMPT_TIMEOUT=18000

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    timeout --kill-after=30s --signal=TERM ${ATTEMPT_TIMEOUT}s \
    env VLLM_WORKER_MULTIPROC_METHOD=spawn \
        NCCL_P2P_DISABLE=1 \
        NCCL_IB_DISABLE=1 \
    uv run simuleval \
        --agent /data/user_data/haolingp/scripts/infer/infinisst_omni.py \
        --agent-class agents.InfiniSSTOmni \
        --source-segment-size ${SOURCE_SEGMENT_SIZE} \
        --prompt-type ${PROMPT_TYPE} \
        --EAST-latency-type low \
        --output ${OUTPUT_PATH} \
        --max-new-tokens 30 \
        --max-cache-chunks 60 \
        --keep-cache-chunks 30 \
        --source-lang English \
        --target-lang Chinese \
        --min-start-sec 2 \
        --source /data/user_data/haolingp/datasets/simul_tst_common/tst.source \
        --target /data/user_data/haolingp/datasets/simul_tst_common/tst.target.zh \
        --use-vllm 1 \
        --temperature 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --model-name ${MODEL_PATH} \
        --quality-metrics BLEU \
        --eval-latency-unit char \
        --sacrebleu-tokenizer zh
    EXIT_CODE=$?

    if { [ $EXIT_CODE -ne 0 ] && [ ! -s "${OUTPUT_PATH}/instances.log" ]; } \
        || [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "simuleval failed/hung (exit $EXIT_CODE). Retry $RETRY_COUNT/$MAX_RETRIES..."
        rm -rf "${OUTPUT_PATH}"
        pkill -9 -P $$ 2>/dev/null || true
        sleep 5
    else
        break
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "simuleval failed after $MAX_RETRIES retries. Exiting."
    exit 1
fi
