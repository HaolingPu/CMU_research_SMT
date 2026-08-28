#!/usr/bin/env bash
# Variant of infer_slurm.sh for LA hallucination probe:
#   - greedy decoding (--temperature 0.0)
#   - adaptive max-new-tokens cap: mnt = max(30, seg_ms / 32)
#       seg960  -> 30
#       seg1920 -> 60
#       seg2880 -> 90
#       seg3840 -> 120
# Output goes to seg${N} (no suffix) so existing eval_all_ckpts.sh picks it up.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
#SBATCH --exclude=babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-n5-32,babel-o5-16,babel-n5-28
#SBATCH --time=1-00:00:00
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH -o slurm_logs/%A_%a.out

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation

MODEL_PATH=$1
PROMPT_TYPE=$2
SOURCE_SEGMENT_SIZE=$((960 * $SLURM_ARRAY_TASK_ID))

MNT=$(( SOURCE_SEGMENT_SIZE / 32 ))
if [ $MNT -lt 30 ]; then MNT=30; fi

OUTPUT_PATH=${MODEL_PATH}/evaluation/acl_6060/en-zh/seg${SOURCE_SEGMENT_SIZE}
if [ "$PROMPT_TYPE" == "EAST" ]; then
    OUTPUT_PATH=${OUTPUT_PATH}_low
fi

echo "seg=${SOURCE_SEGMENT_SIZE} mnt=${MNT} temperature=0 output=${OUTPUT_PATH}"

MAX_RETRIES=3
RETRY_COUNT=0
ATTEMPT_TIMEOUT=5400

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    timeout --kill-after=30s --signal=TERM ${ATTEMPT_TIMEOUT}s \
    env VLLM_WORKER_MULTIPROC_METHOD=spawn \
        NCCL_P2P_DISABLE=1 \
        NCCL_IB_DISABLE=1 \
    uv run simuleval \
        --agent /home/haolingp/CMU_research_SMT/scripts/infer/infinisst_omni.py \
        --agent-class agents.InfiniSSTOmni \
        --source-segment-size ${SOURCE_SEGMENT_SIZE} \
        --prompt-type ${PROMPT_TYPE} \
        --EAST-latency-type low \
        --output ${OUTPUT_PATH} \
        --max-new-tokens ${MNT} \
        --max-cache-chunks 60 \
        --keep-cache-chunks 30 \
        --source-lang English \
        --target-lang Chinese \
        --min-start-sec 2 \
        --source /data/user_data/haolingp/datasets/acl_6060/dev.source \
        --target /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.target.zh \
        --use-vllm 1 \
        --temperature 0.0 \
        --top-p 1.0 \
        --top-k -1 \
        --model-name ${MODEL_PATH} \
        --quality-metrics BLEU \
        --eval-latency-unit char \
        --sacrebleu-tokenizer zh
    EXIT_CODE=$?

    if { [ $EXIT_CODE -ne 0 ] && [ ! -s "${OUTPUT_PATH}/instances.log" ]; } \
        || [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "exit=$EXIT_CODE empty/hung; retry $RETRY_COUNT/$MAX_RETRIES"
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
