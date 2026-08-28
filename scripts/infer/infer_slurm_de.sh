#!/usr/bin/env bash
# en->de version of infer_slurm.sh
# Chunk-level simuleval (BLEU + AL) on ACL 6060 dev with the de reference.

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
##SBATCH --requeue
#SBATCH --exclude=babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-n5-32,babel-o5-16,babel-o5-32,babel-n5-28,babel-s5-24,babel-o5-28,babel-p5-24,babel-p5-20,babel-p5-28
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH -o slurm_logs/%A_%a.out

# Usage: sbatch infer_slurm_de.sh <MODEL_PATH> <PROMPT_TYPE>
#   PROMPT_TYPE: Standard | EAST

# Fast-fail if the home NFS mount is missing on this node (some babel
# nodes occasionally come up without /home/haolingp visible; without this
# guard the retry loop just burns 3 attempts at exit-127 from missing
# conda/uv).
if [ ! -f /home/haolingp/miniconda3/etc/profile.d/conda.sh ]; then
    echo "[FATAL] /home/haolingp/miniconda3 not visible on $(hostname) — likely NFS not mounted. Aborting." >&2
    exit 1
fi

source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate evaluation

MODEL_PATH=$1
PROMPT_TYPE=$2
SOURCE_SEGMENT_SIZE=$((960 * $SLURM_ARRAY_TASK_ID))

OUTPUT_PATH=${MODEL_PATH}/evaluation/acl_6060/en-de/seg${SOURCE_SEGMENT_SIZE}
if [ "$PROMPT_TYPE" == "EAST" ]; then
    OUTPUT_PATH=${OUTPUT_PATH}_low
fi

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
        --max-new-tokens 60 \
        --max-cache-chunks 60 \
        --keep-cache-chunks 30 \
        --source-lang English \
        --target-lang German \
        --min-start-sec 2 \
        --source /data/user_data/haolingp/datasets/acl_6060/dev.source \
        --target /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.target.de \
        --use-vllm 1 \
        --temperature 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --model-name ${MODEL_PATH} \
        --quality-metrics BLEU \
        --eval-latency-unit word \
        --sacrebleu-tokenizer 13a
    EXIT_CODE=$?

    if { [ $EXIT_CODE -ne 0 ] && [ ! -s "${OUTPUT_PATH}/instances.log" ]; } \
        || [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
            echo "simuleval hung past ${ATTEMPT_TIMEOUT}s (likely vllm shm_broadcast deadlock). Retry $RETRY_COUNT/$MAX_RETRIES..."
        else
            echo "simuleval exited abnormally (exit code: $EXIT_CODE) and instances.log is empty. Retry $RETRY_COUNT/$MAX_RETRIES..."
        fi
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
