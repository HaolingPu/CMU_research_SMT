#!/usr/bin/env bash

# Run simuleval + streamLAAL + omnisteval on the future-aware test set,
# sweeping over both speeds (1.0 and 0.7) and four segment sizes (960ms..3840ms).
#
# Array layout (1-8):
#   tasks 1-4 -> speed 1.0,  seg = 960 * task_id
#   tasks 5-8 -> speed 0.7,  seg = 960 * (task_id - 4)
#
# Usage:
#   sbatch scripts/infer/infer_future_aware_slurm.sh <model_path> <prompt_type>

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
##SBATCH --requeue
#SBATCH --exclude=babel-p9-32,babel-o5-24,babel-o5-28
#SBATCH --time=1-00:00:00
#SBATCH --array=1-8
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH -o slurm_logs/%A_%a.out


# sbatch scripts/infer/infer_future_aware_slurm.sh /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-hibiki-s-bsz4/v0-20260326-141050-hf Standard
# sbatch scripts/infer/infer_future_aware_slurm.sh /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf Standard
# sbatch scripts/infer/infer_future_aware_slurm.sh /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-consensus-topk5_f200-s-bsz4/v0-20260502-125501-hf/ Standard

find /dev/shm -maxdepth 1 -name "vllm*" -user "$USER" -mmin +30 \
  -exec rm -rf {} + 2>/dev/null || true

source /home/siqiouya/miniconda3/bin/activate omni_inference

MODEL_PATH=$1
PROMPT_TYPE=$2

DATA_ROOT=/data/group_data/li_lab/siqiouya/datasets/future_aware_test

# Map array task -> (speed, seg_index)
SPEEDS=("1.0" "0.7")
SPEED_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
SEG_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 + 1 ))
SPEED=${SPEEDS[$SPEED_IDX]}
SOURCE_SEGMENT_SIZE=$((960 * SEG_IDX))

SOURCE_LIST=${DATA_ROOT}/source_spd_${SPEED}.txt
TARGET_LIST=${DATA_ROOT}/target.txt
SOURCE_TEXT=${DATA_ROOT}/source_text.txt
AUDIO_YAML=${DATA_ROOT}/data_spd_${SPEED}.yaml

OUTPUT_PATH=${MODEL_PATH}/evaluation/future_aware_test/en-zh/spd_${SPEED}_seg${SOURCE_SEGMENT_SIZE}
if [ "$PROMPT_TYPE" == "EAST" ]; then
    OUTPUT_PATH=${OUTPUT_PATH}_low
fi

echo "task=${SLURM_ARRAY_TASK_ID} speed=${SPEED} seg=${SOURCE_SEGMENT_SIZE} out=${OUTPUT_PATH}"

MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    uv run simuleval \
        --agent /home/siqiouya/code/CMU_research_SMT/scripts/infer/infinisst_omni.py \
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
        --source ${SOURCE_LIST} \
        --target ${TARGET_LIST} \
        --use-vllm 1 \
        --temperature 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --model-name ${MODEL_PATH} \
        --quality-metrics BLEU \
        --eval-latency-unit char \
        --sacrebleu-tokenizer zh
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ] && [ ! -s "${OUTPUT_PATH}/instances.log" ]; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "simuleval exited abnormally (exit code: $EXIT_CODE) and instances.log is empty. Retry $RETRY_COUNT/$MAX_RETRIES..."
        rm -rf "${OUTPUT_PATH}"
    else
        break
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "simuleval failed after $MAX_RETRIES retries. Exiting."
    exit 1
fi

export MWERSEGMENTER_ROOT=/home/siqiouya/download/mwerSegmenter
export PYTHONPATH=/home/siqiouya/code/FBK-fairseq
conda activate fbk

streamLAAL \
    --simuleval-instances ${OUTPUT_PATH}/instances.log \
    --source ${SOURCE_TEXT} \
    --reference ${TARGET_LIST} \
    --audio-yaml ${AUDIO_YAML} \
    --sacrebleu-tokenizer zh \
    --latency-unit char \
    > ${OUTPUT_PATH}/streamLAAL.txt 2>&1

# -------- omnisteval (COMET, BLEURT, TASER) --------
conda activate evaluation

SEG_OUT=${OUTPUT_PATH}/segmentation_output
NORM_INSTANCES=${OUTPUT_PATH}/instances.normalized.log

python /home/siqiouya/code/CMU_research_SMT/scripts/infer/normalize_instances.py \
    ${OUTPUT_PATH}/instances.log ${NORM_INSTANCES}

export GEMINI_API_KEY=$(cat /home/siqiouya/.keys/gemini_personal)
export OPENAI_API_KEY=$(cat /home/siqiouya/.keys/openai_personal)

omnisteval longform \
    --speech_segmentation ${AUDIO_YAML} \
    --source_sentences_file ${SOURCE_TEXT} \
    --ref_sentences_file ${TARGET_LIST} \
    --hypothesis_file ${NORM_INSTANCES} \
    --hypothesis_format jsonl \
    --comet \
    --comet_model Unbabel/XCOMET-XL \
    --bleurt \
    --bleurt_model lucadiliello/BLEURT-20 \
    --source_lang "English" \
    --target_lang "Simplified Chinese" \
    --lang zh \
    --char_level \
    --bleu_tokenizer zh \
    --output_folder ${SEG_OUT} \
    --taser \
    --taser_model o3 \
    --taser_concurrency 8 \
    --taser_reasoning_effort low