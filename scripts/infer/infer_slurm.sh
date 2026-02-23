#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
#SBATCH --exclude=babel-p9-32
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=0
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

# sbatch infer_slurm.sh /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-Simul-MuST-C-fixed-s_origin-bsz4/v0-20260223-121324-hf Standard

source /home/siqiouya/miniconda3/bin/activate omni_inference

MODEL_PATH=$1
PROMPT_TYPE=$2

OUTPUT_PATH=${MODEL_PATH}/evaluation/acl_6060/en-zh/seg960
if [ "$PROMPT_TYPE" == "EAST" ]; then
    OUTPUT_PATH=${OUTPUT_PATH}_low
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
uv run simuleval \
    --agent /home/siqiouya/code/CMU_research_SMT/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size 960 \
    --prompt-type ${PROMPT_TYPE} \
    --EAST-latency-type low \
    --output ${OUTPUT_PATH} \
    --max-new-tokens 30 \
    --max-cache-chunks 60 \
    --keep-cache-chunks 30 \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec 2 \
    --source /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.source \
    --target /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.target.zh \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --model-name ${MODEL_PATH} \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh

export MWERSEGMENTER_ROOT=/home/siqiouya/download/mwerSegmenter
export PYTHONPATH=/home/siqiouya/code/FBK-fairseq
conda activate fbk

streamLAAL \
    --simuleval-instances ${OUTPUT_PATH}/instances.log  \
    --source /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
    --reference /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt \
    --audio-yaml /data/user_data/siqiouya/datasets/acl_6060/dev.yaml \
    --sacrebleu-tokenizer zh \
    --latency-unit char