#!/usr/bin/env bash
#SBATCH --job-name=eval_topk5_v2_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --exclude=babel-t9-16,babel-o5-16
#SBATCH -o /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/eval_topk5_v2_v2_%j.out
#SBATCH -e /home/haolingp/CMU_research_SMT/scripts/infer/slurm_logs/eval_topk5_v2_v2_%j.err

set -e
eval "$(conda shell.bash hook)"
conda activate evaluation

export HF_TOKEN=$(cat /home/haolingp/.keys/huggingface)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

CKPT=gigaspeech-zh-consensus-topk5_v2-s-bsz4/v2-20260426-101916-hf
CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni
AUDIO_DEFINITION=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.yaml
TRANSCRIPT_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt
REFERENCE_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt
SCRIPT_DIR=/home/haolingp/CMU_research_SMT/scripts/infer

for seg in 960 1920 2880 3840; do
    out_dir="${CKPT_ROOT}/${CKPT}/evaluation/acl_6060/en-zh/seg${seg}"
    instances="${out_dir}/instances.log"
    norm="${out_dir}/instances.normalized.log"
    seg_out="${out_dir}/segmentation_output"

    if [ ! -s "${instances}" ]; then
        echo "skip seg${seg}: no instances.log"
        continue
    fi

    echo "=== seg${seg} ==="
    python "${SCRIPT_DIR}/normalize_instances.py" "${instances}" "${norm}"
    omnisteval longform \
        --speech_segmentation "${AUDIO_DEFINITION}" \
        --source_sentences_file "${TRANSCRIPT_FILE}" \
        --ref_sentences_file "${REFERENCE_FILE}" \
        --hypothesis_file "${norm}" \
        --hypothesis_format jsonl \
        --comet --comet_model Unbabel/XCOMET-XL \
        --lang zh --char_level --bleu_tokenizer zh \
        --output_folder "${seg_out}"
    echo "=== seg${seg} done ==="
done
echo "ALL DONE"
