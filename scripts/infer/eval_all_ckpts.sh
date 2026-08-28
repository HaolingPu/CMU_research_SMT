#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate evaluation

# HuggingFace auth for gated models like Unbabel/XCOMET-XL
if [ -f "/home/haolingp/.keys/huggingface" ]; then
    export HF_TOKEN=$(cat /home/haolingp/.keys/huggingface)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPTS_FILE="${CKPTS_FILE:-${SCRIPT_DIR}/ckpts.txt}"
CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni

AUDIO_DEFINITION=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.yaml
TRANSCRIPT_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt
REFERENCE_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt

MOSES_TOKENIZER=zh
SACREBLEU_TOKENIZER=zh
CHAR_LEVEL_FLAG="--char_level"

SEGS=(960 1920 2880 3840)

mapfile -t CKPTS < <(grep -vE '^[[:space:]]*(#|$)' "${CKPTS_FILE}" | sed 's:/*$::')

echo "Loaded ${#CKPTS[@]} checkpoints; will score ${#SEGS[@]} latencies each."

for ckpt in "${CKPTS[@]}"; do
    for seg in "${SEGS[@]}"; do
        out_dir="${CKPT_ROOT}/${ckpt}/evaluation/acl_6060/en-zh/seg${seg}"
        instances="${out_dir}/instances.log"
        seg_out="${out_dir}/segmentation_output"
        resegmented="${seg_out}/instances.resegmented.jsonl"
        segment_scores="${seg_out}/segment_scores.tsv"
        avg_scores="${seg_out}/scores.tsv"

        if [ ! -s "${instances}" ]; then
            echo "[SKIP] no instances.log: ${ckpt} seg${seg}"
            continue
        fi

        if [ -s "${avg_scores}" ]; then
            echo "[SKIP] already scored: ${ckpt} seg${seg}"
            continue
        fi

        echo "[RUN] ${ckpt} seg${seg}"
        norm_instances="${out_dir}/instances.normalized.log"
        python "${SCRIPT_DIR}/normalize_instances.py" "${instances}" "${norm_instances}"
        omnisteval longform \
            --speech_segmentation "${AUDIO_DEFINITION}" \
            --source_sentences_file "${TRANSCRIPT_FILE}" \
            --ref_sentences_file "${REFERENCE_FILE}" \
            --hypothesis_file "${norm_instances}" \
            --hypothesis_format jsonl \
            --comet \
            --comet_model Unbabel/XCOMET-XL \
            --lang "${MOSES_TOKENIZER}" \
            ${CHAR_LEVEL_FLAG} \
            --bleu_tokenizer "${SACREBLEU_TOKENIZER}" \
            --output_folder "${seg_out}"

        echo "[DONE] ${ckpt} seg${seg}"
    done
done
echo
echo "All scoring done. Per-seg scores are at <ckpt>/evaluation/acl_6060/en-zh/seg<N>/segmentation_output/scores.tsv"
# python "${SCRIPT_DIR}/aggregate_scores.py"  # not present; aggregate manually
