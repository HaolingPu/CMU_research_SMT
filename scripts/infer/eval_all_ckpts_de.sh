#!/usr/bin/env bash
set -e

if ! command -v conda >/dev/null; then
    echo "[FATAL] conda not found on $(hostname); home NFS likely missing." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate evaluation
command -v omnisteval >/dev/null || { echo "[FATAL] omnisteval missing in env" >&2; exit 1; }

# HuggingFace auth for gated models like Unbabel/XCOMET-XL
if [ -f "/home/haolingp/.keys/huggingface" ]; then
    export HF_TOKEN=$(cat /home/haolingp/.keys/huggingface)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPTS_FILE="${SCRIPT_DIR}/ckpts_de.txt"
CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni

AUDIO_DEFINITION=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.yaml
TRANSCRIPT_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt
REFERENCE_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.de.txt

MOSES_TOKENIZER=de
SACREBLEU_TOKENIZER=13a
CHAR_LEVEL_FLAG="--word_level"   # de: word-level (omnisteval requires explicit --word_level OR --char_level)

SEGS=(960 1920 2880 3840)

mapfile -t CKPTS < <(grep -vE '^[[:space:]]*(#|$)' "${CKPTS_FILE}" | sed 's:/*$::')

echo "Loaded ${#CKPTS[@]} checkpoints (de); will score ${#SEGS[@]} latencies each."

for ckpt in "${CKPTS[@]}"; do
    for seg in "${SEGS[@]}"; do
        out_dir="${CKPT_ROOT}/${ckpt}/evaluation/acl_6060/en-de/seg${seg}"
        instances="${out_dir}/instances.log"
        seg_out="${out_dir}/segmentation_output"
        segment_scores="${seg_out}/segment_scores.tsv"
        avg_scores="${seg_out}/scores.tsv"

        if [ ! -s "${instances}" ]; then
            echo "[SKIP] no instances.log: ${ckpt} seg${seg}"
            continue
        fi

        if [ -s "${avg_scores}" ] && [ -s "${segment_scores}" ]; then
            echo "[SKIP] already scored (avg + segment): ${ckpt} seg${seg}"
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
echo "All scoring done. Per-seg scores at <ckpt>/evaluation/acl_6060/en-de/seg<N>/segmentation_output/scores.tsv"
