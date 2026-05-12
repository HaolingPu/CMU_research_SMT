#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPTS_FILE="${SCRIPT_DIR}/ckpts.txt"
CKPT_ROOT=/data/user_data/siqiouya/ckpts/infinisst-omni

AUDIO_DEFINITION=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.yaml
TRANSCRIPT_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt
REFERENCE_FILE=/data/group_data/li_lab/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt

MOSES_TOKENIZER=zh
SACREBLEU_TOKENIZER=zh
CHAR_LEVEL_FLAG="--char_level"

SEGS=(960 1920 2880 3840)

mapfile -t CKPTS < <(grep -v '^[[:space:]]*$' "${CKPTS_FILE}" | sed 's:/*$::')

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

        # if [ -s "${avg_scores}" ] && [ -s "${segment_scores}" ]; then
        #     echo "[SKIP] already scored (avg + segment): ${ckpt} seg${seg}"
        #     continue
        # fi

        echo "[RUN] ${ckpt} seg${seg}"
        norm_instances="${out_dir}/instances.normalized.log"
        python "${SCRIPT_DIR}/normalize_instances.py" "${instances}" "${norm_instances}"
        export GEMINI_API_KEY=$(cat /home/siqiouya/.keys/gemini_personal)
        export OPENAI_API_KEY=$(cat /home/siqiouya/.keys/openai_personal)
        omnisteval longform \
            --speech_segmentation "${AUDIO_DEFINITION}" \
            --source_sentences_file "${TRANSCRIPT_FILE}" \
            --ref_sentences_file "${REFERENCE_FILE}" \
            --hypothesis_file "${norm_instances}" \
            --hypothesis_format jsonl \
            --comet \
            --comet_model Unbabel/XCOMET-XL \
            --bleurt \
            --bleurt_model lucadiliello/BLEURT-20 \
            --source_lang "English" \
            --target_lang "Simplified Chinese" \
            --lang "${MOSES_TOKENIZER}" \
            ${CHAR_LEVEL_FLAG} \
            --bleu_tokenizer "${SACREBLEU_TOKENIZER}" \
            --output_folder "${seg_out}" \
            --taser \
            --taser_model o3 \
            --taser_concurrency 8 \
            --taser_reasoning_effort low
            
        echo "[DONE] ${ckpt} seg${seg}"
    done
done
