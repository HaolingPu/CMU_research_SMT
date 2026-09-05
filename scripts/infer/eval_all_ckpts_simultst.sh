#!/usr/bin/env bash
# Simul-tst-COMMON (monotonic references) variant of eval_all_ckpts.sh.
# Scores instances.log produced by infer_slurm_simultst.sh.
eval "$(conda shell.bash hook)"
conda activate evaluation

if [ -f "/home/haolingp/.keys/huggingface" ]; then
    export HF_TOKEN=$(cat /home/haolingp/.keys/huggingface)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPTS_FILE="${CKPTS_FILE:-${SCRIPT_DIR}/ckpts_simultst.txt}"
CKPT_ROOT=/data/user_data/haolingp/ckpts/infinisst-omni

AUDIO_DEFINITION=/data/user_data/haolingp/datasets/simul_tst_common/tst.yaml
TRANSCRIPT_FILE=/data/user_data/haolingp/datasets/simul_tst_common/tst.en
REFERENCE_FILE=/data/user_data/haolingp/datasets/simul_tst_common/tst.zh

MOSES_TOKENIZER=zh
SACREBLEU_TOKENIZER=zh
CHAR_LEVEL_FLAG="--char_level"

SEGS=(960 1920 2880 3840)

mapfile -t CKPTS < <(grep -vE '^[[:space:]]*(#|$)' "${CKPTS_FILE}" | sed 's:/*$::')

echo "Loaded ${#CKPTS[@]} checkpoints; will score ${#SEGS[@]} latencies each."

for ckpt in "${CKPTS[@]}"; do
    for seg in "${SEGS[@]}"; do
        out_dir="${CKPT_ROOT}/${ckpt}/evaluation/simul_tst_common/en-zh/seg${seg}"
        instances="${out_dir}/instances.log"
        seg_out="${out_dir}/segmentation_output"
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
echo "All scoring done. Per-seg scores at <ckpt>/evaluation/simul_tst_common/en-zh/seg<N>/segmentation_output/scores.tsv"
