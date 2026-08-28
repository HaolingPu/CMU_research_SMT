#!/usr/bin/env bash
# ============================================================
# FULL canonical pipeline for window=3 on the SPLIT-FIXED qwenasr sentsplit data.
# Identical to run_win3_FULL_chain.sh, but decode root / tag point at the corrected
# sub-sentence split (split_src_text_full_spacy min_words=2 fragment merge).
# Clean A/B vs FULL40k-win3: both period-fixed + same win3+QE pipeline, only the
# sub-sentence split differs -> isolates the sentsplit fix.
#
#   decode(40k, FIX TSV)  ->  period-fix  ->  SEGALE QE-MAX<=3  ->  length filter
#     ->  conv2swift (sample 12.5k)  ->  train  ->  infer(4 seg) + eval
#
# Usage:  bash run_FIX_win3_FULL_chain.sh <DECODE_ARRAY_JOBID>
# ============================================================
set -e

DECODE_JID="${1:-$(cat /tmp/punct_win3_full40k_decode_jid.txt)}"
FS=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling
SEG=${FS}/scripts/segale

DECODE_ROOT=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k_qwenasr_punctsplit_win3_FULL40k
FIXED_ROOT=${DECODE_ROOT}-pfix
OUT_ROOT=${FIXED_ROOT}-segale-p24
QE_FILTERED_DIR=${OUT_ROOT}/qe3-aligned-max
LEN_FILTERED_DIR=${OUT_ROOT}/qe3-aligned-max-len
VARIANT_TAG=FULL40k-win3-punctsplit
EXP=gigaspeech-zh-consensus-${VARIANT_TAG}-s-bsz4

echo "decode root job : ${DECODE_JID}"
echo "DECODE_ROOT     : ${DECODE_ROOT}"
echo "EXP             : ${EXP}"

# 1) period-fix (afterok the whole decode array)
PFIX=$(sbatch --parsable --dependency=afterok:${DECODE_JID} \
  --export=ALL,IN_ROOT=${DECODE_ROOT},OUT_ROOT=${FIXED_ROOT} \
  ${FS}/run_period_fix_nested.sbatch)
echo "period-fix      : ${PFIX}"

# 2) SEGALE align + per-sentence QE-MAX<=3
POST_OUT=$(DEPEND_ON_JOBS=${PFIX} bash ${SEG}/submit_J40k_post.sh \
  ${FIXED_ROOT} ${OUT_ROOT} 40000 3.0)
echo "${POST_OUT}"
QE_FIN_JID=$(echo "${POST_OUT}" | grep -E "^\[qe finalize\]" | grep -oE "[0-9]+" | head -1)
if [[ -z "${QE_FIN_JID}" ]]; then echo "ERROR: could not parse QE finalize job id"; exit 1; fi
echo "qe-finalize job : ${QE_FIN_JID}"

# 3) length-ratio filter (afterok QE finalize)
LEN=$(sbatch --parsable --dependency=afterok:${QE_FIN_JID} \
  --export=ALL,QE_FILTERED_DIR=${QE_FILTERED_DIR},LEN_FILTERED_DIR=${LEN_FILTERED_DIR},MAX_RATIO_REF=1.5,MAX_RATIO_SRC=2.5 \
  ${FS}/run_length_filter.sbatch)
echo "length-filter   : ${LEN}"

# 4) conv2swift — sample 12.5k
CONV=$(sbatch --parsable --dependency=afterok:${LEN} \
  --export=ALL,MANIFEST_ROOT=${LEN_FILTERED_DIR},VARIANT_TAG=${VARIANT_TAG},SAMPLE_N=12500 \
  /home/haolingp/CMU_research_SMT/scripts/train/run_convert2swift_consensus.sbatch)
echo "conv2swift      : ${CONV}"

# 5) train
TRAIN=$(sbatch --parsable --dependency=afterok:${CONV} \
  --export=ALL,VARIANT_TAG=${VARIANT_TAG} \
  /home/haolingp/CMU_research_SMT/scripts/train/train_consensus_s.sh)
echo "train           : ${TRAIN}"

# 6) infer + eval watcher
INFER=$(sbatch --parsable --dependency=afterok:${TRAIN} \
  --export=ALL,EXP=${EXP} \
  /home/haolingp/CMU_research_SMT/scripts/infer/run_infer_after_train_generic.sbatch)
echo "infer/eval      : ${INFER}"

echo
echo "CHAIN: decode=${DECODE_JID} -> pfix=${PFIX} -> [segale/qe ... fin=${QE_FIN_JID}] -> len=${LEN} -> conv=${CONV} -> train=${TRAIN} -> infer/eval=${INFER}"
echo "EXP=${EXP}"
