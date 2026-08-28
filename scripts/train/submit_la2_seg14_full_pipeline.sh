#!/usr/bin/env bash
# LA-2 (rand40k_seg14, N=2 + random window 1..14) full chain.
#   1) convert2swift_LA   -> manifest + chunked audio (+ downsample)
#   2) train_LA_s.sh      (afterok:1)
#   3) launch_infer_after_train.sbatch (afterok:2)
#        -> infer_slurm.sh array + run_eval_all_ckpts.sbatch
#
# All three stages submitted with --requeue so SLURM auto-restarts on
# node failure / preemption. Note: training will restart from base on
# requeue (not mid-ckpt), but train_LA_* runs are ~1h so cost is bounded.

set -e

VARIANT_TAG="${VARIANT_TAG:-40k-seg14-LA2}"
SAMPLE_N="${SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PROMPT_TYPE="${PROMPT_TYPE:-Standard}"

MANIFEST_ROOT="${MANIFEST_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_seg14_segale/qe3_lr}"
EXP_NAME=gigaspeech-zh-LA-${VARIANT_TAG}-s-bsz4
TRAIN_DIR=/home/haolingp/CMU_research_SMT/scripts/train

cd "${TRAIN_DIR}"
mkdir -p slurm_logs

WAIT_FOR_DEP=""
if [[ -n "${WAIT_FOR:-}" ]]; then
  WAIT_FOR_DEP="--dependency=afterany:${WAIT_FOR}"
  echo "[wait] step 1 will wait for: ${WAIT_FOR}"
fi

echo "[1/3] convert2swift_LA  (--requeue)"
CONV_JID=$(sbatch --parsable \
  --requeue \
  ${WAIT_FOR_DEP} \
  --export="ALL,MANIFEST_ROOT=${MANIFEST_ROOT},VARIANT_TAG=${VARIANT_TAG},SAMPLE_N=${SAMPLE_N},SAMPLE_SEED=${SAMPLE_SEED}" \
  ${TRAIN_DIR}/run_convert2swift_LA.sbatch)
echo "  conv jid : ${CONV_JID}"

echo "[2/3] train_LA_s.sh  (--requeue, afterok:${CONV_JID})"
TRAIN_JID=$(sbatch --parsable \
  --requeue \
  --dependency=afterok:${CONV_JID} \
  --job-name=train_LA_${VARIANT_TAG} \
  --export="ALL,VARIANT_TAG=${VARIANT_TAG}" \
  ${TRAIN_DIR}/train_LA_s.sh)
echo "  train jid: ${TRAIN_JID}"

echo "[3/3] infer + longform eval  (--requeue, afterok:${TRAIN_JID})"
INFER_LAUNCH_JID=$(sbatch --parsable \
  --requeue \
  --dependency=afterok:${TRAIN_JID} \
  --export="ALL,EXP_NAME=${EXP_NAME},PROMPT_TYPE=${PROMPT_TYPE}" \
  ${TRAIN_DIR}/launch_infer_after_train.sbatch)
echo "  infer-launch jid: ${INFER_LAUNCH_JID}"

cat <<EOF

============================================================
LA-2 seg14 pipeline submitted (with --requeue on each stage).
  conv2swift_LA       : ${CONV_JID}
  train_LA_s          : ${TRAIN_JID}    (afterok:${CONV_JID})
  infer+eval launcher : ${INFER_LAUNCH_JID}    (afterok:${TRAIN_JID})

  EXP_NAME      : ${EXP_NAME}
  manifest_root : ${MANIFEST_ROOT}
  variant_tag   : ${VARIANT_TAG}
  sample_n      : ${SAMPLE_N}
  prompt_type   : ${PROMPT_TYPE}

After completion:
  ckpt:   /data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}/v*-hf/
  scores: \${ckpt}/evaluation/acl_6060/en-zh/seg{960,1920,2880,3840}/scores.tsv
  long  : \${ckpt}/evaluation/acl_6060/en-zh/seg{...}/segmentation_output/scores.tsv
============================================================
EOF
