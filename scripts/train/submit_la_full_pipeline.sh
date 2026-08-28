#!/usr/bin/env bash
# ============================================================
# LA (local agreement) la_rand40k full chain:
#   1) convert2swift_LA   (build manifest + chunked audio + downsample)
#   2) train_LA_s.sh      (afterok:1)
#   3) launch_infer_after_train.sbatch  (afterok:2)
#        -> internally chains:  infer_slurm.sh (4-array simuleval)
#                              run_eval_all_ckpts.sbatch (longform/streamLAAL+COMET)
#
# After everything finishes, run plot_latency_quality.py manually with the
# new BLEU/StreamLAAL numbers (you'll edit that script's `data` dict).
# ============================================================
set -e

VARIANT_TAG="${VARIANT_TAG:-40k}"
SAMPLE_N="${SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PROMPT_TYPE="${PROMPT_TYPE:-Standard}"

MANIFEST_ROOT="${MANIFEST_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_segale/qe3_lr}"
EXP_NAME=gigaspeech-zh-LA-${VARIANT_TAG}-s-bsz4
TRAIN_DIR=/home/haolingp/CMU_research_SMT/scripts/train

cd "${TRAIN_DIR}"
mkdir -p slurm_logs

# Optional: WAIT_FOR="JID1:JID2:..." -> step 1 starts only after those finish
# (afterany: any state, so a failure in those jobs won't block this chain).
WAIT_FOR_DEP=""
if [[ -n "${WAIT_FOR:-}" ]]; then
  WAIT_FOR_DEP="--dependency=afterany:${WAIT_FOR}"
  echo "[wait] step 1 will wait for: ${WAIT_FOR}"
fi

echo "[1/3] convert2swift_LA"
CONV_JID=$(sbatch --parsable \
  ${WAIT_FOR_DEP} \
  --export="ALL,MANIFEST_ROOT=${MANIFEST_ROOT},VARIANT_TAG=${VARIANT_TAG},SAMPLE_N=${SAMPLE_N},SAMPLE_SEED=${SAMPLE_SEED}" \
  ${TRAIN_DIR}/run_convert2swift_LA.sbatch)
echo "  conv jid : ${CONV_JID}"

echo "[2/3] train_LA_s.sh  (afterok:${CONV_JID})"
TRAIN_JID=$(sbatch --parsable \
  --dependency=afterok:${CONV_JID} \
  --job-name=train_LA_${VARIANT_TAG} \
  --export="ALL,VARIANT_TAG=${VARIANT_TAG}" \
  ${TRAIN_DIR}/train_LA_s.sh)
echo "  train jid: ${TRAIN_JID}"

echo "[3/3] infer + longform eval  (afterok:${TRAIN_JID})"
INFER_LAUNCH_JID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_JID} \
  --export="ALL,EXP_NAME=${EXP_NAME},PROMPT_TYPE=${PROMPT_TYPE}" \
  ${TRAIN_DIR}/launch_infer_after_train.sbatch)
echo "  infer-launch jid: ${INFER_LAUNCH_JID}"

cat <<EOF

============================================================
LA pipeline submitted.
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

Then plot:
  edit /home/haolingp/CMU_research_SMT/scripts/infer/plot_latency_quality.py
  add  "Local-Agreement": [(LAAL,BLEU)x4]  using the seg960..seg3840 scores
  python /home/haolingp/CMU_research_SMT/scripts/infer/plot_latency_quality.py
============================================================
EOF
