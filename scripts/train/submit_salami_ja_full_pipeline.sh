#!/usr/bin/env bash
# en->ja SALAMI (Simul-MuST-C style) full chain:
#   1) run_convert2swift_salami_ja.sbatch   (final_jsonl_salami -> swift manifest)
#   2) train_Simul-MuST-C_s_ja.sh           (afterok:1)  LoRA SFT, 4xL40S
#   3) launch_infer_after_train_salami_ja.sbatch (afterok:2)
#
# Optional env: WAIT_FOR="JID1[:JID2:...]" -> step 1 waits afterok on those first.
set -e

PROMPT_TYPE="${PROMPT_TYPE:-Standard}"
EXP_NAME="${EXP_NAME:-gigaspeech-ja-Simul-MuST-C-s_origin-bsz4_ja}"

TRAIN_DIR=/home/haolingp/CMU_research_SMT/scripts/train
mkdir -p ${TRAIN_DIR}/slurm_logs
cd ${TRAIN_DIR}

WAIT_DEP=""
if [[ -n "${WAIT_FOR:-}" ]]; then
  WAIT_DEP="--dependency=afterok:${WAIT_FOR}"
  echo "[wait] step 1 will wait afterok on: ${WAIT_FOR}"
fi

echo "[1/3] convert2swift_salami_ja"
CONV_JID=$(sbatch --parsable ${WAIT_DEP} \
  ${TRAIN_DIR}/run_convert2swift_salami_ja.sbatch)
echo "  conv jid : ${CONV_JID}"

echo "[2/3] train_Simul-MuST-C_s_ja  (afterok:${CONV_JID})"
TRAIN_JID=$(sbatch --parsable \
  --dependency=afterok:${CONV_JID} \
  --job-name=train_SALAMI_ja \
  ${TRAIN_DIR}/train_Simul-MuST-C_s_ja.sh)
echo "  train jid: ${TRAIN_JID}"

echo "[3/3] infer + longform eval  (afterok:${TRAIN_JID})"
INFER_LAUNCH_JID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_JID} \
  --export="ALL,EXP_NAME=${EXP_NAME},PROMPT_TYPE=${PROMPT_TYPE}" \
  ${TRAIN_DIR}/launch_infer_after_train_salami_ja.sbatch)
echo "  infer-launch jid: ${INFER_LAUNCH_JID}"

cat <<EOF

============================================================
en->ja SALAMI pipeline submitted.
  conv2swift_salami_ja : ${CONV_JID}
  train_SALAMI_ja      : ${TRAIN_JID}             (afterok:${CONV_JID})
  infer+eval launcher  : ${INFER_LAUNCH_JID}      (afterok:${TRAIN_JID})

  EXP_NAME    : ${EXP_NAME}
  prompt_type : ${PROMPT_TYPE}

After completion:
  ckpt   : /data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}/v*-hf/
  scores : \${ckpt}/evaluation/acl_6060/en-ja/seg{960,1920,2880,3840}/scores.tsv
  long   : \${ckpt}/evaluation/acl_6060/en-ja/seg{...}/segmentation_output/scores.tsv
============================================================
EOF
