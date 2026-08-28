#!/usr/bin/env bash
# ============================================================
# en->ja EAST-latency2mult full chain (sub-sentence QE filtered, ~37.5K traj):
#   1) run_convert2swift_east-mult_ja.sbatch
#        builds train_s_ja-EAST-latency2mult_origin.jsonl + chunked audio
#   2) train_EAST-latency2mult_s_ja.sh                 (afterok:1)
#        LoRA SFT on Qwen3-Omni-30B-A3B-Instruct, 4xL40S, 1 epoch
#   3) launch_infer_after_train_ja.sbatch              (afterok:2)
#        chains internally:
#          - infer_slurm_ja.sh             (4-array simuleval, en-ja)
#          - run_eval_all_ckpts_ja.sbatch  (afterok)  longform BLEU/COMET
#
# All steps run on partition=general, qos=normal. Single-job GPU peak:
#   step 1: 0 GPU (CPU only, soundfile I/O)
#   step 2: 4 L40S
#   step 3: 1 launcher GPU + (during stage A) 4 array tasks x 2 L40S = 8 L40S
#           + (after stage A) 1 longform GPU
# Sequential, so the 8-GPU general quota is enough at every moment.
# ============================================================
set -e

PROMPT_TYPE="${PROMPT_TYPE:-Standard}"
EXP_NAME="${EXP_NAME:-gigaspeech-ja-EAST-latency2mult-s_origin-bsz4_ja}"

TRAIN_DIR=/home/haolingp/CMU_research_SMT/scripts/train
mkdir -p ${TRAIN_DIR}/slurm_logs
cd ${TRAIN_DIR}

# Optional: WAIT_FOR="JID1[:JID2:...]" -> step 1 starts only after those finish (afterany).
WAIT_FOR_DEP=""
if [[ -n "${WAIT_FOR:-}" ]]; then
  WAIT_FOR_DEP="--dependency=afterany:${WAIT_FOR}"
  echo "[wait] step 1 will wait for: ${WAIT_FOR}"
fi

echo "[1/3] convert2swift_east-mult_ja"
CONV_JID=$(sbatch --parsable \
  ${WAIT_FOR_DEP} \
  ${TRAIN_DIR}/run_convert2swift_east-mult_ja.sbatch)
echo "  conv jid : ${CONV_JID}"

echo "[2/3] train_EAST-latency2mult_s_ja.sh  (afterok:${CONV_JID})"
TRAIN_JID=$(sbatch --parsable \
  --dependency=afterok:${CONV_JID} \
  --job-name=train_EAST_lat2mult_ja \
  ${TRAIN_DIR}/train_EAST-latency2mult_s_ja.sh)
echo "  train jid: ${TRAIN_JID}"

echo "[3/3] infer + longform eval (ja)  (afterok:${TRAIN_JID})"
INFER_LAUNCH_JID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_JID} \
  --export="ALL,EXP_NAME=${EXP_NAME},PROMPT_TYPE=${PROMPT_TYPE}" \
  ${TRAIN_DIR}/launch_infer_after_train_ja.sbatch)
echo "  infer-launch jid: ${INFER_LAUNCH_JID}"

cat <<EOF

============================================================
en->ja EAST-latency2mult pipeline submitted.
  conv2swift_east_ja  : ${CONV_JID}
  train_EAST_lat2mult : ${TRAIN_JID}             (afterok:${CONV_JID})
  infer+eval launcher : ${INFER_LAUNCH_JID}      (afterok:${TRAIN_JID})

  EXP_NAME    : ${EXP_NAME}
  prompt_type : ${PROMPT_TYPE}

After completion:
  ckpt   : /data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}/v*-hf/
  scores : \${ckpt}/evaluation/acl_6060/en-ja/seg{960,1920,2880,3840}/scores.tsv
  long   : \${ckpt}/evaluation/acl_6060/en-ja/seg{...}/segmentation_output/scores.tsv

Watch:
  squeue -u haolingp
  tail -f ${TRAIN_DIR}/slurm_logs/conv2swift_east_ja_${CONV_JID}.out
============================================================
EOF
