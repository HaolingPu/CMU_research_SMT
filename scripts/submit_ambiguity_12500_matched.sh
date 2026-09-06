#!/usr/bin/env bash
# Submit a count-matched rerun of the completed ambiguity experiment.
set -euo pipefail

REPO=/home/haolingp/CMU_research_SMT
BASE_TAG=ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831
VARIANT_TAG=${BASE_TAG}-n12500-seed42
EXP=gigaspeech-zh-consensus-${VARIANT_TAG}-s-bsz4
RUN_DIR=/home/haolingp/slurm_runs/${VARIANT_TAG}
MANIFEST=${RUN_DIR}/run_manifest.txt
DATA_DIR=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests
SOURCE_MANIFEST=${DATA_DIR}/train_s_zh-consensus-${BASE_TAG}_full.jsonl
TARGET_MANIFEST=${DATA_DIR}/train_s_zh-consensus-${VARIANT_TAG}.jsonl
CKPTS_FILE=${RUN_DIR}/ckpts.txt
CKPTS_SIMULTST_FILE=${RUN_DIR}/ckpts_simultst.txt

TRAIN_EXCLUDE=babel-p9-32,babel-o9-24,babel-q9-32,babel-t5-28
INFER_EXCLUDE=babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-n5-32,babel-o5-16,babel-n5-28,babel-q5-32,babel-s5-24,babel-q5-24,babel-o5-28,babel-p5-20,babel-p5-24,babel-o9-24,babel-o9-28,babel-q9-32,babel-t5-28
INFER_EXCLUDE_ENCODED=${INFER_EXCLUDE//,/;}

mkdir -p "${RUN_DIR}/logs"
if [[ -s "${MANIFEST}" ]]; then
  echo "ERROR: ${MANIFEST} already exists; refusing duplicate submission" >&2
  exit 1
fi

RESAMPLE=$(sbatch --parsable \
  --output="${RUN_DIR}/logs/resample_%j.out" \
  --error="${RUN_DIR}/logs/resample_%j.err" \
  --export="ALL,SOURCE_MANIFEST=${SOURCE_MANIFEST},TARGET_MANIFEST=${TARGET_MANIFEST},SAMPLE_N=12500,SAMPLE_SEED=42" \
  "${REPO}/scripts/train/run_resample_consensus_manifest.sbatch")

TRAIN=$(sbatch --parsable \
  --job-name=ambiguity-n12500-train \
  --partition=preempt \
  --qos=preempt_qos \
  --exclude="${TRAIN_EXCLUDE}" \
  --dependency="afterok:${RESAMPLE}" \
  --output="${RUN_DIR}/logs/train_%A_%a.out" \
  --error="${RUN_DIR}/logs/train_%A_%a.err" \
  --export="ALL,VARIANT_TAG=${VARIANT_TAG}" \
  "${REPO}/scripts/train/train_consensus_s.sh")

EVAL_LAUNCHER=$(sbatch --parsable \
  --partition=preempt \
  --qos=preempt_cpu_qos \
  --dependency="afterok:${TRAIN}" \
  --output="${RUN_DIR}/logs/eval_launcher_%j.out" \
  --error="${RUN_DIR}/logs/eval_launcher_%j.err" \
  --export="ALL,EXP=${EXP},RUN_SIMULTST=1,CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CHILD_EXCLUDE_ENCODED=${INFER_EXCLUDE_ENCODED},CKPTS_FILE=${CKPTS_FILE},CKPTS_SIMULTST_FILE=${CKPTS_SIMULTST_FILE},PIPELINE_MANIFEST=${MANIFEST}" \
  "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")

cat > "${MANIFEST}" <<EOF
run_tag=${VARIANT_TAG}
purpose=count_matched_control_for_${BASE_TAG}
source_survivor_pool=17306
train_sample_n=12500
sample_seed=42
source_manifest=${SOURCE_MANIFEST}
training_manifest=${TARGET_MANIFEST}
experiment=${EXP}
resample=${RESAMPLE}
train=${TRAIN}
eval_launcher=${EVAL_LAUNCHER}
evaluation_sets=acl_6060,simul_tst_common
infer_exclude=${INFER_EXCLUDE}
status=submitted
EOF

printf 'resample=%s\ntrain=%s\neval_launcher=%s\nmanifest=%s\n' \
  "${RESAMPLE}" "${TRAIN}" "${EVAL_LAUNCHER}" "${MANIFEST}"
