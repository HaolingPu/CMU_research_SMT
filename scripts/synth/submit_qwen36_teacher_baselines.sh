#!/usr/bin/env bash
# Submit the source-matched Qwen3.6 teacher ablations.
#
# EAST generation starts first. Simul-MuST-C generation starts after EAST
# training, while evaluation gates prevent two 16-GPU evaluation suites from
# overlapping. Generation and MetricX throttles keep total usage at 24 GPUs.

set -euo pipefail

REPO=/home/haolingp/CMU_research_SMT
BASELINE_COMMIT="${BASELINE_COMMIT:-9604507}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/haolingp/slurm_runs/qwen36-teacher-baselines-${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen36_teacher_ablation/${RUN_TAG}}"
MODEL_PATH="${MODEL_PATH:-/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests}"
NUM_TASKS="${NUM_TASKS:-24}"
NUM_QE_SHARDS="${NUM_QE_SHARDS:-24}"
EAST_GEN_PARALLELISM="${EAST_GEN_PARALLELISM:-20}"
PIPELINE_PARALLELISM="${PIPELINE_PARALLELISM:-8}"
SAMPLE_SIZE=12500
SAMPLE_SEED=42
PRIOR_EVAL_GATE="${PRIOR_EVAL_GATE:-}"
START_DEPENDENCY="${START_DEPENDENCY:-}"
EXCLUDE="${EXCLUDE:-babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-n5-32,babel-o5-16,babel-n5-28,babel-q5-32,babel-s5-24,babel-q5-24,babel-o5-28,babel-p5-20,babel-p5-24,babel-o9-24,babel-q9-32,babel-t5-28,babel-v9-28}"

if [[ "${SAMPLE_SIZE}" -ne 12500 || "${SAMPLE_SEED}" -ne 42 ]]; then
  echo "The frozen recipe requires SAMPLE_SIZE=12500 and SAMPLE_SEED=42" >&2
  exit 2
fi
if [[ "${EAST_GEN_PARALLELISM}" -gt 20 || "${PIPELINE_PARALLELISM}" -gt 8 ]]; then
  echo "Configured throttles can exceed the 24-GPU cap while training/evaluation runs" >&2
  exit 2
fi

cd "${REPO}"
mkdir -p "${RUN_ROOT}/code_snapshot/salami" "${RUN_ROOT}/east/logs" "${RUN_ROOT}/simul_must_c/logs"
SNAPSHOT="${RUN_ROOT}/code_snapshot"
GENERATION_RUNTIME="${RUN_ROOT}/generation_runtime"

snapshot_file() {
  local path=$1
  mkdir -p "${SNAPSHOT}/$(dirname "${path}")"
  git show "${BASELINE_COMMIT}:${path}" > "${SNAPSHOT}/${path}"
}

snapshot_file data_synthesis/codes/gigaspeech/llm_output_gigaspeech_trajectory.py
snapshot_file data_synthesis/codes/gigaspeech/salami/llm_output_salami.py
snapshot_file data_synthesis/codes/gigaspeech/fix_llm_raw.py
snapshot_file data_synthesis/codes/gigaspeech/post_process_llm_output_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/find_bad_json_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/multi_trajectory_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/check_streaming_dataset.py
snapshot_file data_synthesis/codes/gigaspeech/convert_metricx_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/filter_metricx_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/final_output_gigaspeech.py
snapshot_file data_synthesis/codes/gigaspeech/check_salami_final.py
snapshot_file data_synthesis/codes/gigaspeech/salami/map_salami_to_offline_gigaspeech.py
CODE_SNAPSHOT="${SNAPSHOT}/data_synthesis/codes/gigaspeech"
GENERATION_CODE="${GENERATION_RUNTIME}/data_synthesis/codes/gigaspeech"
python scripts/synth/patch_legacy_teacher_generator.py \
  --input "${CODE_SNAPSHOT}/llm_output_gigaspeech_trajectory.py" \
  --output "${GENERATION_CODE}/llm_output_gigaspeech_trajectory.py"
python scripts/synth/patch_legacy_teacher_generator.py \
  --input "${CODE_SNAPSHOT}/salami/llm_output_salami.py" \
  --output "${GENERATION_CODE}/salami/llm_output_salami.py"

EAST_VARIANT="east-even-qwen36-teacher-${RUN_TAG}-n12500-seed42"
SIMUL_VARIANT="simul-must-c-fixed-v2-qwen36-teacher-${RUN_TAG}-n12500-seed42"
EAST_EXP="gigaspeech-zh-consensus-${EAST_VARIANT}-s-bsz4"
SIMUL_EXP="gigaspeech-zh-consensus-${SIMUL_VARIANT}-s-bsz4"
EAST_BASE="${OUTPUT_ROOT}/east"
SIMUL_BASE="${OUTPUT_ROOT}/simul_must_c"
EAST_MANIFEST="${MANIFEST_ROOT}/train_s_zh-consensus-${EAST_VARIANT}.jsonl"
SIMUL_MANIFEST="${MANIFEST_ROOT}/train_s_zh-consensus-${SIMUL_VARIANT}.jsonl"
EAST_RUN_MANIFEST="${RUN_ROOT}/east/run_manifest.txt"
SIMUL_RUN_MANIFEST="${RUN_ROOT}/simul_must_c/run_manifest.txt"

cat > "${EAST_RUN_MANIFEST}" <<EOF
method=EAST-even
teacher_model=${MODEL_PATH}
baseline_code_commit=${BASELINE_COMMIT}
training_examples=${SAMPLE_SIZE}
sample_seed=${SAMPLE_SEED}
training_script=${REPO}/scripts/train/train_consensus_s.sh
training_recipe=frozen_unchanged
output_root=${EAST_BASE}
training_manifest=${EAST_MANIFEST}
experiment=${EAST_EXP}
EOF
cat > "${SIMUL_RUN_MANIFEST}" <<EOF
method=Simul-MuST-C-fixed-v2
teacher_model=${MODEL_PATH}
baseline_code_commit=${BASELINE_COMMIT}
training_examples=${SAMPLE_SIZE}
sample_seed=${SAMPLE_SEED}
training_script=${REPO}/scripts/train/train_consensus_s.sh
training_recipe=frozen_unchanged
output_root=${SIMUL_BASE}
training_manifest=${SIMUL_MANIFEST}
experiment=${SIMUL_EXP}
EOF

GPU_ARGS=(--partition=preempt --qos=preempt_qos --requeue --exclude="${EXCLUDE}")
CPU_ARGS=(--partition=preempt --qos=preempt_cpu_qos --requeue)
EAST_START_ARGS=()
if [[ -n "${START_DEPENDENCY}" ]]; then
  EAST_START_ARGS+=(--dependency="${START_DEPENDENCY}")
fi

EAST_GEN=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  "${EAST_START_ARGS[@]}" \
  --array="0-$((NUM_TASKS - 1))%${EAST_GEN_PARALLELISM}" \
  --job-name=q36_east_gen \
  --output="${RUN_ROOT}/east/logs/generate_%A_%a.out" \
  --error="${RUN_ROOT}/east/logs/generate_%A_%a.err" \
  --export="ALL,METHOD=east,BASE=${EAST_BASE},CODE_SNAPSHOT=${GENERATION_CODE},MODEL_PATH=${MODEL_PATH},INPUT_TSV=${INPUT_TSV},NUM_TASKS=${NUM_TASKS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_generate.sbatch")

EAST_PREP=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterok:${EAST_GEN}" \
  --job-name=q36_east_prepare \
  --output="${RUN_ROOT}/east/logs/prepare_%j.out" \
  --error="${RUN_ROOT}/east/logs/prepare_%j.err" \
  --export="ALL,METHOD=east,BASE=${EAST_BASE},CODE_SNAPSHOT=${CODE_SNAPSHOT},INPUT_TSV=${INPUT_TSV},NUM_QE_SHARDS=${NUM_QE_SHARDS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_prepare.sbatch")

EAST_QE=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  --dependency="afterok:${EAST_PREP}" \
  --array="0-$((NUM_QE_SHARDS - 1))%${PIPELINE_PARALLELISM}" \
  --job-name=q36_east_metricx \
  --output="${RUN_ROOT}/east/logs/metricx_%A_%a.out" \
  --error="${RUN_ROOT}/east/logs/metricx_%A_%a.err" \
  --export="ALL,BASE=${EAST_BASE},NUM_QE_SHARDS=${NUM_QE_SHARDS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_metricx.sbatch")

EAST_FINAL=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterok:${EAST_QE}" \
  --job-name=q36_east_finalize \
  --output="${RUN_ROOT}/east/logs/finalize_%j.out" \
  --error="${RUN_ROOT}/east/logs/finalize_%j.err" \
  --export="ALL,METHOD=east,BASE=${EAST_BASE},CODE_SNAPSHOT=${CODE_SNAPSHOT},OUTPUT_MANIFEST=${EAST_MANIFEST},POOL_MANIFEST=${MANIFEST_ROOT}/train_xl_case_robust_asr-filtered_zh-EAST-latency2mult.jsonl,TEMPLATE_MANIFEST=${MANIFEST_ROOT}/train_s_zh-EAST-even12500.jsonl,NUM_QE_SHARDS=${NUM_QE_SHARDS},SAMPLE_SIZE=${SAMPLE_SIZE},SAMPLE_SEED=${SAMPLE_SEED},INPUT_TSV=${INPUT_TSV}" \
  "${REPO}/scripts/synth/run_teacher_baseline_finalize.sbatch")

EAST_TRAIN=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  --dependency="afterok:${EAST_FINAL}" \
  --array=2 \
  --job-name=q36_east_train \
  --output="${RUN_ROOT}/east/logs/train_%A_%a.out" \
  --error="${RUN_ROOT}/east/logs/train_%A_%a.err" \
  --export="ALL,VARIANT_TAG=${EAST_VARIANT}" \
  "${REPO}/scripts/train/train_consensus_s.sh")

EAST_EVAL_DEP="afterok:${EAST_TRAIN}"
if [[ -n "${PRIOR_EVAL_GATE}" ]]; then
  EAST_EVAL_DEP="${EAST_EVAL_DEP}:${PRIOR_EVAL_GATE}"
fi
EAST_LAUNCH=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="${EAST_EVAL_DEP}" \
  --job-name=q36_east_eval_launch \
  --output="${RUN_ROOT}/east/logs/eval_launcher_%j.out" \
  --error="${RUN_ROOT}/east/logs/eval_launcher_%j.err" \
  --export="ALL,EXP=${EAST_EXP},RUN_SIMULTST=1,CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CHILD_EXCLUDE=${EXCLUDE},CKPTS_FILE=${RUN_ROOT}/east/ckpts.txt,CKPTS_SIMULTST_FILE=${RUN_ROOT}/east/ckpts_simultst.txt,PIPELINE_MANIFEST=${EAST_RUN_MANIFEST}" \
  "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")
EAST_GATE=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterany:${EAST_LAUNCH}" \
  --job-name=q36_east_eval_gate \
  --output="${RUN_ROOT}/east/logs/eval_gate_%j.out" \
  --error="${RUN_ROOT}/east/logs/eval_gate_%j.err" \
  --export="ALL,PIPELINE_MANIFEST=${EAST_RUN_MANIFEST}" \
  "${REPO}/scripts/infer/wait_for_manifest_evaluations.sbatch")

SIMUL_GEN=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  --dependency="afterok:${EAST_TRAIN}" \
  --array="0-$((NUM_TASKS - 1))%${PIPELINE_PARALLELISM}" \
  --job-name=q36_simul_gen \
  --output="${RUN_ROOT}/simul_must_c/logs/generate_%A_%a.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/generate_%A_%a.err" \
  --export="ALL,METHOD=simul-must-c,BASE=${SIMUL_BASE},CODE_SNAPSHOT=${GENERATION_CODE},MODEL_PATH=${MODEL_PATH},INPUT_TSV=${INPUT_TSV},NUM_TASKS=${NUM_TASKS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_generate.sbatch")

SIMUL_PREP=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterok:${SIMUL_GEN}" \
  --job-name=q36_simul_prepare \
  --output="${RUN_ROOT}/simul_must_c/logs/prepare_%j.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/prepare_%j.err" \
  --export="ALL,METHOD=simul-must-c,BASE=${SIMUL_BASE},CODE_SNAPSHOT=${CODE_SNAPSHOT},INPUT_TSV=${INPUT_TSV},NUM_QE_SHARDS=${NUM_QE_SHARDS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_prepare.sbatch")

SIMUL_QE=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  --dependency="afterok:${SIMUL_PREP}" \
  --array="0-$((NUM_QE_SHARDS - 1))%${PIPELINE_PARALLELISM}" \
  --job-name=q36_simul_metricx \
  --output="${RUN_ROOT}/simul_must_c/logs/metricx_%A_%a.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/metricx_%A_%a.err" \
  --export="ALL,BASE=${SIMUL_BASE},NUM_QE_SHARDS=${NUM_QE_SHARDS}" \
  "${REPO}/scripts/synth/run_teacher_baseline_metricx.sbatch")

SIMUL_FINAL=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterok:${SIMUL_QE}" \
  --job-name=q36_simul_finalize \
  --output="${RUN_ROOT}/simul_must_c/logs/finalize_%j.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/finalize_%j.err" \
  --export="ALL,METHOD=simul-must-c,BASE=${SIMUL_BASE},CODE_SNAPSHOT=${CODE_SNAPSHOT},OUTPUT_MANIFEST=${SIMUL_MANIFEST},POOL_MANIFEST=${MANIFEST_ROOT}/train_xl_case_robust_asr-filtered_zh-Simul-MuST-C_fixed_v2.jsonl,TEMPLATE_MANIFEST=${MANIFEST_ROOT}/train_s_zh-Simul-MuST-C-fixed_v2_origin.jsonl,NUM_QE_SHARDS=${NUM_QE_SHARDS},SAMPLE_SIZE=${SAMPLE_SIZE},SAMPLE_SEED=${SAMPLE_SEED},INPUT_TSV=${INPUT_TSV}" \
  "${REPO}/scripts/synth/run_teacher_baseline_finalize.sbatch")

SIMUL_TRAIN=$(sbatch --parsable \
  "${GPU_ARGS[@]}" \
  --dependency="afterok:${SIMUL_FINAL}" \
  --array=2 \
  --job-name=q36_simul_train \
  --output="${RUN_ROOT}/simul_must_c/logs/train_%A_%a.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/train_%A_%a.err" \
  --export="ALL,VARIANT_TAG=${SIMUL_VARIANT}" \
  "${REPO}/scripts/train/train_consensus_s.sh")

SIMUL_LAUNCH=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterok:${SIMUL_TRAIN}:${EAST_GATE}" \
  --job-name=q36_simul_eval_launch \
  --output="${RUN_ROOT}/simul_must_c/logs/eval_launcher_%j.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/eval_launcher_%j.err" \
  --export="ALL,EXP=${SIMUL_EXP},RUN_SIMULTST=1,CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CHILD_EXCLUDE=${EXCLUDE},CKPTS_FILE=${RUN_ROOT}/simul_must_c/ckpts.txt,CKPTS_SIMULTST_FILE=${RUN_ROOT}/simul_must_c/ckpts_simultst.txt,PIPELINE_MANIFEST=${SIMUL_RUN_MANIFEST}" \
  "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")
SIMUL_GATE=$(sbatch --parsable \
  "${CPU_ARGS[@]}" \
  --dependency="afterany:${SIMUL_LAUNCH}" \
  --job-name=q36_simul_eval_gate \
  --output="${RUN_ROOT}/simul_must_c/logs/eval_gate_%j.out" \
  --error="${RUN_ROOT}/simul_must_c/logs/eval_gate_%j.err" \
  --export="ALL,PIPELINE_MANIFEST=${SIMUL_RUN_MANIFEST}" \
  "${REPO}/scripts/infer/wait_for_manifest_evaluations.sbatch")

{
  printf 'generation=%s\nprepare=%s\nmetricx=%s\nfinalize=%s\ntrain=%s\neval_launcher=%s\neval_gate=%s\n' \
    "${EAST_GEN}" "${EAST_PREP}" "${EAST_QE}" "${EAST_FINAL}" "${EAST_TRAIN}" "${EAST_LAUNCH}" "${EAST_GATE}"
} >> "${EAST_RUN_MANIFEST}"
{
  printf 'generation=%s\nprepare=%s\nmetricx=%s\nfinalize=%s\ntrain=%s\neval_launcher=%s\neval_gate=%s\n' \
    "${SIMUL_GEN}" "${SIMUL_PREP}" "${SIMUL_QE}" "${SIMUL_FINAL}" "${SIMUL_TRAIN}" "${SIMUL_LAUNCH}" "${SIMUL_GATE}"
} >> "${SIMUL_RUN_MANIFEST}"

cat > "${RUN_ROOT}/run_manifest.txt" <<EOF
run_tag=${RUN_TAG}
repo_commit=$(git rev-parse HEAD)
baseline_code_commit=$(git rev-parse "${BASELINE_COMMIT}")
generation_runtime=${GENERATION_CODE}
teacher_model=${MODEL_PATH}
training_examples=${SAMPLE_SIZE}
sample_seed=${SAMPLE_SEED}
num_generation_tasks=${NUM_TASKS}
east_generation_parallelism=${EAST_GEN_PARALLELISM}
pipeline_parallelism=${PIPELINE_PARALLELISM}
gpu_cap=24
prior_eval_gate=${PRIOR_EVAL_GATE}
start_dependency=${START_DEPENDENCY}
exclude=${EXCLUDE}
east_manifest=${EAST_RUN_MANIFEST}
simul_must_c_manifest=${SIMUL_RUN_MANIFEST}
EOF

printf 'RUN_ROOT=%s\nEAST_GEN=%s\nEAST_TRAIN=%s\nEAST_GATE=%s\nSIMUL_GEN=%s\nSIMUL_TRAIN=%s\nSIMUL_GATE=%s\n' \
  "${RUN_ROOT}" "${EAST_GEN}" "${EAST_TRAIN}" "${EAST_GATE}" \
  "${SIMUL_GEN}" "${SIMUL_TRAIN}" "${SIMUL_GATE}"
