#!/usr/bin/env bash
# Submit the 40K ambiguity-ICL consensus experiment after prompt approval.

set -euo pipefail

REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
SEG="${FS}/scripts/segale"
RUNS_ROOT="/home/haolingp/slurm_runs"
DATA_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"

TOTAL_ROWS="${TOTAL_ROWS:-40000}"
NUM_DECODE_TASKS="${NUM_DECODE_TASKS:-12}"
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-12}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-8}"
NUM_POST_SHARDS="${NUM_POST_SHARDS:-24}"
POST_CONCURRENCY="${POST_CONCURRENCY:-24}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
TRAIN_SAMPLE_N="${TRAIN_SAMPLE_N:-40000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
PROMPT_VERSION="future_set_v2_two_groups"
PREFIX_NORMALIZATION="case-insensitive-word-boundary"
VALIDATION_PILOT="future-set-v2-prefixnorm-pilot10-r1-20260831-113627"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_ambiguity_40k.sh plan [RUN_TAG]
  bash scripts/submit_ambiguity_40k.sh submit [RUN_TAG]
  bash scripts/submit_ambiguity_40k.sh status RUN_TAG

`plan` prints and validates the experiment without submitting jobs. `submit`
queues decode, filtering, 40K training, inference, BLEU, latency, and COMET.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_login_node() {
  command -v sbatch >/dev/null || die "sbatch is unavailable; run on a BABEL login node"
  [[ "$(hostname)" == login* ]] || die "run on a BABEL login node, not $(hostname)"
  [[ -d "${REPO}/.git" ]] || die "missing checkout: ${REPO}"
}

validate_config() {
  (( TOTAL_ROWS > 0 )) || die "TOTAL_ROWS must be positive"
  (( NUM_DECODE_TASKS > 0 )) || die "NUM_DECODE_TASKS must be positive"
  (( DECODE_CONCURRENCY > 0 && DECODE_CONCURRENCY <= NUM_DECODE_TASKS )) || \
    die "DECODE_CONCURRENCY must be in [1, NUM_DECODE_TASKS]"
  (( 2 * DECODE_CONCURRENCY <= 24 )) || \
    die "decode requests $((2 * DECODE_CONCURRENCY)) GPUs; BABEL limit is 24"
  (( NUM_CONCURRENT_CASES > 0 && NUM_CONCURRENT_CASES <= 16 )) || \
    die "NUM_CONCURRENT_CASES must be in [1, 16] to match the sampler servers"
  (( POST_CONCURRENCY > 0 && POST_CONCURRENCY <= NUM_POST_SHARDS )) || \
    die "POST_CONCURRENCY must be in [1, NUM_POST_SHARDS]"
  (( POST_CONCURRENCY <= 24 )) || die "post-processing exceeds the 24-GPU limit"
  (( TARGETED_NUM_FUTURES > 0 && TARGETED_NUM_FUTURES % 2 == 0 )) || \
    die "TARGETED_NUM_FUTURES must be a positive even number"
  (( TRAIN_SAMPLE_N >= 0 )) || die "TRAIN_SAMPLE_N cannot be negative"
}

print_plan() {
  local run_tag=$1
  cat <<EOF
Run tag              : ${run_tag}
Input                 : ${INPUT_TSV}
Decode                : ${TOTAL_ROWS} rows, ${NUM_DECODE_TASKS} tasks, ${DECODE_CONCURRENCY} concurrent
Decode GPUs           : 2/job x ${DECODE_CONCURRENCY} = $((2 * DECODE_CONCURRENCY))
Cases per worker      : ${NUM_CONCURRENT_CASES}
GPU 0                 : Qwen3.8-27B-FP8 + Gemma-4-E2B samplers
GPU 1                 : Qwen3.6-35B-A3B-FP8 translator/probe
Sampler prompt        : ${PROMPT_VERSION} ($((TARGETED_NUM_FUTURES / 2)) plausible + $((TARGETED_NUM_FUTURES / 2)) contrastive per model)
Prefix normalization  : ${PREFIX_NORMALIZATION}
Consensus             : min_voters_ratio=${MIN_VOTERS_RATIO}
Post-processing       : SEGALE ${NUM_POST_SHARDS} shards, MetricX QE <= ${QE_THRESHOLD}, length ${MIN_RATIO_REF}:${MAX_RATIO_REF}
Training sample target: ${TRAIN_SAMPLE_N} (0 means all surviving examples)
Final evaluation      : BLEU + latency + Unbabel/XCOMET-XL COMET on ACL 6060 dev and Simul-tst-COMMON
Output                : ${DATA_ROOT}/${run_tag}
Model                 : /data/user_data/haolingp/ckpts/infinisst-omni/gigaspeech-zh-consensus-${run_tag}-s-bsz4
EOF
}

status_run() {
  local run_tag="${1:?RUN_TAG required}"
  local manifest="${RUNS_ROOT}/${run_tag}/run_manifest.txt"
  [[ -f "${manifest}" ]] || die "manifest not found: ${manifest}"
  cat "${manifest}"
  echo
  squeue -u haolingp -o "%.18i %.10P %.26j %.2t %.10M %.10l %R"
}

submit_run() {
  local run_tag=$1
  require_login_node
  validate_config

  local run_dir="${RUNS_ROOT}/${run_tag}"
  local log_dir="${run_dir}/logs"
  local manifest="${run_dir}/run_manifest.txt"
  local decode_root="${DATA_ROOT}/${run_tag}"
  local post_root="${decode_root}-segale"
  local shards_root="${post_root}/shards"
  local aligned_merged="${post_root}/aligned_all.jsonl"
  local metricx_root="${post_root}/metricx-aligned"
  local qe_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max"
  local length_filtered="${post_root}/qe${QE_THRESHOLD%.*}-aligned-max-len"
  local exp="gigaspeech-zh-consensus-${run_tag}-s-bsz4"
  local ckpts_file="${run_dir}/ckpts.txt"

  [[ ! -e "${run_dir}" ]] || die "run already exists: ${run_dir}"
  mkdir -p "${log_dir}"

  local git_commit
  git_commit=$(git -C "${REPO}" rev-parse HEAD)
  local git_branch
  git_branch=$(git -C "${REPO}" branch --show-current)
  cat >"${manifest}" <<EOF
run_tag=${run_tag}
created=$(date --iso-8601=seconds)
git_commit=${git_commit}
git_branch=${git_branch}
prompt_version=${PROMPT_VERSION}
prefix_normalization=${PREFIX_NORMALIZATION}
validation_pilot=${VALIDATION_PILOT}
sampler_1=gemma-4-E2B-it
sampler_2=Qwen3.8-27B-FP8
translator_probe=Qwen3.6-35B-A3B-FP8
input_tsv=${INPUT_TSV}
total_rows=${TOTAL_ROWS}
num_decode_tasks=${NUM_DECODE_TASKS}
decode_concurrency=${DECODE_CONCURRENCY}
decode_gpu_peak=$((2 * DECODE_CONCURRENCY))
num_concurrent_cases=${NUM_CONCURRENT_CASES}
targeted_num_futures=${TARGETED_NUM_FUTURES}
plausible_per_sampler=$((TARGETED_NUM_FUTURES / 2))
contrastive_per_sampler=$((TARGETED_NUM_FUTURES / 2))
max_raw_candidates_per_prefix=$((2 * TARGETED_NUM_FUTURES))
min_voters_ratio=${MIN_VOTERS_RATIO}
future_source_window=1
num_post_shards=${NUM_POST_SHARDS}
post_concurrency=${POST_CONCURRENCY}
qe_threshold=${QE_THRESHOLD}
length_ratio_ref=${MIN_RATIO_REF}:${MAX_RATIO_REF}
train_sample_n=${TRAIN_SAMPLE_N}
sample_seed=${SAMPLE_SEED}
quality_metrics=BLEU,Unbabel/XCOMET-XL
eval_sets=acl_6060_dev,simul_tst_common
decode_root=${decode_root}
post_root=${post_root}
length_filtered=${length_filtered}
experiment=${exp}
EOF

  print_plan "${run_tag}"
  echo "Submitting from commit ${git_commit}"

  local decode_jid
  decode_jid=$(sbatch --parsable \
    --array="0-$((NUM_DECODE_TASKS - 1))%${DECODE_CONCURRENCY}" \
    --output="${log_dir}/decode_%A_%a.out" --error="${log_dir}/decode_%A_%a.err" \
    --export="ALL,INPUT_TSV=${INPUT_TSV},OUTPUT_ROOT=${decode_root},TOTAL_ROWS=${TOTAL_ROWS},NUM_TASKS=${NUM_DECODE_TASKS},NUM_CONCURRENT_CASES=${NUM_CONCURRENT_CASES},TARGETED_NUM_FUTURES=${TARGETED_NUM_FUTURES},MIN_VOTERS_RATIO=${MIN_VOTERS_RATIO},FUTURE_SRC_WINDOW=1" \
    "${FS}/run_ambiguity_q38_gemma_q36_preempt.sbatch")
  echo "decode=${decode_jid}" | tee -a "${manifest}"

  local prep_jid
  prep_jid=$(sbatch --parsable --dependency="afterok:${decode_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/segale_prepare_%j.out" --error="${log_dir}/segale_prepare_%j.err" \
    --export="ALL,CONSENSUS_ROOT=${decode_root},OUT_ROOT=${post_root},NUM_DOCS=${TOTAL_ROWS},SYS_ID=${run_tag},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_prepare_shards_24.sbatch")
  echo "segale_prepare=${prep_jid}" | tee -a "${manifest}"

  local align_jid
  align_jid=$(sbatch --parsable --dependency="afterok:${prep_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
    --output="${log_dir}/segale_%A_%a.out" --error="${log_dir}/segale_%A_%a.err" \
    --export="ALL,SHARDS_ROOT=${shards_root}" "${SEG}/run_segale_align_24gpu_preempt.sbatch")
  echo "segale_align=${align_jid}" | tee -a "${manifest}"

  local merge_jid
  merge_jid=$(sbatch --parsable --dependency="afterok:${align_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output="${log_dir}/segale_merge_%j.out" --error="${log_dir}/segale_merge_%j.err" \
    --wrap="source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && PYTHONNOUSERSITE=1 python ${SEG}/merge_aligned_shards.py --shards-root ${shards_root} --output ${aligned_merged} --num-shards ${NUM_POST_SHARDS}")
  echo "segale_merge=${merge_jid}" | tee -a "${manifest}"

  local qe_prep_jid
  qe_prep_jid=$(sbatch --parsable --dependency="afterok:${merge_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/qe_prepare_%j.out" --error="${log_dir}/qe_prepare_%j.err" \
    --export="ALL,ALIGNED_FILE=${aligned_merged},CONSENSUS_ROOT=${decode_root},METRICX_RUN_DIR=${metricx_root},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_qe_prepare_24.sbatch")
  echo "qe_prepare=${qe_prep_jid}" | tee -a "${manifest}"

  local qe_predict_jid
  qe_predict_jid=$(sbatch --parsable --dependency="afterok:${qe_prep_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --array="0-$((NUM_POST_SHARDS - 1))%${POST_CONCURRENCY}" \
    --output="${log_dir}/metricx_%A_%a.out" --error="${log_dir}/metricx_%A_%a.err" \
    --export="ALL,METRICX_RUN_DIR=${metricx_root}" "${SEG}/run_metricx_qe_24gpu_preempt.sbatch")
  echo "qe_predict=${qe_predict_jid}" | tee -a "${manifest}"

  local qe_finalize_jid
  qe_finalize_jid=$(sbatch --parsable --dependency="afterok:${qe_predict_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/qe_finalize_%j.out" --error="${log_dir}/qe_finalize_%j.err" \
    --export="ALL,METRICX_RUN_DIR=${metricx_root},QE_FILTERED_DIR=${qe_filtered},QE_THRESHOLD=${QE_THRESHOLD},NUM_SHARDS=${NUM_POST_SHARDS}" \
    "${SEG}/run_qe_finalize_24.sbatch")
  echo "qe_finalize=${qe_finalize_jid}" | tee -a "${manifest}"

  local length_jid
  length_jid=$(sbatch --parsable --dependency="afterok:${qe_finalize_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/length_%j.out" --error="${log_dir}/length_%j.err" \
    --export="ALL,INPUT_DIR=${qe_filtered},OUTPUT_DIR=${length_filtered},MIN_RATIO_REF=${MIN_RATIO_REF},MAX_RATIO_REF=${MAX_RATIO_REF}" \
    "${SEG}/run_length_ratio_filter.sbatch")
  echo "length_filter=${length_jid}" | tee -a "${manifest}"

  local convert_jid
  convert_jid=$(sbatch --parsable --dependency="afterok:${length_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/convert_%j.out" --error="${log_dir}/convert_%j.err" \
    --export="ALL,MANIFEST_ROOT=${length_filtered},VARIANT_TAG=${run_tag},SAMPLE_N=${TRAIN_SAMPLE_N},SAMPLE_SEED=${SAMPLE_SEED}" \
    "${REPO}/scripts/train/run_convert2swift_consensus.sbatch")
  echo "convert=${convert_jid}" | tee -a "${manifest}"

  local train_jid
  train_jid=$(sbatch --parsable --dependency="afterok:${convert_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --output="${log_dir}/train_%A_%a.out" --error="${log_dir}/train_%A_%a.err" \
    --export="ALL,VARIANT_TAG=${run_tag}" "${REPO}/scripts/train/train_consensus_s.sh")
  echo "train=${train_jid}" | tee -a "${manifest}"

  local eval_launcher_jid
  eval_launcher_jid=$(sbatch --parsable --dependency="afterok:${train_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/eval_launcher_%j.out" --error="${log_dir}/eval_launcher_%j.err" \
    --export="ALL,EXP=${exp},CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CKPTS_FILE=${ckpts_file},RUN_SIMULTST=1,CKPTS_SIMULTST_FILE=${run_dir}/ckpts_simultst.txt,PIPELINE_MANIFEST=${manifest}" \
    "${REPO}/scripts/infer/run_infer_after_train_generic.sbatch")
  echo "eval_launcher=${eval_launcher_jid}" | tee -a "${manifest}"

  echo
  echo "Submitted ${run_tag}"
  echo "Manifest: ${manifest}"
  echo "No stage runs before its afterok dependency succeeds."
}

case "${1:-plan}" in
  plan)
    validate_config
    print_plan "${2:-ambiguity-q38-gemma-q36-strict-40k-$(date +%Y%m%d)}"
    ;;
  submit)
    submit_run "${2:-ambiguity-q38-gemma-q36-strict-40k-$(date +%Y%m%d-%H%M%S)}"
    ;;
  status)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    status_run "$2"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
