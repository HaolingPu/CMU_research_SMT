#!/usr/bin/env bash
# Submit the 35K v3-JSON + sentence-boundary consensus experiment (2026-09-08).
# Same pipeline as submit_ambiguity_40k.sh; differences: 35,000 rows, 16 cases per
# worker, the v3 suffix-ICL prompt with vLLM JSON-schema output, source-only sampler
# context, sentence-anchor window, sentence-end completion, 12,500-row training sample.

set -euo pipefail

REPO="/home/haolingp/CMU_research_SMT"
FS="${REPO}/data_synthesis/codes/gigaspeech/future_sampling"
SEG="${FS}/scripts/segale"
RUNS_ROOT="/home/haolingp/slurm_runs"
DATA_ROOT="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"

TOTAL_ROWS="${TOTAL_ROWS:-35000}"
NUM_DECODE_TASKS="${NUM_DECODE_TASKS:-24}"
DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-12}"
NUM_CONCURRENT_CASES="${NUM_CONCURRENT_CASES:-16}"
NUM_POST_SHARDS="${NUM_POST_SHARDS:-24}"
POST_CONCURRENCY="${POST_CONCURRENCY:-24}"
TARGETED_NUM_FUTURES="${TARGETED_NUM_FUTURES:-20}"
MIN_VOTERS_RATIO="${MIN_VOTERS_RATIO:-1.0}"
QE_THRESHOLD="${QE_THRESHOLD:-3.0}"
MIN_RATIO_REF="${MIN_RATIO_REF:-0.7}"
MAX_RATIO_REF="${MAX_RATIO_REF:-1.5}"
TRAIN_SAMPLE_N="${TRAIN_SAMPLE_N:-12500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
INPUT_TSV="${INPUT_TSV:-/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv}"
PROMPT_VERSION="future_set_v3_suffix_icl"
SAMPLER_OUTPUT="json_schema_structured_outputs"
PREFIX_NORMALIZATION="case-insensitive-word-boundary"
VALIDATION_PILOT="suffix-icl-v3-50cases-20260907-130701 (text) + v3json-speed-50cases-20260908T0204Z (JSON)"
# Nodes with confirmed vLLM/NCCL faults (pilot and inference exclusion lists, 2026-09).
DECODE_EXCLUDE="${DECODE_EXCLUDE:-babel-o9-24,babel-q9-32,babel-t5-28,babel-v9-28,babel-p5-24,babel-o5-28,babel-p5-20,babel-o5-24,babel-q5-16,babel-q5-20,babel-n5-32,babel-p9-32,babel-p9-28,babel-m5-32}"
INFER_EXCLUDE="${INFER_EXCLUDE:-babel-p9-32,babel-p9-28,babel-m5-32,babel-o5-24,babel-q5-16,babel-q5-20,babel-n5-32,babel-o5-16,babel-n5-28,babel-q5-32,babel-s5-24,babel-q5-24,babel-o5-28,babel-p5-20,babel-p5-24,babel-o9-24,babel-o9-28,babel-q9-32,babel-t5-28,babel-v9-28}"
# Backfill windows shorter than this end in TIMEOUT, which Slurm does not requeue (40k lesson).
DECODE_TIME_MIN="${DECODE_TIME_MIN:-08:00:00}"
# Decoder flags beyond the frozen 40k set. No commas: they travel through --export.
EXTRA_DECODER_ARGS="${EXTRA_DECODER_ARGS:---targeted-sampler-context source-only --future-source-window-mode sentence-anchor --sentence-end-completion --sentence-end-boundary-mode conservative --sentence-end-punctuation match-source --targeted-prompt-version future_set_v3_suffix_icl}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_v3json_35k.sh plan [RUN_TAG]
  bash scripts/submit_v3json_35k.sh submit [RUN_TAG]
  bash scripts/submit_v3json_35k.sh status RUN_TAG

`plan` prints and validates the experiment without submitting jobs. `submit`
queues decode, filtering, 12.5K training, inference, BLEU, latency, and COMET.
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
  local busy=0
  if command -v squeue >/dev/null; then
    busy=$(squeue -u haolingp -h -t RUNNING,PENDING -o "%b" 2>/dev/null | { grep -o "gpu[:a-zA-Z0-9]*:[0-9]*" || true; } | awk -F: '{s+=$NF} END {print s+0}')
  fi
  (( busy + 2 * DECODE_CONCURRENCY <= 24 )) || \
    die "account already has ${busy} GPUs queued or running; decode would exceed the 24-GPU cap"
  (( NUM_CONCURRENT_CASES > 0 && NUM_CONCURRENT_CASES <= 16 )) || \
    die "NUM_CONCURRENT_CASES must be in [1, 16] to match the sampler servers"
  (( POST_CONCURRENCY > 0 && POST_CONCURRENCY <= NUM_POST_SHARDS )) || \
    die "POST_CONCURRENCY must be in [1, NUM_POST_SHARDS]"
  (( POST_CONCURRENCY <= 24 )) || die "post-processing exceeds the 24-GPU limit"
  (( TARGETED_NUM_FUTURES > 0 && TARGETED_NUM_FUTURES % 2 == 0 )) || \
    die "TARGETED_NUM_FUTURES must be a positive even number"
  (( TRAIN_SAMPLE_N >= 0 )) || die "TRAIN_SAMPLE_N cannot be negative"
  [[ "${EXTRA_DECODER_ARGS}" != *,* ]] || die "EXTRA_DECODER_ARGS must not contain commas (Slurm --export)"
  [[ "${EXTRA_DECODER_ARGS}" == *"--targeted-prompt-version ${PROMPT_VERSION}"* ]] || \
    die "EXTRA_DECODER_ARGS must select ${PROMPT_VERSION}"
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
Sampler prompt        : ${PROMPT_VERSION} (up to $((TARGETED_NUM_FUTURES / 2)) plausible + $((TARGETED_NUM_FUTURES / 2)) contrastive per model, ${SAMPLER_OUTPUT})
Extra decoder flags   : ${EXTRA_DECODER_ARGS}
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
  local git_dirty
  git_dirty=$(git -C "${REPO}" status --porcelain --untracked-files=no | wc -l | tr -d " ")
  (( git_dirty == 0 )) || die "BABEL checkout has ${git_dirty} uncommitted tracked change(s); commit or stash before submitting"
  cat >"${manifest}" <<EOF
run_tag=${run_tag}
created=$(date --iso-8601=seconds)
git_commit=${git_commit}
git_branch=${git_branch}
git_dirty_tracked_files=${git_dirty}
prompt_version=${PROMPT_VERSION}
sampler_output=${SAMPLER_OUTPUT}
extra_decoder_args=${EXTRA_DECODER_ARGS}
decoder_settings_source=per-utterance JSON decoder_settings (authoritative; extra_decoder_args is the request)
decode_exclude=${DECODE_EXCLUDE}
decode_time_min=${DECODE_TIME_MIN}
infer_exclude=${INFER_EXCLUDE}
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
train_sample_target=${TRAIN_SAMPLE_N}
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
    --job-name="v3json35k_decode" \
    --array="0-$((NUM_DECODE_TASKS - 1))%${DECODE_CONCURRENCY}" \
    --time-min="${DECODE_TIME_MIN}" --exclude="${DECODE_EXCLUDE}" \
    --output="${log_dir}/decode_%A_%a.out" --error="${log_dir}/decode_%A_%a.err" \
    --export="ALL,INPUT_TSV=${INPUT_TSV},OUTPUT_ROOT=${decode_root},TOTAL_ROWS=${TOTAL_ROWS},ROW_OFFSET=0,NUM_TASKS=${NUM_DECODE_TASKS},NUM_CONCURRENT_CASES=${NUM_CONCURRENT_CASES},AMBIGUITY_TUNING_TASK_ID=0,AMBIGUITY_TUNING_CONCURRENCY=${NUM_CONCURRENT_CASES},TARGETED_NUM_FUTURES=${TARGETED_NUM_FUTURES},MIN_VOTERS_RATIO=${MIN_VOTERS_RATIO},FUTURE_SRC_WINDOW=1,PROMPT_VERSION=${PROMPT_VERSION},EXTRA_DECODER_ARGS=${EXTRA_DECODER_ARGS},QWEN38_MODEL=/data/user_data/haolingp/models/Qwen3.8-27B-FP8,GEMMA_MODEL=/data/user_data/haolingp/models/gemma-4-E2B-it,QWEN36_MODEL=/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8" \
    "${FS}/run_ambiguity_q38_gemma_q36_preempt.sbatch")
  echo "decode=${decode_jid}" | tee -a "${manifest}"

  # Gate 1: exactly TOTAL_ROWS distinct, loadable utterance JSONs decoded with PROMPT_VERSION,
  # before any post-processing (40k lesson: verify the count, never assume it).
  local decode_gate_jid
  decode_gate_jid=$(sbatch --parsable --dependency="afterok:${decode_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --job-name="v3json35k_decode_gate" \
    --output="${log_dir}/decode_gate_%j.out" --error="${log_dir}/decode_gate_%j.err" \
    --export="ALL,DECODE_ROOT=${decode_root},TOTAL_ROWS=${TOTAL_ROWS},PROMPT_VERSION=${PROMPT_VERSION},MANIFEST=${manifest}" \
    "${REPO}/scripts/verify_decode_root.sbatch")
  echo "decode_gate=${decode_gate_jid}" | tee -a "${manifest}"

  local prep_jid
  prep_jid=$(sbatch --parsable --dependency="afterok:${decode_gate_jid}" \
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

  # Gate 2: the training manifest must hold exactly TRAIN_SAMPLE_N rows (count-matched control);
  # the convert step silently keeps all survivors when the pool is smaller (40k lesson).
  local train_gate_jid
  train_gate_jid=$(sbatch --parsable --dependency="afterok:${convert_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue --cpus-per-task=1 --mem=2G --time=00:10:00 \
    --job-name="v3json35k_train_gate" \
    --output="${log_dir}/train_gate_%j.out" --error="${log_dir}/train_gate_%j.err" \
    --wrap="set -euo pipefail; D=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests; T=\$D/train_s_zh-consensus-${run_tag}.jsonl; F=\$D/train_s_zh-consensus-${run_tag}_full.jsonl; rows=\$(wc -l < \$T); pool=\$( [ -f \$F ] && wc -l < \$F || echo \$rows ); printf 'survivor_pool=%s\ntrain_rows_actual=%s\ntraining_manifest=%s\n' \$pool \$rows \$T >> ${manifest}; if [ ${TRAIN_SAMPLE_N} -gt 0 ] && [ \$rows -ne ${TRAIN_SAMPLE_N} ]; then echo \"[GATE] training manifest has \$rows rows, expected ${TRAIN_SAMPLE_N} (pool \$pool)\" >&2; echo 'train_gate=FAILED_count_mismatch' >> ${manifest}; exit 1; fi; echo \"[GATE] \$rows rows from a pool of \$pool\"")
  echo "train_gate=${train_gate_jid}" | tee -a "${manifest}"

  local train_jid
  train_jid=$(sbatch --parsable --dependency="afterok:${train_gate_jid}" \
    --partition=preempt --qos=preempt_qos --requeue \
    --output="${log_dir}/train_%A_%a.out" --error="${log_dir}/train_%A_%a.err" \
    --export="ALL,VARIANT_TAG=${run_tag}" "${REPO}/scripts/train/train_consensus_s.sh")
  echo "train=${train_jid}" | tee -a "${manifest}"

  local eval_launcher_jid
  eval_launcher_jid=$(sbatch --parsable --dependency="afterok:${train_jid}" \
    --partition=preempt --qos=preempt_cpu_qos --requeue \
    --output="${log_dir}/eval_launcher_%j.out" --error="${log_dir}/eval_launcher_%j.err" \
    --export="ALL,EXP=${exp},CHILD_PARTITION=preempt,CHILD_GPU_QOS=preempt_qos,CHILD_EXCLUDE_ENCODED=${INFER_EXCLUDE//,/;},CKPTS_FILE=${ckpts_file},RUN_SIMULTST=1,CKPTS_SIMULTST_FILE=${run_dir}/ckpts_simultst.txt,PIPELINE_MANIFEST=${manifest}" \
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
    print_plan "${2:-v3json-boundary-q38-gemma-q36-strict-35k-$(date +%Y%m%d)}"
    ;;
  submit)
    submit_run "${2:-v3json-boundary-q38-gemma-q36-strict-35k-$(date +%Y%m%d-%H%M%S)}"
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
