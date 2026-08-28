#!/usr/bin/env bash
# End-to-end SEGALE sub-sentence QE pipeline for SALAMI ja (post fix_llm_raw).
# Submits: align(24) -> merge -> qe_prep(24-shard split) -> metricx_qe(24) -> finalize(t=4, all-pass).
set -e

BASE=/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja
CODE=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech
SAL=${CODE}/salami
SEGCODE=${CODE}/future_sampling/scripts/segale

ROOT=${BASE}/segale_pipeline
CONS_ROOT=${ROOT}/consensus_format
SHARDS_ROOT=${ROOT}/shards
ALIGNED_MERGED=${ROOT}/aligned_all.jsonl
METRICX_RUN_DIR=${ROOT}/metricx_aligned
FILTERED=${ROOT}/metricx_filtered_t4.0.jsonl
REPORT=${ROOT}/finalize_report.jsonl

NUM_SHARDS=24
TSV=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv

mkdir -p "${ROOT}" "${BASE}/slurm_logs"

echo "[1/6] Build consensus-format JSONs from llm_output_merged..."
rm -rf "${CONS_ROOT}"
python "${SAL}/salami_to_consensus_format.py" \
  --merged-dir   "${BASE}/llm_output_merged" \
  --manifest-tsv "${TSV}" \
  --stream-dir   "${BASE}/streaming_salami_dataset" \
  --out-root     "${CONS_ROOT}" \
  --sys-id       "salami_ja" \
  --latency      "offline" \
  --target-lang  "ja"

N_CONS=$(find "${CONS_ROOT}" -name '*.json' | wc -l)
echo "consensus jsons: ${N_CONS}"

echo
echo "[2/6] prepare_segale_shards (num-shards=${NUM_SHARDS})..."
rm -rf "${SHARDS_ROOT}"
python "${SEGCODE}/prepare_segale_shards.py" \
  --consensus-root "${CONS_ROOT}" \
  --out-root       "${ROOT}" \
  --num-docs       "${N_CONS}" \
  --num-shards     "${NUM_SHARDS}" \
  --sys-id         "salami_ja"

echo
echo "[3/6] Submit SEGALE align (24-shard preempt)..."
ALIGN_JOB=$(sbatch --parsable \
  --export="ALL,SHARDS_ROOT=${SHARDS_ROOT},TASK_LANG=ja" \
  "${SAL}/run_segale_align_24_preempt.sbatch")
echo "[align] ${ALIGN_JOB}"

echo
echo "[4/6] Submit merge (afterok:${ALIGN_JOB})..."
MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:${ALIGN_JOB}" \
  --time=00:20:00 \
  --partition=preempt --qos=preempt_qos --requeue \
  --cpus-per-task=2 --mem=8G \
  --job-name=salami_seg_merge \
  -o ${BASE}/slurm_logs/seg_merge_%j.out \
  -e ${BASE}/slurm_logs/seg_merge_%j.err \
  --wrap "source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/haolingp/conda_envs/segale && python ${SEGCODE}/merge_aligned_shards.py --shards-root ${SHARDS_ROOT} --output ${ALIGNED_MERGED} --num-shards ${NUM_SHARDS}")
echo "[merge] ${MERGE_JOB}"

echo
echo "[5/6] Submit QE prepare + split (afterok:${MERGE_JOB})..."
QE_PREP_JOB=$(sbatch --parsable \
  --dependency="afterok:${MERGE_JOB}" \
  --time=00:30:00 \
  --partition=preempt --qos=preempt_qos --requeue \
  --cpus-per-task=4 --mem=16G \
  --job-name=salami_seg_qeprep \
  -o ${BASE}/slurm_logs/seg_qeprep_%j.out \
  -e ${BASE}/slurm_logs/seg_qeprep_%j.err \
  --wrap "set -e; source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate metricx && export PYTHONNOUSERSITE=1 && python ${CODE}/future_sampling/convert_metricx_consensus_aligned.py --aligned ${ALIGNED_MERGED} --consensus-root ${CONS_ROOT} --output ${METRICX_RUN_DIR}/metricx_input.jsonl --require-stream-json && rm -rf ${METRICX_RUN_DIR}/metricx_shards && mkdir -p ${METRICX_RUN_DIR}/metricx_shards && split -d -a 2 -n l/${NUM_SHARDS} ${METRICX_RUN_DIR}/metricx_input.jsonl ${METRICX_RUN_DIR}/metricx_shards/input_ && echo done lines=\$(wc -l < ${METRICX_RUN_DIR}/metricx_input.jsonl)")
echo "[qe_prep] ${QE_PREP_JOB}"

echo
echo "[6/6] Submit MetricX QE 24-shard (afterok:${QE_PREP_JOB}) + finalize (afterok:metricxqe)..."
QE_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_PREP_JOB}" \
  --export="ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR}" \
  "${SAL}/run_metricx_qe_24_preempt.sbatch")
echo "[metricx_qe] ${QE_JOB}"

FIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${QE_JOB}" \
  --time=00:30:00 \
  --partition=preempt --qos=preempt_qos --requeue \
  --cpus-per-task=2 --mem=12G \
  --job-name=salami_seg_fin \
  -o ${BASE}/slurm_logs/seg_fin_%j.out \
  -e ${BASE}/slurm_logs/seg_fin_%j.err \
  --wrap "set -e; source /home/haolingp/miniconda3/etc/profile.d/conda.sh && conda activate metricx && export PYTHONNOUSERSITE=1 && cat ${METRICX_RUN_DIR}/metricx_shards/output_*.jsonl > ${METRICX_RUN_DIR}/metricx_output.jsonl && echo merged lines=\$(wc -l < ${METRICX_RUN_DIR}/metricx_output.jsonl) && python ${CODE}/east/finalize_segale_qe_east.py --metricx-output ${METRICX_RUN_DIR}/metricx_output.jsonl --consensus-format-root ${CONS_ROOT} --filtered-output ${FILTERED} --threshold 4.0 --report ${REPORT}")
echo "[finalize] ${FIN_JOB}"

echo
echo "Pipeline submitted (SALAMI ja sub-sentence QE):"
echo "  prepare        : done (consensus=${N_CONS}, shards=${NUM_SHARDS})"
echo "  align(24)      : ${ALIGN_JOB}"
echo "  merge          : ${MERGE_JOB}   (afterok:${ALIGN_JOB})"
echo "  qe_prepare     : ${QE_PREP_JOB} (afterok:${MERGE_JOB})"
echo "  metricx_qe(24) : ${QE_JOB}      (afterok:${QE_PREP_JOB})"
echo "  finalize t=4   : ${FIN_JOB}     (afterok:${QE_JOB})"
echo
echo "Outputs (when done):"
echo "  aligned merged : ${ALIGNED_MERGED}"
echo "  metricx output : ${METRICX_RUN_DIR}/metricx_output.jsonl"
echo "  filtered t=4   : ${FILTERED}"
echo "  report         : ${REPORT}"
