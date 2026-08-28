#!/usr/bin/env bash
# Submit the win3 downstream chain (segale -> metricx QE -> period+length fix ->
# conv2swift -> train), each afterok-gated on the previous. Decode (8690947) is
# the root. Infer/eval is submitted separately after train (HF path is timestamped).
set -e

DECODE_JID="${1:-$(cat /tmp/win3_decode_jid.txt)}"
FS=/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/future_sampling
DECODE_OUT=/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k_qwenasr_sentsplit_win3
SEGROOT=${DECODE_OUT}-segale-p24
SHARDS_ROOT=${SEGROOT}/shards
METRICX_RUN_DIR=${SEGROOT}/metricx-aligned
QE_FILTERED=${SEGROOT}/qe3-aligned-max
FIXED=${SEGROOT}/qe3-aligned-max-fixed
VARIANT_TAG=INTERSECT-win3-fixed

echo "decode root job: ${DECODE_JID}"

# 2) SEGALE prep
J2=$(sbatch --parsable --dependency=afterok:${DECODE_JID} \
  --export=ALL,CONSENSUS_ROOT=${DECODE_OUT},OUT_ROOT=${SEGROOT} \
  ${FS}/scripts/segale/run_segale_prep_win3.sbatch)
echo "segale-prep : ${J2}"

# 3) SEGALE align (array 0-7)
J3=$(sbatch --parsable --dependency=afterok:${J2} \
  --export=ALL,SHARDS_ROOT=${SHARDS_ROOT},TASK_LANG=zh \
  ${FS}/scripts/segale/run_segale_align_8gpu.sbatch)
echo "segale-align: ${J3}"

# 4) MetricX prep
J4=$(sbatch --parsable --dependency=afterok:${J3} \
  --export=ALL,EXPERIMENT_DIR=${SEGROOT},METRICX_RUN_DIR=${METRICX_RUN_DIR} \
  ${FS}/scripts/minp/run_metricx_qe_prepare.sbatch)
echo "metricx-prep: ${J4}"

# 5) MetricX QE (array 0-7)
J5=$(sbatch --parsable --dependency=afterok:${J4} \
  --export=ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR} \
  ${FS}/scripts/minp/run_metricx_qe_8gpu.sbatch)
echo "metricx-qe  : ${J5}"

# 6) MetricX finalize (merge + filter)
J6=$(sbatch --parsable --dependency=afterok:${J5} \
  --export=ALL,METRICX_RUN_DIR=${METRICX_RUN_DIR},EXPERIMENT_DIR=${SEGROOT},FILTERED_OUTPUT_DIR=${QE_FILTERED},QE_THRESHOLD=3.0 \
  ${FS}/scripts/minp/run_metricx_qe_finalize.sbatch)
echo "metricx-fin : ${J6}"

# 7) period + length fix
J7=$(sbatch --parsable --dependency=afterok:${J6} \
  --export=ALL,IN_DIR=${QE_FILTERED},OUT_DIR=${FIXED} \
  ${FS}/scripts/minp/run_fix_win3.sbatch)
echo "fix         : ${J7}"

# 8) conv2swift (-> train_s_zh-consensus-INTERSECT-win3-fixed.jsonl, downsample<=12500)
J8=$(sbatch --parsable --dependency=afterok:${J7} \
  --export=ALL,MANIFEST_ROOT=${FIXED},VARIANT_TAG=${VARIANT_TAG},SAMPLE_N=12500 \
  /home/haolingp/CMU_research_SMT/scripts/train/run_convert2swift_consensus.sbatch)
echo "conv2swift  : ${J8}"

# 9) train
J9=$(sbatch --parsable --dependency=afterok:${J8} \
  --export=ALL,VARIANT_TAG=${VARIANT_TAG} \
  /home/haolingp/CMU_research_SMT/scripts/train/train_consensus_s.sh)
echo "train       : ${J9}"

echo
echo "CHAIN: decode=${DECODE_JID} -> segprep=${J2} -> segalign=${J3} -> mprep=${J4} -> mqe=${J5} -> mfin=${J6} -> fix=${J7} -> conv=${J8} -> train=${J9}"
echo "After train, infer with: sbatch infer_slurm.sh <ckpt>/v0-<ts>-hf/ Standard ; then add to ckpts.txt + eval."
echo "${J9}" > /tmp/win3_train_jid.txt
