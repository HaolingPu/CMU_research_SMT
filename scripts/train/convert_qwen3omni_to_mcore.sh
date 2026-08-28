#!/usr/bin/env bash
#SBATCH --job-name=qwen3omni_mcore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=general
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

set -e

# One-shot: HF download (Qwen3-Omni-30B-A3B-Instruct) -> mcore conversion.
# Output: /data/user_data/haolingp/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/

HF_DIR=/data/user_data/haolingp/models/Qwen3-Omni-30B-A3B-Instruct
MCORE_DIR=/data/user_data/haolingp/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore
HF_TOKEN=$(cat ~/.keys/huggingface 2>/dev/null || echo "")

mkdir -p "$(dirname "${MCORE_DIR}")"

# Step 1: download HF format (resumable)
if [ ! -f "${HF_DIR}/config.json" ]; then
    echo "[DOWNLOAD] HF model -> ${HF_DIR}"
    HF_TOKEN="${HF_TOKEN}" huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --local-dir "${HF_DIR}" \
        --local-dir-use-symlinks False
else
    echo "[SKIP] HF model already at ${HF_DIR}"
fi

# Step 2: HF -> mcore via swift docker
apptainer exec \
  --nv \
  --env "MODELSCOPE_CACHE=/home/haolingp/.cache/modelscope/" \
  --env "SSL_CERT_FILE=/home/haolingp/CMU_research_SMT/scripts/train/cacert.pem" \
  docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1 \
  bash -c "
    echo '=== Megatron status ==='
    python -c 'import megatron; print(megatron.__file__)' 2>&1 || true
    python -c 'import megatron.training' 2>&1 || true
    python -c 'import megatron.legacy.model' 2>&1 || true
    echo '=== conversion ==='
    swift export \
        --model ${HF_DIR} \
        --to_mcore true \
        --torch_dtype bfloat16 \
        --device_map auto \
        --output_dir ${MCORE_DIR}
"

echo "===== DONE ====="
ls -la "${MCORE_DIR}/" | head -20
