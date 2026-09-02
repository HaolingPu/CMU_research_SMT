#!/bin/bash
#SBATCH --job-name=trajectory_viewer
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/haolingp/slurm_logs/trajectory_viewer_%j.out

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/haolingp/CMU_research_SMT}"
BUNDLE_DIR="${BUNDLE_DIR:?Set BUNDLE_DIR to the generated review bundle}"
PORT="${PORT:-8765}"

echo "node=$(hostname)"
echo "port=${PORT}"
echo "bundle=${BUNDLE_DIR}"

python3 "${CODE_ROOT}/data_synthesis/tools/trajectory_viewer/server.py" \
  "${BUNDLE_DIR}" --host 0.0.0.0 --port "${PORT}"
