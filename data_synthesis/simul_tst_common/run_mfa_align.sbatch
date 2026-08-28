#!/bin/bash
#SBATCH --job-name=simultst-mfa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/haolingp/data_synthesis/simul_tst_common/slurm_mfa_%j.log

set -e

BUILD=/data/user_data/haolingp/data_synthesis/simul_tst_common
source /home/haolingp/miniconda3/etc/profile.d/conda.sh
conda activate SMT

mfa align "$BUILD/mfa_corpus" english_us_arpa english_us_arpa "$BUILD/mfa_aligned" \
    --output_format json \
    --temporary_directory "$BUILD/mfa_tmp" \
    --beam 100 --retry_beam 400 \
    -j 8 --clean

echo "MFA DONE $(date)"
