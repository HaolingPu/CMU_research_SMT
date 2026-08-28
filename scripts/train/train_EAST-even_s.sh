#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=general
#SBATCH --exclude=babel-p9-32,babel-p5-24,babel-t9-16
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

# EAST-even zh baseline (12.5k) -- adapted to haolingp env (was mentor siqiouya's).
# Trains directly from the existing converted swift manifest (12.5k examples);
# audio clips referenced inside are li_lab-group-readable. Megatron hyperparams
# mirror train_LA_s.sh (the proven haolingp recipe), which matches the original.
# Produces ckpt at /data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}

WANDB_API_KEY=$(cat /home/haolingp/.keys/wandb 2>/dev/null || echo "")
HF_TOKEN=$(cat /home/haolingp/.keys/huggingface 2>/dev/null || echo "")
WANDB_MODE=${WANDB_API_KEY:+online}
WANDB_MODE=${WANDB_MODE:-disabled}

EXP_NAME=gigaspeech-zh-EAST-even-s-bsz4
TRAIN_DATASET=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_zh-EAST-even12500.jsonl

apptainer exec \
  --nv \
  --env "MODELSCOPE_CACHE=/home/haolingp/.cache/modelscope/" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "NCCL_IB_DISABLE=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY}" \
  --env "WANDB_MODE=${WANDB_MODE}" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "EXP_NAME=${EXP_NAME}" \
  --env "train_dataset=${TRAIN_DATASET}" \
  --env "SSL_CERT_FILE=/home/haolingp/CMU_research_SMT/scripts/train/cacert.pem" \
  docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1 \
  bash -c '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ENABLE_AUDIO_OUTPUT=False \
megatron sft \
    --load /data/user_data/haolingp/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/ \
    --dataset ${train_dataset} \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --vit_gradient_checkpointing false \
    --packing true \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --weight_decay 0.01 \
    --clip_grad 1.0 \
    --max_epochs 1 \
    --save /data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME} \
    --log_interval 10 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_project gigaspeech_zh \
    --wandb_exp_name ${EXP_NAME}

BASE_DIR=/data/user_data/haolingp/ckpts/infinisst-omni/${EXP_NAME}
LATEST_CKPT=$(ls -td "$BASE_DIR"/v*-* 2>/dev/null | grep -v -- "-hf$" | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Warning: No checkpoint found for ${EXP_NAME}"
    exit 1
fi

echo "Exporting checkpoint: $LATEST_CKPT"

swift export \
    --mcore_adapters "${LATEST_CKPT}/" \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir "${LATEST_CKPT}-hf/"
'
