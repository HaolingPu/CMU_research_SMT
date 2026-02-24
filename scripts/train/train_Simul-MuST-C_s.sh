#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=general
#SBATCH --exclude=babel-p9-32
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

WANDB_API_KEY=$(cat /home/siqiouya/.keys/wandb)
HF_TOKEN=$(cat /home/siqiouya/.keys/huggingface)

apptainer exec \
  --nv \
  --env "MODELSCOPE_CACHE=/home/siqiouya/.cache/modelscope/" \
  --env "MEGATRON_LM_PATH=/home/siqiouya/code/Megatron-LM/" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "NCCL_IB_DISABLE=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY}" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1 \
  bash -c '
export train_dataset=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_zh-Simul-MuST-C-fixed_v2_origin.jsonl

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ENABLE_AUDIO_OUTPUT=False \
megatron sft \
    --load /data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/ \
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
    --save /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4 \
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
    --wandb_exp_name gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4

BASE_DIR=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4
LATEST_CKPT=$(ls -td "$BASE_DIR"/v*-* 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Warning: No checkpoint found for gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4"
    exit 1
fi

echo "Exporting checkpoint: $LATEST_CKPT"

swift export \
    --mcore_adapters "${LATEST_CKPT}/" \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir "${LATEST_CKPT}-hf/"
'
