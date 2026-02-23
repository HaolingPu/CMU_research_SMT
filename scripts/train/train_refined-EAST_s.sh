export train_dataset=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_zh-refined-EAST_origin.jsonl

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
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
    --save /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-refined-EAST-s_origin-bsz4 \
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
    --wandb_exp_name gigaspeech-zh-refined-EAST-s_origin-bsz4
    
BASE_DIR=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-refined-EAST-s_origin-bsz4
LATEST_CKPT=$(ls -td "$BASE_DIR"/v*-* 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Warning: No checkpoint found for gigaspeech-zh-refined-EAST-s_origin-bsz4"
    continue
fi

echo "Exporting checkpoint: $LATEST_CKPT"

swift export \
    --mcore_adapters "${LATEST_CKPT}/" \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir "${LATEST_CKPT}-hf/"

# hf upload owaski/gigaspeech-zh-refined-EAST-s_origin-bsz4 ${LATEST_CKPT}-hf