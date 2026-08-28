WANDB_API_KEY=$(cat /home/siqiouya/.keys/wandb)
HF_TOKEN=$(cat /home/siqiouya/.keys/huggingface)

apptainer shell \
  --nv \
  --env "MODELSCOPE_CACHE=/home/siqiouya/.cache/modelscope/" \
  --env "MEGATRON_LM_PATH=/home/siqiouya/code/Megatron-LM/" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "NCCL_IB_DISABLE=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY}" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1