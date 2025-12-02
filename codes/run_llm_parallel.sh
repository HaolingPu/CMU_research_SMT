#!/bin/bash

NGPU=8   # You requested 8 GPUs
DATA_ROOT="/data/user_data/haolingp/outputs/llm_segmentation_json/en000"

# Count total parquets
TOTAL=$(ls /data/group_data/li_lab/siqiouya/datasets/yodas-granary/data/en000 | wc -l)
PER=$(( (TOTAL + NGPU - 1) / NGPU ))

echo "TOTAL=$TOTAL, each GPU handles ~$PER parquets"

for ((i=0; i<NGPU; i++)); do
    START=$(( i * PER ))
    END=$(( (i+1)*PER - 1 ))

    echo "GPU $i handles parquet $START → $END"

    CUDA_VISIBLE_DEVICES=$i \
    python /data/user_data/haolingp/codes/llm_output.py \
        --lang en000 \
        --parquet-start $START \
        --parquet-end $END \
        > logs/llm_gpu${i}.out 2>&1 &
done

wait
echo "=== LLM segmentation done on all GPUs ==="
