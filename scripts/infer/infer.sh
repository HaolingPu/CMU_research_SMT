source /home/siqiouya/miniconda3/bin/activate omni_inference

# EAST
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
uv run simuleval \
    --agent /home/siqiouya/code/CMU_research_SMT/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size 960 \
    --prompt-type EAST \
    --EAST-latency-type low \
    --output /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-refined-EAST-s_origin-bsz4/v1-20260213-234837-hf/evaluation/acl_6060/en-zh/seg960_low \
    --max-new-tokens 30 \
    --max-cache-chunks 60 \
    --keep-cache-chunks 30 \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec 0 \
    --source /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.source \
    --target /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.target.zh \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --model-name /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-refined-EAST-s_origin-bsz4/v1-20260213-234837-hf \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh

# Standard
source /home/siqiouya/miniconda3/bin/activate omni_inference
MODEL_PATH=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
uv run simuleval \
    --agent /home/siqiouya/code/CMU_research_SMT/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size 960 \
    --prompt-type Standard \
    --output ${MODEL_PATH}/evaluation/acl_6060/en-zh/seg960 \
    --max-new-tokens 30 \
    --max-cache-chunks 60 \
    --keep-cache-chunks 30 \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec 2.0 \
    --source /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.source \
    --target /data/group_data/li_lab/siqiouya/datasets/acl_6060/dev.target.zh \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --model-name ${MODEL_PATH} \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh

export MWERSEGMENTER_ROOT=/home/siqiouya/download/mwerSegmenter
export PYTHONPATH=/home/siqiouya/code/FBK-fairseq
conda activate fbk

streamLAAL \
    --simuleval-instances ${MODEL_PATH}/evaluation/acl_6060/en-zh/seg960/instances.log  \
    --source /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
    --reference /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt \
    --audio-yaml /data/user_data/siqiouya/datasets/acl_6060/dev.yaml \
    --sacrebleu-tokenizer zh \
    --latency-unit char