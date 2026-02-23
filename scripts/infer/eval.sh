export MWERSEGMENTER_ROOT=/home/siqiouya/download/mwerSegmenter
export PYTHONPATH=/home/siqiouya/code/FBK-fairseq
conda activate fbk

streamLAAL \
    --simuleval-instances /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-refined-EAST-mult-s_origin-bsz4/v0-20260214-025648-hf/evaluation/acl_6060/en-zh/seg960/instances.log  \
    --source /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
    --reference /data/user_data/siqiouya/datasets/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt \
    --audio-yaml /data/user_data/siqiouya/datasets/acl_6060/dev.yaml \
    --sacrebleu-tokenizer zh \
    --latency-unit char