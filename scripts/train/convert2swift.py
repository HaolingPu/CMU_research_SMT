import os
import json
import io
import soundfile as sf
import pandas as pd

from tqdm import tqdm

tsv_path = '/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv'
orig_manifest = pd.read_csv(tsv_path, sep='\t')

# EAST
# manifest_root = "/data/group_data/li_lab/haolingp/gigaspeech/final_jsonl_dataset" 
# audio_clips_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_zh_EAST/"
# output_filename = "train_xl_case_robust_asr-filtered_zh-EAST.jsonl"

# Refined EAST
manifest_root = "/data/group_data/li_lab/haolingp/gigaspeech/final_jsonl_refined_east/"
audio_clips_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_zh_refined-EAST/"
output_filename = "train_xl_case_robust_asr-filtered_zh-refined-EAST.jsonl"

output_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
os.makedirs(audio_clips_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)
audio_ids = os.listdir(manifest_root)
latency2instances = {}

for latency in ['low', 'medium', 'high']:
    latency_traj = []
    instances = []

    for audio_id in tqdm(audio_ids, desc="Processing audio ids"):
        latency_traj_path = os.path.join(manifest_root, audio_id, f"{latency}_latency.jsonl")
        if not os.path.exists(latency_traj_path):
            continue

        with open(latency_traj_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                latency_traj.append(data)
    
    pbar = tqdm(latency_traj, desc="Processing {} latency, skipped 0 instances".format(latency))
    n_skip = 0
    for traj in pbar:
        #从原始 manifest 里用 utt_id 找到原音频片段
        audio_path, start, duration = orig_manifest[orig_manifest['id'] == traj['utt_id']].iloc[0]['audio'].split(':')
        wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
        #读出这段 wav，采样率要求 16k
        #按 15360 sample 切块，也就是 15360 / 16000 = 0.96s
        assert sr == 16000
        stepsize = 15360

        audio_id, segment_id = traj['utt_id'].split('_')

        audio_clips_dir = os.path.join(audio_clips_root, latency, audio_id, segment_id)
        os.makedirs(audio_clips_dir, exist_ok=True)
        audio_clip_paths = []

        # 每个 chunk 写成一个单独 wav 文件
        for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            wav_clip = wav[i : i + stepsize]
            clip_path = os.path.join(audio_clips_dir, f"{idx}.wav")
            sf.write(clip_path, wav_clip, sr)
            audio_clip_paths.append(clip_path)
        #如果 chunk 数和 traj['target'] 数对不上，就跳过这个样本
        if len(audio_clip_paths) != len(traj['target']):
            n_skip += 1
            pbar.set_description(f"Processing {latency} latency, skipped {n_skip} instances")
            continue

        # 然后拼成一条多轮训练样本：
        # system: “你是同传，按某种 latency 做英译中”
        # user: <audio>
        # assistant: 当前 chunk 对应目标文本
        n_chunk = min(len(audio_clip_paths), len(traj['target']))

        messages = [
            {"role": "system", "content": f"You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text with {traj['latency']} latency."},
        ]
        for target in traj['target'][:n_chunk]:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": target})
        instance = {
            "messages": messages,
            "audios": audio_clip_paths[:n_chunk],
        }
        instances.append(instance)
    latency2instances[latency] = instances

with open(os.path.join(output_root, output_filename), 'w') as f:
    for latency in ['low', 'medium', 'high']:
        for instance in latency2instances[latency]:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")

# #{
#   "messages": [
#     {"role": "system", "content": "...with low/medium/high latency."},
#     {"role": "user", "content": "<audio>"},
#     {"role": "assistant", "content": "..."},
#     ...
#   ],
#   "audios": ["/path/0.wav", "/path/1.wav", ...]
# }