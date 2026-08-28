import os
import json
import io
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm

tsv_path = '/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv'
orig_manifest = pd.read_csv(tsv_path, sep='\t')

# Simul-MuST-C
manifest_root = "/data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/final_jsonl_salami/"
audio_clips_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_zh_Simul-MuST-C_fixed_v2/"
output_filename = "train_xl_case_robust_asr-filtered_zh-Simul-MuST-C_fixed_v2.jsonl"

output_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
os.makedirs(audio_clips_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)

audio_ids = os.listdir(manifest_root)

latency = 'offline'
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

    audio_path, start, duration = orig_manifest[orig_manifest['id'] == traj['utt_id']].iloc[0]['audio'].split(':')
    wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))

    assert sr == 16000
    multiplier = np.random.randint(1, 12)
    stepsize = 15360 * multiplier

    audio_id, segment_id = traj['utt_id'].split('_')

    audio_clips_dir = os.path.join(audio_clips_root, audio_id, segment_id, f"multiplier_{multiplier}")
    os.makedirs(audio_clips_dir, exist_ok=True)
    audio_clip_paths = []

    for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
        wav_clip = wav[i : i + stepsize]
        clip_path = os.path.join(audio_clips_dir, f"{idx}.wav")
        sf.write(clip_path, wav_clip, sr)
        audio_clip_paths.append(clip_path)

    targets = []
    for i in range(0, len(traj['target']), multiplier):
        targets.append("".join(traj['target'][i:i+multiplier])) # no space for Chinese

    if len(audio_clip_paths) != len(targets):
        n_skip += 1
        pbar.set_description(f"Processing, skipped {n_skip} instances")
        continue

    messages = [
        {"role": "system", "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."},
    ]
    for target in targets:
        messages.append({"role": "user", "content": "<audio>"})
        messages.append({"role": "assistant", "content": target})
    instance = {
        "messages": messages,
        "audios": audio_clip_paths,
        "multiplier": multiplier,
    }
    instances.append(instance)

with open(os.path.join(output_root, output_filename), 'w') as f:
    for instance in instances:
        f.write(json.dumps(instance, ensure_ascii=False) + "\n")