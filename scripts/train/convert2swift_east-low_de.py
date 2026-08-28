"""
en->de EAST-low recipe: read ONLY low_latency.jsonl (~12.5k post-QE rows),
multiplier uniformly random in [1, 12] per instance.

Outputs:
  audio: /data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_de_EAST-low/
  manifest: /data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_de-EAST-low_origin.jsonl

de: space-separated when joining target chunks (word-level).
"""
import os
import json
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm

tsv_path = '/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv'
orig_manifest = pd.read_csv(tsv_path, sep='\t')

manifest_root    = "/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_de/segale_qe/final_jsonl_east"
audio_clips_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_de_EAST-low/"
output_filename  = "train_s_de-EAST-low_origin.jsonl"
output_root      = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
os.makedirs(audio_clips_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)
audio_ids = os.listdir(manifest_root)

latency = "low"
latency_traj = []
for audio_id in tqdm(audio_ids, desc=f"Loading {latency} latency manifests"):
    p = os.path.join(manifest_root, audio_id, f"{latency}_latency.jsonl")
    if not os.path.exists(p):
        continue
    with open(p, 'r') as f:
        for line in f:
            latency_traj.append(json.loads(line))

print(f"loaded {len(latency_traj)} {latency}-latency trajectories")

instances = []
pbar = tqdm(latency_traj, desc=f"Processing {latency} de-low, skipped 0")
n_skip = 0
for traj in pbar:
    rows = orig_manifest[orig_manifest['id'] == traj['utt_id']]
    if len(rows) == 0:
        n_skip += 1
        pbar.set_description(f"Processing {latency} de-low, skipped {n_skip}")
        continue
    audio_path, start, duration = rows.iloc[0]['audio'].split(':')
    wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
    assert sr == 16000

    multiplier = int(np.random.randint(1, 13))   # 1..12 inclusive
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
        targets.append(" ".join(traj['target'][i:i+multiplier]))   # de: word-level, space-joined

    if len(audio_clip_paths) != len(targets):
        n_skip += 1
        pbar.set_description(f"Processing {latency} de-low, mult={multiplier}, skipped {n_skip}")
        continue

    messages = [{"role": "system",
                 "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into German text."}]
    for target in targets:
        messages.append({"role": "user",      "content": "<audio>"})
        messages.append({"role": "assistant", "content": target})
    instances.append({"messages": messages,
                      "audios": audio_clip_paths,
                      "multiplier": multiplier})

with open(os.path.join(output_root, output_filename), 'w') as f:
    for instance in instances:
        f.write(json.dumps(instance, ensure_ascii=False) + "\n")
print(f"wrote {len(instances)} instances to {output_root}{output_filename}")
