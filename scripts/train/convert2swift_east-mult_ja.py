"""
en->ja version of convert2swift_east-mult.py

Reads the ja sub-sentence-QE-filtered EAST trajectory tree and emits a swift
SFT manifest in the latency2mult format used by train_EAST-latency2mult_s_ja.sh.

Audio source: same English audio TSV as zh (en->* shares the same source audio).
Per-latency multiplier ranges follow the original (low=1, med=2, high=3-12).
"""
import os
import json
import io
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm

# Master English audio TSV (id -> "audio:start:duration"); shared across target langs.
tsv_path = '/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv'
orig_manifest = pd.read_csv(tsv_path, sep='\t')

# ja sub-sentence QE-filtered EAST trajectories (~37.5K total, ~12.5K per latency).
manifest_root = "/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/segale_qe/final_jsonl_east"
audio_clips_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_ja_EAST-latency2mult/"
# Direct output to the _s filename expected by train_EAST-latency2mult_s_ja.sh,
# since the ja data is already small after sub-sentence QE.
output_filename = "train_s_ja-EAST-latency2mult_origin.jsonl"

output_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
os.makedirs(audio_clips_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)
audio_ids = os.listdir(manifest_root)
latency2multiplier = {
    'low': (1, 2),      # multiplier 1
    'medium': (2, 3),   # multiplier 2
    'high': (3, 13),    # multiplier 3-12
}

instances = []

for latency, (mult_low, mult_high) in latency2multiplier.items():
    latency_traj = []

    for audio_id in tqdm(audio_ids, desc=f"Loading {latency} latency manifests"):
        latency_traj_path = os.path.join(manifest_root, audio_id, f"{latency}_latency.jsonl")
        if not os.path.exists(latency_traj_path):
            continue

        with open(latency_traj_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                latency_traj.append(data)

    pbar = tqdm(latency_traj, desc=f"Processing {latency} latency, skipped 0 instances")
    n_skip = 0
    for traj in pbar:

        audio_path, start, duration = orig_manifest[orig_manifest['id'] == traj['utt_id']].iloc[0]['audio'].split(':')
        wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))

        assert sr == 16000
        multiplier = np.random.randint(mult_low, mult_high)
        stepsize = 15360 * multiplier

        audio_id, segment_id = traj['utt_id'].split('_')

        audio_clips_dir = os.path.join(audio_clips_root, latency, audio_id, segment_id, f"multiplier_{multiplier}")
        os.makedirs(audio_clips_dir, exist_ok=True)
        audio_clip_paths = []

        for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            wav_clip = wav[i : i + stepsize]
            clip_path = os.path.join(audio_clips_dir, f"{idx}.wav")
            sf.write(clip_path, wav_clip, sr)
            audio_clip_paths.append(clip_path)

        targets = []
        for i in range(0, len(traj['target']), multiplier):
            targets.append("".join(traj['target'][i:i+multiplier]))  # Japanese: no inter-token space, same as Chinese

        if len(audio_clip_paths) != len(targets):
            n_skip += 1
            pbar.set_description(f"Processing {latency} latency, multiplier {multiplier}, skipped {n_skip} instances")
            continue

        messages = [
            {"role": "system", "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Japanese text."},
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
