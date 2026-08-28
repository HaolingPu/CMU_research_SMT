#!/usr/bin/env python3
"""Controlled conv2swift for the matched-ID old-vs-new pipeline experiment.

Reads consensus per-utt JSONs (task_*/per_utt/{utt_id}.json) for a FIXED id-list,
in the SAME order, with a DETERMINISTIC per-utt multiplier (seeded by utt_id) so
that two runs (old pipeline vs new pipeline) over the same id-list produce
byte-identical audio chunking and differ ONLY in the target_trajectory text.
"""
import argparse, json, os, zlib
from glob import glob

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. "
    "You will be given chunks of English audio and you need to translate the audio into Chinese text."
)
CHUNK_SAMPLES = 15360  # 960 ms at 16 kHz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-root", required=True, help="Consensus root containing task_*/per_utt/*.json")
    p.add_argument("--id-list", required=True, help="File with one utt_id per line (defines set AND order).")
    p.add_argument("--tsv-path", required=True, help="Manifest TSV for audio lookup.")
    p.add_argument("--audio-clips-root", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def per_id_multiplier(utt_id, seed):
    h = zlib.crc32(utt_id.encode()) & 0xFFFFFFFF
    return int(np.random.default_rng(seed + h).integers(1, 13))


def main():
    args = parse_args()
    orig = pd.read_csv(args.tsv_path, sep="\t")
    audio_by_id = dict(zip(orig["id"].astype(str), orig["audio"].astype(str)))
    os.makedirs(args.audio_clips_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # id -> json path (recursive over task_*/per_utt)
    index = {os.path.basename(p)[:-5]: p
             for p in glob(os.path.join(args.manifest_root, "task_*", "per_utt", "*.json"))}

    with open(args.id_list) as f:
        ids = [l.strip() for l in f if l.strip()]

    n_skip = 0
    instances = []
    for utt_id in tqdm(ids, desc=os.path.basename(args.output_file)):
        jp = index.get(utt_id)
        if jp is None or utt_id not in audio_by_id:
            n_skip += 1; continue
        item = json.load(open(jp))
        target_trajectory = item.get("target_trajectory")
        if not target_trajectory:
            n_skip += 1; continue

        audio_path, start, duration = audio_by_id[utt_id].split(":")
        try:
            wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
        except Exception:
            n_skip += 1; continue
        assert sr == 16000

        multiplier = per_id_multiplier(utt_id, args.seed)
        stepsize = CHUNK_SAMPLES * multiplier
        audio_id, segment_id = utt_id.split("_")
        clip_dir = os.path.join(args.audio_clips_root, audio_id, segment_id, f"multiplier_{multiplier}")
        os.makedirs(clip_dir, exist_ok=True)

        clip_paths = []
        for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            cp = os.path.join(clip_dir, f"{idx}.wav")
            sf.write(cp, wav[i:i + stepsize], sr)
            clip_paths.append(cp)

        targets = ["".join(target_trajectory[i:i + multiplier])
                   for i in range(0, len(target_trajectory), multiplier)]
        if len(clip_paths) != len(targets):
            n_skip += 1; continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in targets:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": t})
        instances.append({"messages": messages, "audios": clip_paths, "multiplier": multiplier})

    with open(args.output_file, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"wrote {args.output_file}: {len(instances)} rows  (skipped {n_skip})")


if __name__ == "__main__":
    main()
