import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
            
DEFAULT_TSV = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
DEFAULT_OUTPUT_ROOT = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
DEFAULT_AUDIO_CLIPS_ROOT_TEMPLATE = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_zh_consensus-{tag}/"
DEFAULT_OUTPUT_FILENAME_TEMPLATE = "train_s_zh-consensus-{tag}.jsonl"

SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. "
    "You will be given chunks of English audio and you need to translate the audio into Chinese text."
)
CHUNK_SAMPLES = 15360  # 960 ms at 16 kHz


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert consensus-decoding JSON outputs into Swift/Megatron training JSONL."
    )
    p.add_argument("--manifest-root", required=True,
                   help="Directory of per-utterance {utt_id}.json files.")
    p.add_argument("--variant-tag", required=True,
                   help="Short tag used in output filenames and audio clip paths (e.g. top5, minp0.1, topp0.9).")
    p.add_argument("--tsv-path", default=DEFAULT_TSV)
    p.add_argument("--audio-clips-root", default=None)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output-filename", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Optional cap for debugging.")
    args = p.parse_args()

    if args.audio_clips_root is None:
        args.audio_clips_root = DEFAULT_AUDIO_CLIPS_ROOT_TEMPLATE.format(tag=args.variant_tag)
    if args.output_filename is None:
        args.output_filename = DEFAULT_OUTPUT_FILENAME_TEMPLATE.format(tag=args.variant_tag)
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    orig_manifest = pd.read_csv(args.tsv_path, sep="\t")
    os.makedirs(args.audio_clips_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    json_files = sorted(glob(os.path.join(args.manifest_root, "*.json")))
    if args.max_rows is not None:
        json_files = json_files[: args.max_rows]

    instances = []
    n_skip = 0
    pbar = tqdm(json_files, desc="Processing, skipped 0 instances")
    for json_file in pbar:
        with open(json_file, "r") as f:
            item = json.load(f)

        utt_id = item["utt_id"]
        target_trajectory = item["target_trajectory"]

        rows = orig_manifest[orig_manifest["id"] == utt_id]
        if len(rows) == 0:
            n_skip += 1
            pbar.set_description(f"Processing, skipped {n_skip} instances")
            continue

        audio_path, start, duration = rows.iloc[0]["audio"].split(":")
        try:
            wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
        except sf.LibsndfileError as e:
            n_skip += 1
            pbar.set_description(f"Processing, skipped {n_skip} instances")
            print(f"[skip] {utt_id}: {audio_path}:{start}:{duration} — {e}")
            continue
        assert sr == 16000

        multiplier = int(rng.integers(1, 13))
        stepsize = CHUNK_SAMPLES * multiplier

        audio_id, segment_id = utt_id.split("_")
        audio_clips_dir = os.path.join(
            args.audio_clips_root, audio_id, segment_id, f"multiplier_{multiplier}"
        )
        os.makedirs(audio_clips_dir, exist_ok=True)

        audio_clip_paths = []
        for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            wav_clip = wav[i : i + stepsize]
            clip_path = os.path.join(audio_clips_dir, f"{idx}.wav")
            sf.write(clip_path, wav_clip, sr)
            audio_clip_paths.append(clip_path)

        targets = []
        for i in range(0, len(target_trajectory), multiplier):
            targets.append("".join(target_trajectory[i : i + multiplier]))

        if len(audio_clip_paths) != len(targets):
            n_skip += 1
            pbar.set_description(f"Processing, skipped {n_skip} instances")
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for target in targets:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": target})

        instances.append({
            "messages": messages,
            "audios": audio_clip_paths,
            "multiplier": multiplier,
        })

    output_path = os.path.join(args.output_root, args.output_filename)
    with open(output_path, "w") as f:
        for instance in instances:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    print(f"Wrote {len(instances)} instances to {output_path} (skipped {n_skip}).")


if __name__ == "__main__":
    main()

"""
for k in 1 5 10 20; do
  python scripts/train/convert2swift_consensus.py \
    --manifest-root /data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/consensus_future/topk/consensus_decoding_en_zh_top${k}_qe3 \
    --variant-tag topk${k}
done

python scripts/train/convert2swift_consensus.py \
    --manifest-root /data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/consensus_future/topk/consensus_decoding_en_zh_top5_v2_qe3 \
    --variant-tag topk5_v2

python scripts/train/convert2swift_consensus.py \
    --manifest-root /data/group_data/li_lab/siqiouya/datasets/gigaspeech/consensus_decoding/topk/consensus_decoding_en_zh_top_5_futures200-segale/qe3-lr-aligned-full \
    --variant-tag topk5_f200
"""