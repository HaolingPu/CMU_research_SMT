import os
import json
import argparse
import soundfile as sf
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest_root",
        default="/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/prefix_alignment/pa_40k/qe3-lr-aligned",
        help="Per-utt PA JSONs that already passed MetricX-QE + length-ratio filters.",
    )
    ap.add_argument(
        "--orig_tsv",
        default="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv",
        help="Original gigaspeech tsv with audio:start:duration column.",
    )
    ap.add_argument(
        "--audio_clips_root",
        default="/data/group_data/li_lab/haolingp/datasets/gigaspeech/audio_clips_zh_PA-40k",
    )
    ap.add_argument(
        "--output_jsonl",
        default="/data/group_data/li_lab/haolingp/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered_zh-PA-40k.jsonl",
    )
    ap.add_argument("--stepsize", type=int, default=15360, help="960ms @ 16kHz")
    ap.add_argument(
        "--system_prompt",
        default=(
            "You are a professional simultaneous interpreter. You will be given chunks "
            "of English audio and you need to translate the audio into Chinese text."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.audio_clips_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    orig = pd.read_csv(args.orig_tsv, sep="\t")
    orig_by_id = orig.set_index("id")["audio"].to_dict()

    files = sorted(os.listdir(args.manifest_root))
    instances = []
    n_skip_len = 0
    n_skip_missing = 0
    pbar = tqdm(files, desc="PA -> swift")
    for fname in pbar:
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.manifest_root, fname)) as f:
            traj = json.load(f)

        utt_id = traj.get("utt_id") or traj.get("id")
        target = traj["target"]

        if utt_id not in orig_by_id:
            n_skip_missing += 1
            continue
        audio_path, start, duration = orig_by_id[utt_id].split(":")
        wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
        assert sr == 16000

        audio_id, segment_id = utt_id.split("_")
        clip_dir = os.path.join(args.audio_clips_root, audio_id, segment_id)
        os.makedirs(clip_dir, exist_ok=True)

        clip_paths = []
        for idx, i in enumerate(range(0, wav.shape[0], args.stepsize)):
            clip = wav[i : i + args.stepsize]
            cp = os.path.join(clip_dir, f"{idx}.wav")
            sf.write(cp, clip, sr)
            clip_paths.append(cp)

        if len(clip_paths) != len(target):
            n_skip_len += 1
            pbar.set_description(
                f"PA -> swift (skip_len={n_skip_len}, skip_missing={n_skip_missing})"
            )
            continue

        messages = [{"role": "system", "content": args.system_prompt}]
        for tgt in target:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": tgt})

        instances.append({"messages": messages, "audios": clip_paths})

    with open(args.output_jsonl, "w") as f:
        for ins in instances:
            f.write(json.dumps(ins, ensure_ascii=False) + "\n")

    print(f"Wrote {len(instances)} instances to {args.output_jsonl}")
    print(f"Skipped: len-mismatch={n_skip_len}, missing-in-tsv={n_skip_missing}")


if __name__ == "__main__":
    main()
