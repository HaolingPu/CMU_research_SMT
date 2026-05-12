#!/usr/bin/env python
"""Synthesize ElevenLabs WAVs for the future-aware test set.

Reads the future_aware_testset_v2 TSV (id, source, reference, ...), synthesizes
each English `source` to a WAV via the ElevenLabs API, and writes
`source.txt` (paths), `target.txt` (Chinese references), and a StreamLAAL-
compatible `data_<speed_tag>.yaml` manifest with per-utterance segments.

API key is read from the ELEVENLABS_API_KEY env var.
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings
from tqdm import tqdm


DEFAULT_TSV = "/data/group_data/li_lab/siqiouya/datasets/future_aware_test/future_aware_testset_v2.tsv"
DEFAULT_OUTPUT_DIR = "/data/group_data/li_lab/siqiouya/datasets/future_aware_test"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George
DEFAULT_MODEL_ID = "eleven_v3"
DEFAULT_OUTPUT_FORMAT = "wav_16000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-tsv", default=DEFAULT_TSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Voice speed (ElevenLabs accepts 0.7-1.2; <1 slower, >1 faster).",
    )
    p.add_argument(
        "--slower",
        action="store_true",
        help="Shortcut: --speed 0.7 plus a [slower] tag prepended to every source. Overrides --speed.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY env var is not set.", file=sys.stderr)
        return 2

    if args.slower:
        speed = 0.7
        text_prefix = "[slower] "
        speed_tag = "spd_slower"
    else:
        speed = args.speed
        text_prefix = ""
        speed_tag = f"spd_{args.speed}"

    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / f"wav_{speed_tag}"
    wav_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t")
    for col in ("id", "source", "reference"):
        if col not in df.columns:
            print(f"ERROR: TSV missing required column: {col}", file=sys.stderr)
            return 2

    client = ElevenLabs(api_key=api_key)
    voice_settings = VoiceSettings(speed=speed)

    print(
        f"Synthesizing {len(df)} rows | voice={args.voice_id} model={args.model_id} "
        f"format={args.output_format} speed={speed} prefix={text_prefix!r}"
    )
    print(f"Output: {wav_dir}")

    n_synth = 0
    n_skip = 0
    failed_ids: list[str] = []
    successful_rows: list[tuple[str, str, str]] = []  # (id, wav_path, reference)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = str(row["id"])
        source_text = str(row["source"])
        reference = str(row["reference"])
        wav_path = wav_dir / f"{utt_id}.wav"

        if wav_path.exists() and not args.overwrite:
            n_skip += 1
            successful_rows.append((utt_id, str(wav_path), reference))
            continue

        try:
            audio_iter = client.text_to_speech.convert(
                args.voice_id,
                text=text_prefix + source_text,
                model_id=args.model_id,
                output_format=args.output_format,
                voice_settings=voice_settings,
            )
            audio_bytes = b"".join(audio_iter)
            wav_path.write_bytes(audio_bytes)
            n_synth += 1
            successful_rows.append((utt_id, str(wav_path), reference))
        except Exception as e:
            print(f"\n[WARN] {utt_id} failed: {e}", file=sys.stderr)
            failed_ids.append(utt_id)

    source_txt = output_dir / f"source_{speed_tag}.txt"
    target_txt = output_dir / "target.txt"
    source_text_txt = output_dir / "source_text.txt"
    data_yaml = output_dir / f"data_{speed_tag}.yaml"
    id_to_source = {str(row["id"]): str(row["source"]) for _, row in df.iterrows()}
    with source_txt.open("w") as fs, target_txt.open("w") as ft, \
         source_text_txt.open("w") as fe, data_yaml.open("w") as fy:
        yaml_entries = []
        for utt_id, wav_path, reference in successful_rows:
            fs.write(f"{wav_path}\n")
            ft.write(f"{reference}\n")
            fe.write(f"{id_to_source[utt_id]}\n")
            with wave.open(wav_path, "rb") as w:
                duration = round(w.getnframes() / w.getframerate(), 3)
            yaml_entries.append(
                f"{{duration: {duration}, offset: 0, rW: 0, "
                f"speaker_id: NA, uW: 0, wav: {Path(wav_path).name}}}"
            )
        fy.write("[" + ",\n  ".join(yaml_entries) + "]\n")

    print(
        f"\nDone. synthesized={n_synth} skipped(existing)={n_skip} "
        f"failed={len(failed_ids)} manifest_rows={len(successful_rows)}"
    )
    print(f"  {source_txt}")
    print(f"  {target_txt}")
    print(f"  {source_text_txt}")
    print(f"  {data_yaml}")
    if failed_ids:
        print(f"Failed ids: {failed_ids}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
