#!/usr/bin/env python3
"""
Export MFA corpus from GigaSpeech manifest TSV.

For each row:
  - read `audio` field formatted as: /path/to/file.opus:start_frame:num_frames
  - extract the segment to wav (16k mono) with ffmpeg
  - write cleaned transcript to .lab (default from src_text_full)

Output layout (default):
  <output_dir>/<recording_id>/<utt_id>.wav
  <output_dir>/<recording_id>/<utt_id>.lab
where recording_id = utt_id without the last "_<idx>" suffix.
"""

import argparse
import ast
import csv
import json
import math
import os
import re
import subprocess
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GigaSpeech MFA corpus (wav/lab).")
    parser.add_argument("--input-tsv", required=True, help="Manifest TSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write MFA corpus.")
    parser.add_argument("--id-column", default="id", help="Utt ID column in TSV.")
    parser.add_argument("--audio-column", default="audio", help="Audio spec column in TSV.")
    parser.add_argument(
        "--text-column",
        default="src_text_full",
        help="Primary text column for .lab (default: src_text_full).",
    )
    parser.add_argument(
        "--fallback-text-columns",
        default="src_text,asr",
        help="Comma-separated fallback text columns when primary column is empty/invalid.",
    )
    parser.add_argument(
        "--audio-unit-sr",
        type=int,
        default=16000,
        help="Sample rate assumed by audio offset/num_frames in audio spec.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Output wav sample rate.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Worker id for row split.")
    parser.add_argument("--num-tasks", type=int, default=1, help="Total number of workers.")
    parser.add_argument("--max-rows", type=int, default=None, help="Process first N assigned rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wav/lab.")
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write all wav/lab directly under output-dir (no recording subfolders).",
    )
    parser.add_argument(
        "--save-index-jsonl",
        default=None,
        help="Optional path to save export index jsonl.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_text_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    raw = str(raw_value).strip()
    if not raw:
        return []

    parsed: Any = None
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        out = []
        for item in parsed:
            s = str(item).strip()
            if s:
                out.append(s)
        if out:
            return out

    # Fallback: treat as one sentence, stripping bracket/quote wrappers if present.
    s = raw
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1]
    s = s.strip()
    return [s] if s else []


def get_text_for_lab(row: Dict[str, str], text_column: str, fallback_cols: List[str]) -> str:
    candidates = [text_column] + fallback_cols
    for col in candidates:
        if col not in row:
            continue
        value = row.get(col, "")
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue

        if col == text_column:
            sentences = parse_text_list(value)
            joined = " ".join(sentences).strip()
            cleaned = clean_text(joined)
        else:
            cleaned = clean_text(value)

        if cleaned:
            return cleaned
    return ""


def parse_audio_spec(spec: str) -> Tuple[str, int, int]:
    """
    Parse "/path/to/audio.opus:start_frame:num_frames".
    """
    m = re.match(r"^(.*):([0-9]+):([0-9]+)$", spec.strip())
    if not m:
        raise ValueError(f"Bad audio spec format: {spec}")
    path = m.group(1)
    start = int(m.group(2))
    num_frames = int(m.group(3))
    return path, start, num_frames


def recording_id_from_utt(utt_id: str) -> str:
    if "_" not in utt_id:
        return utt_id
    return utt_id.rsplit("_", 1)[0]


def split_output_paths(output_dir: str, utt_id: str, flat_output: bool) -> Tuple[str, str]:
    if flat_output:
        base_dir = output_dir
    else:
        base_dir = os.path.join(output_dir, recording_id_from_utt(utt_id))
    os.makedirs(base_dir, exist_ok=True)
    return (
        os.path.join(base_dir, f"{utt_id}.wav"),
        os.path.join(base_dir, f"{utt_id}.lab"),
    )


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total_rows = sum(1 for _ in f) - 1
    if total_rows <= task_id:
        return 0
    return int(math.ceil((total_rows - task_id) / num_tasks))


def iter_assigned_rows(
    input_tsv: str, task_id: int, num_tasks: int
) -> Iterator[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def extract_segment_ffmpeg(
    src_audio: str,
    start_frame: int,
    num_frames: int,
    audio_unit_sr: int,
    target_sr: int,
    out_wav: str,
) -> None:
    start_sec = start_frame / float(audio_unit_sr)
    dur_sec = num_frames / float(audio_unit_sr)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{dur_sec:.6f}",
        "-i",
        src_audio,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        out_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def main() -> None:
    args = parse_args()
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise ValueError(f"Invalid task split: task-id={args.task_id}, num-tasks={args.num_tasks}")
    if args.audio_unit_sr <= 0 or args.target_sr <= 0:
        raise ValueError("Sample rates must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)

    fallback_cols = [x.strip() for x in args.fallback_text_columns.split(",") if x.strip()]
    assigned = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
    if args.max_rows is not None:
        assigned = min(assigned, args.max_rows)

    print(
        f"[Task {args.task_id}] Exporting MFA corpus: input={args.input_tsv}, "
        f"assigned_rows={assigned}, output={args.output_dir}"
    )

    written = 0
    skipped_existing = 0
    failed = 0
    processed = 0
    index_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=assigned, desc=f"export_task_{args.task_id}")

    for row_idx, row in iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks):
        if args.max_rows is not None and processed >= args.max_rows:
            break
        processed += 1

        utt_id = str(row.get(args.id_column, "")).strip()
        if not utt_id:
            utt_id = f"row_{row_idx:09d}"

        wav_path, lab_path = split_output_paths(args.output_dir, utt_id, args.flat_output)

        if (
            os.path.exists(wav_path)
            and os.path.exists(lab_path)
            and not args.overwrite
        ):
            skipped_existing += 1
            pbar.update(1)
            continue

        text = get_text_for_lab(row, args.text_column, fallback_cols)
        if not text:
            failed += 1
            pbar.update(1)
            continue

        audio_spec = str(row.get(args.audio_column, "")).strip()
        try:
            src_audio, start_frame, num_frames = parse_audio_spec(audio_spec)
            if not os.path.exists(src_audio):
                raise FileNotFoundError(src_audio)

            extract_segment_ffmpeg(
                src_audio=src_audio,
                start_frame=start_frame,
                num_frames=num_frames,
                audio_unit_sr=args.audio_unit_sr,
                target_sr=args.target_sr,
                out_wav=wav_path,
            )

            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(text)

            written += 1
            if args.save_index_jsonl is not None:
                index_rows.append(
                    {
                        "utt_id": utt_id,
                        "row_index": row_idx,
                        "wav": wav_path,
                        "lab": lab_path,
                        "audio_spec": audio_spec,
                        "n_frames": row.get("n_frames"),
                        "text_column": args.text_column,
                    }
                )
        except Exception as e:
            failed += 1
            err_path = lab_path + ".error.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utt_id": utt_id,
                        "row_index": row_idx,
                        "audio_spec": audio_spec,
                        "error": str(e),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        pbar.update(1)

    pbar.close()

    if args.save_index_jsonl is not None and index_rows:
        os.makedirs(os.path.dirname(args.save_index_jsonl) or ".", exist_ok=True)
        with open(args.save_index_jsonl, "w", encoding="utf-8") as f:
            for item in index_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[Task {args.task_id}] Done. written={written}, "
        f"skipped_existing={skipped_existing}, failed={failed}"
    )


if __name__ == "__main__":
    main()