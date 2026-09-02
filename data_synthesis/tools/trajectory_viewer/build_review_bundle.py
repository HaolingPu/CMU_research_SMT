#!/usr/bin/env python3
"""Build a portable mentor-review bundle from consensus decode outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


CHUNK_RE = re.compile(r"^Chunk (\d+)/(\d+)$")
GROUP_RE = re.compile(
    r"^\[(Raw|Selected) candidates\] (.*?) \| (plausible|contrastive) \| "
    r"model=(\S+) mode=(\S+) count=(\d+)$"
)
ITEM_RE = re.compile(r"^\s+\d+\.\s+(.*)$")
FILTER_RE = re.compile(r"^\s*Filter summary: kept=(\d+)/(\d+); dropped: (.*)$")
ACTION_RE = re.compile(r"^-> (READ|WRITE) delta=(.*)$")
FINAL_RE = re.compile(r"^\s*\[Final\] delta=(.*)$")
AUDIO_RE = re.compile(r"^(.*):(\d+):(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--decode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--run-name", default="Future Consensus 40K")
    parser.add_argument("--audio-unit-sr", type=int, default=16000)
    parser.add_argument("--audio-bitrate", default="48k")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def literal(value: str, default: Any = "") -> Any:
    try:
        return ast.literal_eval(value.strip())
    except (SyntaxError, ValueError):
        return default


def parse_future_log(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_group: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            chunk_match = CHUNK_RE.match(line)
            if chunk_match:
                if current is not None:
                    steps.append(current)
                current = {
                    "step": int(chunk_match.group(1)),
                    "total_steps": int(chunk_match.group(2)),
                    "selected_futures": [],
                    "raw_stats": [],
                }
                active_group = None
                continue
            if current is None:
                continue

            if line.startswith("source_observed: "):
                current["source_observed"] = literal(line.split(": ", 1)[1])
                active_group = None
                continue
            if line.startswith("future_source_prefix: "):
                current["future_source_prefix"] = literal(line.split(": ", 1)[1])
                active_group = None
                continue
            if line.startswith("committed_before: "):
                current["committed_before"] = literal(line.split(": ", 1)[1])
                active_group = None
                continue
            if line.startswith("committed_after: "):
                current["committed_after"] = literal(line.split(": ", 1)[1])
                active_group = None
                continue

            group_match = GROUP_RE.match(line)
            if group_match:
                kind, label, mode, model, _, count = group_match.groups()
                active_group = {
                    "label": label,
                    "model": model,
                    "mode": mode,
                    "count": int(count),
                    "candidates": [],
                }
                target = "raw_stats" if kind == "Raw" else "selected_futures"
                current[target].append(active_group)
                continue

            item_match = ITEM_RE.match(line)
            if item_match and active_group is not None:
                candidate = literal(item_match.group(1), item_match.group(1).strip())
                if "candidates" in active_group:
                    active_group["candidates"].append(str(candidate))
                continue

            filter_match = FILTER_RE.match(line)
            if filter_match and active_group is not None:
                active_group["kept"] = int(filter_match.group(1))
                active_group["requested"] = int(filter_match.group(2))
                active_group["dropped"] = filter_match.group(3)
                active_group.pop("candidates", None)
                active_group = None
                continue

            action_match = ACTION_RE.match(line)
            if action_match:
                current["action"] = action_match.group(1)
                current["delta"] = literal(action_match.group(2))
                active_group = None
                continue

            final_match = FINAL_RE.match(line)
            if final_match:
                delta = literal(final_match.group(1))
                current["action"] = "WRITE" if delta else "READ"
                current["delta"] = delta
                active_group = None

    if current is not None:
        steps.append(current)
    return steps


def task_number(path: Path) -> int:
    match = re.search(r"task_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def index_complete_cases(decode_root: Path) -> dict[str, tuple[Path, Path, Path]]:
    indexed: dict[str, tuple[Path, Path, Path]] = {}
    task_dirs = sorted(decode_root.glob("task_*"), key=task_number)
    for task_dir in task_dirs:
        per_utt = task_dir / "per_utt"
        verbose = task_dir / "verbose"
        if not per_utt.is_dir() or not verbose.is_dir():
            continue
        for json_path in per_utt.glob("*.json"):
            utt_id = json_path.stem
            log_path = verbose / f"verbose_{utt_id}.log"
            if log_path.is_file() and utt_id not in indexed:
                indexed[utt_id] = (json_path, log_path, task_dir)
    return indexed


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def merge_case(
    row_index: int,
    row: dict[str, str],
    decoded: dict[str, Any],
    parsed_steps: list[dict[str, Any]],
    task_name: str,
) -> dict[str, Any]:
    source_chunks = list(decoded.get("src_trajectory") or [])
    target_deltas = list(decoded.get("target_trajectory") or [])
    actions = list(decoded.get("actions") or [])
    parsed_by_step = {int(step["step"]) - 1: step for step in parsed_steps}

    steps: list[dict[str, Any]] = []
    source_so_far = ""
    translation_so_far = ""
    total = max(len(source_chunks), len(target_deltas), len(actions))
    for index in range(total):
        chunk = str(source_chunks[index] if index < len(source_chunks) else "")
        delta = str(target_deltas[index] if index < len(target_deltas) else "")
        action = str(actions[index] if index < len(actions) else ("WRITE" if delta else "READ"))
        source_so_far += chunk
        translation_so_far += delta
        log_step = parsed_by_step.get(index, {})
        steps.append(
            {
                "step": index + 1,
                "source_chunk": chunk,
                "source_cumulative": source_so_far.strip(),
                "translation_delta": delta,
                "translation_cumulative": translation_so_far,
                "action": action,
                "future_source_prefix": log_step.get("future_source_prefix", ""),
                "selected_futures": log_step.get("selected_futures", []),
                "raw_stats": log_step.get("raw_stats", []),
            }
        )

    metrics = {
        key: finite_number(value)
        for key, value in (decoded.get("metrics") or {}).items()
    }
    return {
        "utt_id": decoded.get("utt_id") or row.get("id"),
        "row_index": row_index,
        "task": task_name,
        "audio_spec": row.get("audio", ""),
        "audio_url": f"audio/{row.get('id')}.mp3",
        "speaker": row.get("speaker", ""),
        "source_full_text": decoded.get("source_full_text", ""),
        "source_sentences": decoded.get("src_text_full", []),
        "prediction": decoded.get("prediction", ""),
        "reference_text": decoded.get("reference_text", ""),
        "metrics": metrics,
        "steps": steps,
        "write_steps": sum(action == "WRITE" for action in actions),
        "read_steps": sum(action == "READ" for action in actions),
    }


def parse_audio_spec(spec: str) -> tuple[Path, int, int]:
    match = AUDIO_RE.match(spec.strip())
    if not match:
        raise ValueError(f"Invalid audio spec: {spec}")
    return Path(match.group(1)), int(match.group(2)), int(match.group(3))


def extract_audio(
    audio_spec: str,
    output_path: Path,
    audio_unit_sr: int,
    bitrate: str,
) -> float:
    source, start_frame, num_frames = parse_audio_spec(audio_spec)
    start_seconds = start_frame / float(audio_unit_sr)
    duration_seconds = num_frames / float(audio_unit_sr)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")
    return duration_seconds


def iter_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            yield index, row


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    input_tsv = Path(args.input_tsv)
    decode_root = Path(args.decode_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    audio_dir = output_dir / "audio"
    raw_json_dir = output_dir / "raw" / "per_utt"
    raw_log_dir = output_dir / "raw" / "verbose"
    for directory in (data_dir, audio_dir, raw_json_dir, raw_log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    static_dir = Path(__file__).resolve().parent / "static"
    for source in static_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)

    complete = index_complete_cases(decode_root)
    cases: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    audio_failures: list[dict[str, str]] = []

    for row_index, row in iter_rows(input_tsv):
        utt_id = str(row.get("id", ""))
        paths = complete.get(utt_id)
        if not paths:
            continue
        json_path, log_path, task_dir = paths
        with json_path.open("r", encoding="utf-8") as stream:
            decoded = json.load(stream)
        parsed_steps = parse_future_log(log_path)
        case = merge_case(row_index, row, decoded, parsed_steps, task_dir.name)

        if not args.skip_audio:
            try:
                case["audio_duration_seconds"] = extract_audio(
                    row.get("audio", ""),
                    audio_dir / f"{utt_id}.mp3",
                    args.audio_unit_sr,
                    args.audio_bitrate,
                )
            except Exception as error:  # Keep the textual review usable.
                case["audio_url"] = ""
                audio_failures.append({"utt_id": utt_id, "error": str(error)})
        else:
            case["audio_url"] = ""

        shutil.copy2(json_path, raw_json_dir / json_path.name)
        shutil.copy2(log_path, raw_log_dir / log_path.name)
        cases.append(case)
        manifest_rows.append(
            {
                "row_index": row_index,
                "utt_id": utt_id,
                "task": task_dir.name,
                "steps": len(case["steps"]),
                "write_steps": case["write_steps"],
                "bleu_char": case["metrics"].get("bleu_char"),
                "laal_text": case["metrics"].get("laal_text"),
                "audio": row.get("audio", ""),
            }
        )
        if len(cases) >= args.limit:
            break

    if len(cases) < args.limit:
        raise RuntimeError(
            f"Only found {len(cases)} complete JSON+verbose cases; requested {args.limit}"
        )

    payload = {
        "meta": {
            "run_name": args.run_name,
            "case_count": len(cases),
            "selection": "First complete cases in input TSV order with JSON and verbose log",
            "decode_root": str(decode_root),
            "input_tsv": str(input_tsv),
            "audio_failures": audio_failures,
        },
        "cases": cases,
    }
    with (data_dir / "review.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))

    with (output_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built {len(cases)} cases at {output_dir}")
    print(f"Audio failures: {len(audio_failures)}")


if __name__ == "__main__":
    main()
