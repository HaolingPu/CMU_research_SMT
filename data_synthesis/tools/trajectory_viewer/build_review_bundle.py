#!/usr/bin/env python3
"""Build a portable mentor-review bundle from consensus decode outputs.

Two input layouts are supported:

* ``--decode-root``: a production decode root (``task_*/per_utt/*.json`` +
  ``task_*/verbose/verbose_<utt>.log``) plus ``--input-tsv`` for row order and
  audio specs.
* ``--raw-dir``: a flat ``per_utt/`` + ``verbose/`` pair (for example the
  ``raw/`` folder of a previously built bundle). Row order, task names and
  audio metadata are taken from ``--order-from`` (an existing
  ``data/review.json``) when given, otherwise from the TSV, otherwise from the
  natural order of the utterance ids.

Besides ``data/review.json`` the builder now writes one
``data/consensus/<utt_id>.json`` per case with the full next-token consensus
trace parsed from the verbose log: per consensus step the accepted token or
stop reason, the intersection, and every retained future's top candidates with
probabilities. The page loads these lazily.
"""

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
TOO_FEW_RE = re.compile(r"^\s*-> READ \(too few futures\)\s*$")
FINAL_RE = re.compile(r"^\s*\[Final\] delta=(.*)$")
AUDIO_RE = re.compile(r"^(.*):(\d+):(\d+)$")
REPR = r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
RAW_SAMPLING_RE = re.compile(r"^\[Step 1-1\] raw_future_sampling total=(\d+) accepted=(\d+)$")
CONS_HDR_RE = re.compile(r"^\[Step 4-5\] consensus summary:\s*$")
STEP_ACC_RE = re.compile(rf"^\s{{2}}step=(\d+) accepted=({REPR}) pending=({REPR})\s*$")
STEP_STOP_RE = re.compile(rf"^\s{{2}}step=(\d+|filter) stop=(\S+) intersection=(\[.*\]) pending=({REPR})\s*$")
FUTURE_RE = re.compile(r"^\s{4}future\[(\d+)\] candidates=(\d+): \[(.*)\]\s*$")
PAIR_RE = re.compile(rf"({REPR}):([0-9.eE+-]+)")
HORIZON_RE = re.compile(rf"^\[Step 5\.5\] horizon_filter: dropped pending=({REPR}) \(len=(\d+) < min_horizon=(\d+)\) -> READ\s*$")
TRIM_RE = re.compile(rf"^\[Step 6-7\] (pending_before_trim|commit_after_trim)=({REPR})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", default=None)
    parser.add_argument("--decode-root", default=None)
    parser.add_argument("--raw-dir", default=None,
                        help="flat per_utt/ + verbose/ directory (alternative to --decode-root)")
    parser.add_argument("--order-from", default=None,
                        help="existing data/review.json to reuse row order, task names and audio metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--run-name", default="Future Consensus 40K")
    parser.add_argument("--audio-unit-sr", type=int, default=16000)
    parser.add_argument("--audio-bitrate", default="48k")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--top-candidates", type=int, default=8,
                        help="candidates kept per future per consensus step in the detail files")
    parser.add_argument("--max-intersection", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.decode_root and not args.raw_dir:
        parser.error("one of --decode-root or --raw-dir is required")
    if args.decode_root and not args.input_tsv:
        parser.error("--decode-root requires --input-tsv")
    return args


def literal(value: str, default: Any = "") -> Any:
    try:
        return ast.literal_eval(value.strip())
    except (SyntaxError, ValueError):
        return default


def parse_pairs(body: str, top_n: int) -> list[list[Any]]:
    pairs: list[list[Any]] = []
    for match in PAIR_RE.finditer(body):
        token = literal(match.group(1), match.group(1))
        try:
            prob = float(match.group(2))
        except ValueError:
            continue
        pairs.append([str(token), prob])
        if len(pairs) >= top_n:
            break
    return pairs


def new_step(number: int, total: int) -> dict[str, Any]:
    return {
        "step": number,
        "total_steps": total,
        "selected_futures": [],
        "raw_stats": [],
        "consensus": [],
        "raw_total": None,
        "raw_accepted": None,
        "horizon": None,
        "pending_before_trim": None,
        "commit_after_trim": None,
        "too_few_futures": False,
        "final_completion": False,
    }


def parse_future_log(path: Path, top_n: int = 8, max_intersection: int = 24) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_group: dict[str, Any] | None = None
    active_cons: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            chunk_match = CHUNK_RE.match(line)
            if chunk_match:
                if current is not None:
                    steps.append(current)
                current = new_step(int(chunk_match.group(1)), int(chunk_match.group(2)))
                active_group = None
                active_cons = None
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
                active_cons = None
                continue

            raw_match = RAW_SAMPLING_RE.match(line)
            if raw_match:
                current["raw_total"] = int(raw_match.group(1))
                current["raw_accepted"] = int(raw_match.group(2))
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
            if item_match and active_group is not None and active_cons is None:
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

            if CONS_HDR_RE.match(line):
                active_group = None
                active_cons = None
                continue

            acc_match = STEP_ACC_RE.match(line)
            if acc_match:
                active_group = None
                active_cons = {
                    "k": int(acc_match.group(1)),
                    "kind": "accepted",
                    "token": str(literal(acc_match.group(2), "")),
                    "pending": str(literal(acc_match.group(3), "")),
                    "futures": [],
                }
                current["consensus"].append(active_cons)
                continue

            stop_match = STEP_STOP_RE.match(line)
            if stop_match:
                active_group = None
                k_raw = stop_match.group(1)
                intersection = literal(stop_match.group(3), [])
                if not isinstance(intersection, list):
                    intersection = []
                active_cons = {
                    "k": int(k_raw) if k_raw.isdigit() else k_raw,
                    "kind": "stop",
                    "stop_reason": stop_match.group(2),
                    "intersection": [str(x) for x in intersection[:max_intersection]],
                    "intersection_size": len(intersection),
                    "pending": str(literal(stop_match.group(4), "")),
                    "futures": [],
                }
                current["consensus"].append(active_cons)
                if k_raw == "filter":
                    active_cons = None
                continue

            future_match = FUTURE_RE.match(line)
            if future_match and active_cons is not None:
                active_cons["futures"].append({
                    "i": int(future_match.group(1)),
                    "n": int(future_match.group(2)),
                    "top": parse_pairs(future_match.group(3), top_n),
                })
                continue

            horizon_match = HORIZON_RE.match(line)
            if horizon_match:
                current["horizon"] = {
                    "dropped": str(literal(horizon_match.group(1), "")),
                    "len": int(horizon_match.group(2)),
                    "min_horizon": int(horizon_match.group(3)),
                }
                active_cons = None
                continue

            trim_match = TRIM_RE.match(line)
            if trim_match:
                current[trim_match.group(1)] = str(literal(trim_match.group(2), ""))
                active_cons = None
                continue

            if TOO_FEW_RE.match(line):
                current["too_few_futures"] = True
                current["action"] = "READ"
                current["delta"] = ""
                active_cons = None
                continue

            action_match = ACTION_RE.match(line)
            if action_match:
                current["action"] = action_match.group(1)
                current["delta"] = literal(action_match.group(2))
                active_group = None
                active_cons = None
                continue

            final_match = FINAL_RE.match(line)
            if final_match:
                delta = literal(final_match.group(1))
                current["action"] = "WRITE" if delta else "READ"
                current["delta"] = delta
                current["final_completion"] = True
                active_group = None
                active_cons = None

    if current is not None:
        steps.append(current)
    return steps


def summarize_consensus(step: dict[str, Any]) -> dict[str, Any]:
    """Small per-step summary that lives in review.json; the trace goes to the detail file."""
    cons = step.get("consensus") or []
    accepted = [c["token"] for c in cons if c.get("kind") == "accepted"]
    stops = [c for c in cons if c.get("kind") == "stop"]
    stop_reason = stops[-1]["stop_reason"] if stops else None
    for c in cons:
        if c.get("kind") == "accepted" and c.get("futures"):
            n = len(c["futures"])
            top1 = sum(1 for f in c["futures"] if f["top"] and f["top"][0][0] == c["token"])
            probs = [next((p for t, p in f["top"] if t == c["token"]), 0.0) for f in c["futures"]]
            c["n_futures"] = n
            c["top1_agree"] = top1
            c["mean_p"] = round(sum(probs) / n, 4) if n else None
            c["min_p"] = round(min(probs), 4) if probs else None
        elif c.get("kind") == "stop" and c.get("futures"):
            c["n_futures"] = len(c["futures"])
    return {
        "steps": len(cons),
        "accepted": accepted,
        "stop_reason": stop_reason,
        "horizon_dropped": (step.get("horizon") or {}).get("dropped"),
        "raw_total": step.get("raw_total"),
        "raw_accepted": step.get("raw_accepted"),
        "too_few_futures": bool(step.get("too_few_futures")),
        "final_completion": bool(step.get("final_completion")),
    }


def task_number(path: Path) -> int:
    match = re.search(r"task_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def index_complete_cases(decode_root: Path) -> dict[str, tuple[Path, Path, str]]:
    indexed: dict[str, tuple[Path, Path, str]] = {}
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
                indexed[utt_id] = (json_path, log_path, task_dir.name)
    return indexed


def index_raw_dir(raw_dir: Path) -> dict[str, tuple[Path, Path, str]]:
    indexed: dict[str, tuple[Path, Path, str]] = {}
    for json_path in (raw_dir / "per_utt").glob("*.json"):
        utt_id = json_path.stem
        log_path = raw_dir / "verbose" / f"verbose_{utt_id}.log"
        if log_path.is_file():
            indexed[utt_id] = (json_path, log_path, "raw")
    return indexed


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def flatten_selected(step: dict[str, Any]) -> list[dict[str, Any]]:
    """future[i] in the consensus trace indexes the selected futures in group order."""
    flat: list[dict[str, Any]] = []
    for group in step.get("selected_futures") or []:
        for text in group.get("candidates") or []:
            flat.append({
                "i": len(flat),
                "label": group.get("label") or group.get("model") or "sampler",
                "mode": group.get("mode", ""),
                "text": text,
            })
    return flat


def merge_case(
    row_index: int,
    row: dict[str, Any],
    decoded: dict[str, Any],
    parsed_steps: list[dict[str, Any]],
    task_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_chunks = list(decoded.get("src_trajectory") or [])
    target_deltas = list(decoded.get("target_trajectory") or [])
    actions = list(decoded.get("actions") or [])
    parsed_by_step = {int(step["step"]) - 1: step for step in parsed_steps}

    steps: list[dict[str, Any]] = []
    detail_steps: list[dict[str, Any]] = []
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
        summary = summarize_consensus(log_step) if log_step else None
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
                "consensus_summary": summary,
            }
        )
        detail_steps.append(
            {
                "step": index + 1,
                "futures": flatten_selected(log_step) if log_step else [],
                "consensus": log_step.get("consensus", []),
                "horizon": log_step.get("horizon"),
                "pending_before_trim": log_step.get("pending_before_trim"),
                "commit_after_trim": log_step.get("commit_after_trim"),
                "too_few_futures": bool(log_step.get("too_few_futures")),
                "final_completion": bool(log_step.get("final_completion")),
                "committed_before": log_step.get("committed_before"),
            }
        )

    metrics = {
        key: finite_number(value)
        for key, value in (decoded.get("metrics") or {}).items()
    }
    utt_id = decoded.get("utt_id") or row.get("id")
    cons_steps = sum((s["consensus_summary"] or {}).get("steps", 0) for s in steps)
    horizon_drops = sum(1 for s in steps if (s["consensus_summary"] or {}).get("horizon_dropped"))
    too_few = sum(1 for s in steps if (s["consensus_summary"] or {}).get("too_few_futures"))
    case = {
        "utt_id": utt_id,
        "row_index": row_index,
        "task": task_name,
        "audio_spec": row.get("audio", ""),
        "audio_url": f"audio/{utt_id}.mp3",
        "speaker": row.get("speaker", ""),
        "source_full_text": decoded.get("source_full_text", ""),
        "source_sentences": decoded.get("src_text_full", []),
        "prediction": decoded.get("prediction", ""),
        "reference_text": decoded.get("reference_text", ""),
        "metrics": metrics,
        "steps": steps,
        "write_steps": sum(action == "WRITE" for action in actions),
        "read_steps": sum(action == "READ" for action in actions),
        "consensus_steps": cons_steps,
        "horizon_drops": horizon_drops,
        "too_few_future_steps": too_few,
        "detail_url": f"data/consensus/{utt_id}.json",
        "raw_log_url": f"raw/verbose/verbose_{utt_id}.log",
    }
    detail = {"utt_id": utt_id, "steps": detail_steps}
    return case, detail


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
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-ss", f"{start_seconds:.6f}", "-t", f"{duration_seconds:.6f}",
        "-i", str(source), "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-b:a", bitrate, str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")
    return duration_seconds


def iter_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            yield index, row


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    detail_dir = data_dir / "consensus"
    audio_dir = output_dir / "audio"
    raw_json_dir = output_dir / "raw" / "per_utt"
    raw_log_dir = output_dir / "raw" / "verbose"
    for directory in (data_dir, detail_dir, audio_dir, raw_json_dir, raw_log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    static_dir = Path(__file__).resolve().parent / "static"
    for source in static_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)

    if args.decode_root:
        complete = index_complete_cases(Path(args.decode_root))
    else:
        complete = index_raw_dir(Path(args.raw_dir))

    # Row order + metadata: existing review.json > TSV > natural utt order.
    order: list[tuple[int, dict[str, Any]]] = []
    prior: dict[str, dict[str, Any]] = {}
    if args.order_from:
        with Path(args.order_from).open("r", encoding="utf-8") as stream:
            prior_payload = json.load(stream)
        for case in prior_payload.get("cases", []):
            prior[case["utt_id"]] = case
        for case in sorted(prior_payload.get("cases", []), key=lambda c: (c.get("row_index", 10**9), c["utt_id"])):
            order.append((case.get("row_index", 10**9), {"id": case["utt_id"], "audio": case.get("audio_spec", ""), "speaker": case.get("speaker", "")}))
    elif args.input_tsv:
        order = list(iter_rows(Path(args.input_tsv)))
    else:
        order = [(i, {"id": u}) for i, u in enumerate(sorted(complete, key=natural_key))]

    cases: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    audio_failures: list[dict[str, str]] = []

    for row_index, row in order:
        utt_id = str(row.get("id", ""))
        paths = complete.get(utt_id)
        if not paths:
            continue
        json_path, log_path, task_name = paths
        if task_name == "raw" and utt_id in prior:
            task_name = prior[utt_id].get("task", "raw")
        with json_path.open("r", encoding="utf-8") as stream:
            decoded = json.load(stream)
        parsed_steps = parse_future_log(log_path, args.top_candidates, args.max_intersection)
        case, detail = merge_case(row_index, row, decoded, parsed_steps, task_name)

        audio_target = audio_dir / f"{utt_id}.mp3"
        if not args.skip_audio and row.get("audio"):
            try:
                case["audio_duration_seconds"] = extract_audio(
                    row.get("audio", ""), audio_target, args.audio_unit_sr, args.audio_bitrate,
                )
            except Exception as error:  # Keep the textual review usable.
                case["audio_url"] = ""
                audio_failures.append({"utt_id": utt_id, "error": str(error)})
        elif audio_target.is_file():
            case["audio_duration_seconds"] = (prior.get(utt_id) or {}).get("audio_duration_seconds")
        else:
            case["audio_url"] = ""
            case["audio_duration_seconds"] = (prior.get(utt_id) or {}).get("audio_duration_seconds")

        if json_path.resolve() != (raw_json_dir / json_path.name).resolve():
            shutil.copy2(json_path, raw_json_dir / json_path.name)
        if log_path.resolve() != (raw_log_dir / log_path.name).resolve():
            shutil.copy2(log_path, raw_log_dir / log_path.name)
        with (detail_dir / f"{utt_id}.json").open("w", encoding="utf-8") as stream:
            json.dump(detail, stream, ensure_ascii=False, separators=(",", ":"))

        cases.append(case)
        manifest_rows.append(
            {
                "row_index": row_index,
                "utt_id": utt_id,
                "task": task_name,
                "steps": len(case["steps"]),
                "write_steps": case["write_steps"],
                "consensus_steps": case["consensus_steps"],
                "horizon_drops": case["horizon_drops"],
                "bleu_char": case["metrics"].get("bleu_char"),
                "laal_text": case["metrics"].get("laal_text"),
                "audio": row.get("audio", ""),
            }
        )
        if len(cases) >= args.limit:
            break

    if not cases:
        raise RuntimeError("No complete JSON+verbose cases found")
    if len(cases) < args.limit and (args.decode_root or args.input_tsv):
        print(f"WARNING: only {len(cases)} complete cases found; requested {args.limit}")

    payload = {
        "meta": {
            "run_name": args.run_name,
            "case_count": len(cases),
            "selection": "First complete cases in input TSV order with JSON and verbose log",
            "decode_root": str(args.decode_root or args.raw_dir),
            "input_tsv": str(args.input_tsv or args.order_from or ""),
            "audio_failures": audio_failures,
            "consensus_top_candidates": args.top_candidates,
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
    print(f"Consensus detail files: {len(cases)} under {detail_dir}")
    print(f"Audio failures: {len(audio_failures)}")


if __name__ == "__main__":
    main()
