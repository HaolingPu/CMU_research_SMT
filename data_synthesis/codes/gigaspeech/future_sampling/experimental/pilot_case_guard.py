#!/usr/bin/env python3
"""Freeze pilot inputs and validate actual case outputs, never just DONE files."""

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_selection(args):
    csv.field_size_limit(10_000_000)
    with Path(args.input_tsv).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        pool = []
        for index, row in enumerate(reader):
            if index >= args.pool_size:
                break
            pool.append((index, row))
    if not fields or "id" not in fields or len(pool) < args.count:
        raise ValueError("Input lacks IDs or enough rows")
    anchors = [int(x) for x in args.anchor_rows.split(",") if x]
    if len(anchors) > args.count:
        raise ValueError("More diagnostic anchors than requested cases")
    if len(set(anchors)) != len(anchors) or any(i < 0 or i >= len(pool) for i in anchors):
        raise ValueError("Invalid anchor rows")
    rng = random.Random(args.seed)
    chosen = [pool[i] for i in anchors]
    buckets = defaultdict(list)
    for index, row in pool:
        if index not in anchors:
            buckets[row["id"].rsplit("_", 1)[0]].append((index, row))
    groups = sorted(buckets)
    rng.shuffle(groups)
    for values in buckets.values():
        rng.shuffle(values)
    while len(chosen) < args.count:
        for group in groups:
            if buckets[group] and len(chosen) < args.count:
                chosen.append(buckets[group].pop())
    if len({row["id"] for _, row in chosen}) != args.count:
        raise ValueError("Selection contains duplicate IDs")
    output = Path(args.selection_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "input_50.tsv"
    if target.exists():
        raise FileExistsError("Frozen selection already exists; do not overwrite")
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(row for _, row in chosen)
    cases = [{"pilot_row": i, "original_row": original, "utt_id": row["id"],
              "diagnostic_anchor": original in anchors,
              "source_group": row["id"].rsplit("_", 1)[0]}
             for i, (original, row) in enumerate(chosen)]
    manifest = {"source_tsv": args.input_tsv, "source_sha256": digest(args.input_tsv),
                "input_tsv": str(target), "input_sha256": digest(target),
                "pool_size": len(pool), "seed": args.seed, "count": len(cases),
                "selection_method": "diagnostic anchors plus round-robin shuffled source groups",
                "cases": cases}
    atomic_json(output / "cases.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def load_row(path, index):
    csv.field_size_limit(10_000_000)
    with Path(path).open(newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if i == index:
                return row
    raise ValueError(f"Row {index} missing")


def expected_settings(extra, seed, window):
    expected = {
        "future_source_window_mode": "fixed", "future_source_window_chunks": window,
        "future_source_anchor_max_words": 128, "future_join_mode": "space",
        "targeted_sampler_context": "source-and-target", "targeted_sampler_seed": seed,
        "targeted_prompt_version": "future_set_v2_two_groups",
        "targeted_fail_on_api_error": True, "sentence_end_completion": False,
        "sentence_end_boundary_mode": "literal", "sentence_end_punctuation": "off",
        "probe_prompt_order": "historical", "contrastive_notes": False,
    }
    tokens = iter(shlex.split(extra))
    for token in tokens:
        key = token.removeprefix("--").replace("-", "_")
        if key not in expected:
            raise ValueError(f"Unsupported pilot setting: {token}")
        if isinstance(expected[key], bool):
            expected[key] = True
        else:
            value = next(tokens)
            expected[key] = int(value) if isinstance(expected[key], int) else value
    return expected


def validate_result(result, row, expected):
    chunks = ast.literal_eval(row["src_trajectory"])
    if result.get("utt_id") != row["id"] or result.get("src_trajectory") != chunks:
        raise ValueError("Output ID/source trajectory mismatch")
    if result.get("source_full_text") != row["src_text"].strip():
        raise ValueError("Source text mismatch")
    deltas, actions = result.get("target_trajectory"), result.get("actions")
    if not isinstance(deltas, list) or not isinstance(actions, list) or not chunks:
        raise ValueError("Missing trajectories")
    if len(deltas) != len(chunks) or len(actions) != len(chunks):
        raise ValueError("Incomplete trajectory")
    if any(not isinstance(delta, str) or action != ("WRITE" if delta else "READ")
           for delta, action in zip(deltas, actions)):
        raise ValueError("Invalid actions or deltas")
    if result.get("prediction") != "".join(deltas):
        raise ValueError("Prediction does not equal committed deltas")
    recorded = dict(result.get("decoder_settings") or {})
    # Historical v2 outputs predate this field; they must never validate as v3.
    recorded.setdefault("targeted_prompt_version", "future_set_v2_two_groups")
    if recorded != expected:
        raise ValueError("Output was produced with different or unrecorded settings")


def guard_case(args):
    row = load_row(args.input_tsv, args.row_idx)
    task = Path(args.task_dir)
    files = list((task / "per_utt").glob("*.json"))
    valid = False
    error = "missing output"
    try:
        if len(files) != 1:
            raise ValueError(f"Expected one utterance JSON, found {len(files)}")
        result = json.loads(files[0].read_text())
        validate_result(result, row, expected_settings(args.extra_args, args.seed, args.window))
        if not list((task / "verbose").glob("verbose_*.log")):
            raise ValueError("Missing verbose evidence")
        valid = True
    except (ValueError, KeyError, TypeError, OSError) as exc:
        error = str(exc)
    if valid:
        atomic_json(task / "VERIFIED.json", {"utt_id": row["id"], "sha256": digest(files[0]),
                                            "settings": result["decoder_settings"]})
        print(f"[VERIFIED] row={args.row_idx} id={row['id']}")
        return 0
    if args.command == "prepare":
        # Keep interrupted/corrupt attempts intact, outside the active output view.
        if task.exists():
            for name in ("per_utt", "verbose", "DONE.txt", "VERIFIED.json"):
                item = task / name
                if item.exists():
                    archive = task / "attempts" / str(time.time_ns())
                    archive.mkdir(parents=True)
                    item.rename(archive / name)
        print(f"[RETRY REQUIRED] {error}")
        return 0
    print(f"[NOT COMPLETE] {error}", file=sys.stderr)
    return 1


def report_pilot(args):
    manifest = json.loads(Path(args.manifest).read_text())
    selection = json.loads((Path(manifest["selection_dir"]) / "cases.json").read_text())
    output = Path(manifest["output_root"])
    results, summary = [], {}
    for label, extra in manifest["variants"].items():
        scores, latencies, reasons = [], [], defaultdict(int)
        for case in selection["cases"]:
            task = output / f"row_{case['pilot_row']}" / label / "task_00"
            check = argparse.Namespace(command="validate", input_tsv=selection["input_tsv"],
                                       row_idx=case["pilot_row"], task_dir=str(task), extra_args=extra,
                                       seed=manifest["sampler_seed"], window=1)
            if guard_case(check):
                raise ValueError(f"Incomplete pilot: {label} {case['utt_id']}")
            result = json.loads(next((task / "per_utt").glob("*.json")).read_text())
            metrics = result.get("metrics", {})
            bleu, laal = metrics.get("bleu_char"), metrics.get("laal_text")
            for value, values in ((bleu, scores), (laal, latencies)):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(value)
            for event in result.get("boundary_audit", []):
                reasons[event["reason"]] += 1
            results.append({**case, "variant": label, "char_bleu": bleu, "word_laal": laal,
                            "read_steps": result["actions"].count("READ"),
                            "write_steps": result["actions"].count("WRITE"),
                            "prediction": result["prediction"], "reference": result.get("reference_text", ""),
                            "result_json": str(next((task / "per_utt").glob("*.json")))})
        summary[label] = {"valid_cases": len(selection["cases"]),
                          "mean_case_char_bleu": sum(scores) / len(scores) if scores else None,
                          "finite_bleu_cases": len(scores),
                          "mean_word_based_laal": sum(latencies) / len(latencies) if latencies else None,
                          "finite_laal_cases": len(latencies), "boundary_actions": dict(reasons)}
    atomic_json(output / "pilot_summary.json", {
        "note": "Synthesis pilot, not trained speech model evaluation. LAAL is word-based, not milliseconds. "
                "Full-bundle comparison cannot isolate each change. Inspect all cases, including regressions.",
        "summary": summary, "cases": results,
    })
    with (output / "case_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--input-tsv", required=True)
    select.add_argument("--selection-dir", required=True)
    select.add_argument("--pool-size", type=int, default=40000)
    select.add_argument("--count", type=int, default=50)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--anchor-rows", default="10,70,33")
    report = sub.add_parser("report")
    report.add_argument("--manifest", required=True)
    for command in ("validate", "prepare"):
        guard = sub.add_parser(command)
        guard.add_argument("--input-tsv", required=True)
        guard.add_argument("--row-idx", type=int, required=True)
        guard.add_argument("--task-dir", required=True)
        guard.add_argument("--extra-args", default="")
        guard.add_argument("--seed", type=int, default=1015)
        guard.add_argument("--window", type=int, default=1)
    args = parser.parse_args()
    if args.command == "select":
        freeze_selection(args)
    elif args.command == "report":
        report_pilot(args)
    else:
        raise SystemExit(guard_case(args))


if __name__ == "__main__":
    main()
