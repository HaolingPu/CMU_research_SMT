#!/usr/bin/env python3
"""Build a source-matched 12.5K manifest for a translation-teacher ablation.

The existing baseline manifest supplies audio chunks, source examples, chunk
multipliers, and message structure. Only assistant targets are replaced with
targets from a newly generated and quality-filtered trajectory tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EAST_LATENCIES = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("east-even", "simul-must-c"))
    parser.add_argument("--final-root", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--template-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=12_500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def row_key(row: dict[str, Any], method: str) -> tuple[str, str]:
    audios = row.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError("Manifest row has no audio chunks")

    parts = Path(str(audios[0])).parts
    multiplier_index = next(
        (index for index, part in enumerate(parts) if part.startswith("multiplier_")),
        None,
    )
    if multiplier_index is None or multiplier_index < 2:
        raise ValueError(f"Cannot recover utterance id from audio path: {audios[0]}")

    utt_id = f"{parts[multiplier_index - 2]}_{parts[multiplier_index - 1]}"
    if method == "east-even":
        if multiplier_index < 3 or parts[multiplier_index - 3] not in EAST_LATENCIES:
            raise ValueError(f"Cannot recover EAST latency from audio path: {audios[0]}")
        latency = parts[multiplier_index - 3]
    else:
        latency = "offline"
    return utt_id, latency


def load_trajectories(final_root: Path) -> tuple[dict[tuple[str, str], list[str]], int]:
    trajectories: dict[tuple[str, str], list[str]] = {}
    duplicate_count = 0
    files = sorted(final_root.rglob("*_latency.jsonl"))
    if not files:
        raise ValueError(f"No *_latency.jsonl files found under {final_root}")

    for path in files:
        for row in iter_jsonl(path):
            utt_id = str(row.get("utt_id", "")).strip()
            latency = str(row.get("latency", "")).strip()
            targets = row.get("target")
            if not utt_id or latency not in (*EAST_LATENCIES, "offline"):
                continue
            if not isinstance(targets, list) or not targets:
                continue
            key = (utt_id, latency)
            if key in trajectories:
                duplicate_count += 1
                continue
            trajectories[key] = [str(target) for target in targets]
    return trajectories, duplicate_count


def replace_targets(
    row: dict[str, Any], targets: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        multiplier = int(row.get("multiplier"))
    except (TypeError, ValueError):
        return None, "invalid_multiplier"
    if multiplier <= 0:
        return None, "invalid_multiplier"

    grouped_targets = [
        "".join(targets[index : index + multiplier])
        for index in range(0, len(targets), multiplier)
    ]
    audios = row.get("audios")
    if not isinstance(audios, list) or len(audios) != len(grouped_targets):
        return None, "audio_target_length_mismatch"

    updated = copy.deepcopy(row)
    messages = updated.get("messages")
    if not isinstance(messages, list):
        return None, "invalid_messages"
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if len(assistant_messages) != len(grouped_targets):
        return None, "assistant_target_length_mismatch"
    for message, target in zip(assistant_messages, grouped_targets):
        message["content"] = target
    return updated, None


def build_valid_pool(
    pool_manifest: Path,
    method: str,
    trajectories: dict[tuple[str, str], list[str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str], int]:
    valid: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    total = 0
    for row in iter_jsonl(pool_manifest):
        total += 1
        try:
            key = row_key(row, method)
        except ValueError:
            rejected["invalid_pool_key"] += 1
            continue
        targets = trajectories.get(key)
        if targets is None:
            rejected["not_in_filtered_trajectories"] += 1
            continue
        updated, reason = replace_targets(row, targets)
        if updated is None:
            rejected[reason or "unknown"] += 1
            continue
        if key in valid:
            rejected["duplicate_pool_key"] += 1
            continue
        valid[key] = updated
    return valid, rejected, total


def load_template_keys(path: Path, method: str) -> list[tuple[str, str]]:
    keys = [row_key(row, method) for row in iter_jsonl(path)]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Template manifest contains duplicate method keys: {path}")
    return keys


def choose_keys(
    method: str,
    sample_size: int,
    seed: int,
    valid: dict[tuple[str, str], dict[str, Any]],
    template_keys: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    stats: dict[str, dict[str, int]] = {}

    if method == "east-even":
        per_group = sample_size // 3
        required = {
            "low": per_group,
            "medium": per_group,
            "high": per_group + sample_size - (per_group * 3),
        }
        groups = EAST_LATENCIES
    else:
        required = {"offline": sample_size}
        groups = ("offline",)

    template_set = set(template_keys)
    for latency in groups:
        target_count = required[latency]
        matched = [key for key in template_keys if key[1] == latency and key in valid]
        if len(matched) > target_count:
            matched = matched[:target_count]
        remaining = target_count - len(matched)
        candidates = [
            key
            for key in valid
            if key[1] == latency and key not in template_set and key not in selected
        ]
        if len(candidates) < remaining:
            raise ValueError(
                f"Insufficient valid {latency} examples: need {target_count}, "
                f"matched={len(matched)}, replacement_pool={len(candidates)}"
            )
        replacements = rng.sample(candidates, remaining)
        selected.extend(matched)
        selected.extend(replacements)
        stats[latency] = {
            "required": target_count,
            "template_matched": len(matched),
            "replacements": len(replacements),
            "valid_pool": sum(1 for key in valid if key[1] == latency),
        }

    if len(selected) != sample_size or len(selected) != len(set(selected)):
        raise AssertionError("Selection did not produce the requested number of unique keys")
    rng.shuffle(selected)
    return selected, stats


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    trajectories, duplicate_trajectories = load_trajectories(args.final_root)
    valid, rejected, pool_total = build_valid_pool(
        args.pool_manifest, args.method, trajectories
    )
    template_keys = load_template_keys(args.template_manifest, args.method)
    if len(template_keys) != args.sample_size:
        raise ValueError(
            f"Template has {len(template_keys)} rows; expected exactly {args.sample_size}"
        )
    selected_keys, selection_stats = choose_keys(
        args.method, args.sample_size, args.seed, valid, template_keys
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for key in selected_keys:
            handle.write(json.dumps(valid[key], ensure_ascii=False) + "\n")
    temporary_output.replace(args.output)

    report = {
        "method": args.method,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "final_root": str(args.final_root),
        "pool_manifest": str(args.pool_manifest),
        "template_manifest": str(args.template_manifest),
        "output": str(args.output),
        "filtered_trajectory_keys": len(trajectories),
        "duplicate_trajectory_keys": duplicate_trajectories,
        "pool_rows": pool_total,
        "valid_retargeted_pool_rows": len(valid),
        "rejected_pool_rows": dict(sorted(rejected.items())),
        "selection": selection_stats,
        "template_matches_total": sum(
            group["template_matched"] for group in selection_stats.values()
        ),
        "replacement_rows_total": sum(
            group["replacements"] for group in selection_stats.values()
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote exactly {args.sample_size} rows to {args.output}")


if __name__ == "__main__":
    main()
