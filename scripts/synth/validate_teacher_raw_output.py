#!/usr/bin/env python3
"""Validate raw EAST or Simul-MuST-C teacher-generation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EAST_LEVELS = ("low_latency", "medium_latency", "high_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("east", "simul-must-c"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument(
        "--allow-recorded-errors",
        action="store_true",
        help="Count generator error records as complete inputs but skip schema validation for them",
    )
    return parser.parse_args()


def require_text(value: Any, label: str, path: Path) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {label} must be non-empty text")


def validate_east(data: dict[str, Any], path: Path) -> None:
    for level in EAST_LEVELS:
        trajectory = data.get(level)
        if not isinstance(trajectory, dict):
            raise ValueError(f"{path}: missing object {level}")
        english = trajectory.get("English")
        chinese = trajectory.get("Chinese")
        if not isinstance(english, list) or not isinstance(chinese, list):
            raise ValueError(f"{path}: {level} must contain English and Chinese lists")
        if not english or len(english) != len(chinese):
            raise ValueError(f"{path}: {level} has empty or unpaired segments")
        for index, (source, target) in enumerate(zip(english, chinese)):
            require_text(source, f"{level}.English[{index}]", path)
            require_text(target, f"{level}.Chinese[{index}]", path)


def validate_simul(data: dict[str, Any], path: Path) -> None:
    pairs = data.get("segmented_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: segmented_pairs must be a non-empty list")
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{path}: segmented_pairs[{index}] must have two items")
        require_text(pair[0], f"segmented_pairs[{index}][0]", path)
        require_text(pair[1], f"segmented_pairs[{index}][1]", path)
    require_text(data.get("output"), "output", path)


def main() -> None:
    args = parse_args()
    files = sorted(args.output_root.glob("*.json"))
    if len(files) != args.expected_count:
        raise ValueError(
            f"{args.output_root}: found {len(files)} JSON files, "
            f"expected {args.expected_count}"
        )

    utterance_ids: set[str] = set()
    recorded_errors = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level value must be an object")
        utterance_id = str(data.get("utt_id", "")).strip()
        if not utterance_id or utterance_id in utterance_ids:
            raise ValueError(f"{path}: missing or duplicate utt_id {utterance_id!r}")
        utterance_ids.add(utterance_id)
        if data.get("error") or data.get("errors"):
            if not args.allow_recorded_errors:
                raise ValueError(f"{path}: generator recorded an error")
            recorded_errors += 1
            continue
        if args.method == "east":
            validate_east(data, path)
        else:
            validate_simul(data, path)

    print(
        f"Validated {len(files)} {args.method} outputs under {args.output_root} "
        f"with zero schema errors and {recorded_errors} recorded generator errors"
    )


if __name__ == "__main__":
    main()
