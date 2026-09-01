#!/usr/bin/env python3
"""Verify that an external decode produced exactly its assigned TSV slice."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--row-offset", type=int, default=20004)
    parser.add_argument("--num-rows", type=int, default=19996)
    parser.add_argument("--id-column", default="id")
    return parser.parse_args()


def read_expected_ids(args: argparse.Namespace) -> list[str]:
    expected: list[str] = []
    end = args.row_offset + args.num_rows
    with open(args.input_tsv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or args.id_column not in reader.fieldnames:
            raise ValueError(
                f"missing id column {args.id_column!r}; columns={reader.fieldnames}"
            )
        for row_index, row in enumerate(reader):
            if row_index >= end:
                break
            if row_index >= args.row_offset:
                expected.append(str(row[args.id_column]))
    if len(expected) != args.num_rows:
        raise ValueError(
            f"TSV contains only {len(expected)} rows in requested "
            f"slice {args.row_offset}:{end}"
        )
    return expected


def read_actual_ids(output_root: str) -> tuple[list[str], list[str]]:
    paths = sorted(
        glob.glob(os.path.join(os.path.abspath(output_root), "task_*", "per_utt", "*.json"))
    )
    ids: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            ids.append(str(payload["utt_id"]))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    return ids, errors


def main() -> int:
    args = parse_args()
    expected = read_expected_ids(args)
    actual, read_errors = read_actual_ids(args.output_root)
    expected_set = set(expected)
    actual_counts = Counter(actual)
    actual_set = set(actual_counts)
    duplicates = sorted(key for key, count in actual_counts.items() if count > 1)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    print(f"expected={len(expected)} actual_files={len(actual)} unique_actual={len(actual_set)}")
    print(
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"duplicate_ids={len(duplicates)} unreadable={len(read_errors)}"
    )
    for label, values in (
        ("missing", missing),
        ("unexpected", unexpected),
        ("duplicate_ids", duplicates),
        ("unreadable", read_errors),
    ):
        if values:
            print(f"{label}_examples={values[:10]}")

    valid = (
        len(actual) == args.num_rows
        and not missing
        and not unexpected
        and not duplicates
        and not read_errors
    )
    print("VERIFIED" if valid else "FAILED")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
