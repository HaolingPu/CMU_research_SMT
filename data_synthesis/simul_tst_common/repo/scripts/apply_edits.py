#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


SUPPORTED_FORMAT_VERSION = 3


def line_hash(line: str) -> str:
    """
    Create the same line identifier used by create_edits.py.

    Line endings are excluded so that LF and CRLF differences do not affect
    line matching.
    """
    normalized = line.rstrip("\r\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_line_ending(line: str) -> tuple[str, str]:
    """Separate line content from its line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"

    if line.endswith("\n"):
        return line[:-1], "\n"

    if line.endswith("\r"):
        return line[:-1], "\r"

    return line, ""


def apply_character_edits(
    original_text: str,
    character_edits: list[dict],
) -> str:
    """
    Apply character-level edits to one line.

    Positions refer to the original line, so operations are applied from right
    to left to prevent earlier character offsets from shifting.
    """
    result = original_text

    sorted_edits = sorted(
        character_edits,
        key=lambda edit: (edit["start"], edit["end"]),
        reverse=True,
    )

    previous_start = len(original_text) + 1

    for edit in sorted_edits:
        operation = edit["op"]
        start = edit["start"]
        end = edit["end"]

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"Character positions must be integers: {edit}"
            )

        if not (0 <= start <= end <= len(original_text)):
            raise ValueError(
                "Invalid character-edit range: "
                f"start={start}, end={end}, "
                f"line_length={len(original_text)}"
            )

        if end > previous_start:
            raise ValueError(
                "Overlapping character edits were found in one line."
            )

        previous_start = start

        if operation == "delete":
            replacement = ""

        elif operation in {"insert", "replace"}:
            replacement = edit.get("text", "")

        else:
            raise ValueError(
                f"Unsupported character-level operation: {operation}"
            )

        result = result[:start] + replacement + result[end:]

    return result


def build_hash_index(
    lines: list[str],
) -> dict[str, list[int]]:
    """
    Map each line hash to all matching positions.

    Multiple positions are retained because identical lines can occur more
    than once.
    """
    index: dict[str, list[int]] = defaultdict(list)

    for position, line in tqdm(
        enumerate(lines),
        total=len(lines),
        desc="Indexing input lines",
        unit="line",
    ):
        index[line_hash(line)].append(position)

    return dict(index)


def choose_position(
    positions: list[int],
    used_positions: set[int],
) -> int | None:
    """
    Select the first occurrence that has not already been used.

    This prevents several operations for identical lines from all modifying
    the same occurrence.
    """
    for position in positions:
        if position not in used_positions:
            return position

    return None


def resolve_operations(
    lines: list[str],
    operations: list[dict],
) -> tuple[list[dict], int]:
    """
    Resolve hash-based operations to positions in the original input.

    All positions are resolved before modifying the file. This prevents line
    insertions and deletions from invalidating later positions.
    """
    hash_index = build_hash_index(lines)

    resolved: list[dict] = []
    skipped = 0
    used_positions: set[int] = set()

    for operation in tqdm(
        operations,
        desc="Locating edit operations",
        unit="operation",
    ):
        operation_type = operation["op"]

        if operation_type in {"edit_line", "delete_line"}:
            target_hash = operation["line_hash"]
            positions = hash_index.get(target_hash, [])

            position = choose_position(
                positions,
                used_positions,
            )

            if position is None:
                skipped += 1
                continue

            used_positions.add(position)

            if operation_type == "edit_line":
                resolved.append(
                    {
                        "op": "edit_line",
                        "position": position,
                        "character_edits": operation.get(
                            "character_edits",
                            [],
                        ),
                        "line_ending": operation.get("line_ending"),
                    }
                )

            else:
                resolved.append(
                    {
                        "op": "delete_line",
                        "position": position,
                    }
                )

        elif operation_type in {
            "insert_line_before",
            "insert_line_after",
        }:
            anchor_hash = operation["anchor_hash"]
            positions = hash_index.get(anchor_hash, [])

            # Do not mark insertion anchors as used. Several inserted lines may
            # intentionally use the same anchor.
            if not positions:
                skipped += 1
                continue

            anchor_position = positions[0]

            if operation_type == "insert_line_before":
                insertion_position = anchor_position
            else:
                insertion_position = anchor_position + 1

            resolved.append(
                {
                    "op": operation_type,
                    "position": insertion_position,
                    "text": operation["text"],
                }
            )

        elif operation_type == "append_line":
            resolved.append(
                {
                    "op": "append_line",
                    "position": len(lines),
                    "text": operation["text"],
                }
            )

        else:
            raise ValueError(
                f"Unsupported line-level operation: {operation_type}"
            )

    return resolved, skipped


def apply_resolved_operations(
    original_lines: list[str],
    resolved_operations: list[dict],
) -> list[str]:
    """
    Apply resolved line operations from the bottom of the file upward.
    """
    result = original_lines.copy()

    # Python's sort is stable. The secondary priority controls operations that
    # share the same position.
    operation_priority = {
        "append_line": 0,
        "insert_line_after": 0,
        "insert_line_before": 0,
        "edit_line": 1,
        "delete_line": 2,
    }

    resolved_operations.sort(
        key=lambda operation: (
            operation["position"],
            operation_priority[operation["op"]],
        ),
        reverse=True,
    )

    for operation in tqdm(
        resolved_operations,
        desc="Applying edit operations",
        unit="operation",
    ):
        operation_type = operation["op"]
        position = operation["position"]

        if operation_type == "edit_line":
            original_text, original_ending = split_line_ending(
                result[position]
            )

            corrected_text = apply_character_edits(
                original_text,
                operation["character_edits"],
            )

            # None means that create_edits.py did not request a change.
            requested_ending = operation.get("line_ending")

            if requested_ending is None:
                corrected_ending = original_ending
            else:
                corrected_ending = requested_ending

            result[position] = corrected_text + corrected_ending

        elif operation_type == "delete_line":
            del result[position]

        elif operation_type in {
            "insert_line_before",
            "insert_line_after",
            "append_line",
        }:
            result.insert(position, operation["text"])

        else:
            raise ValueError(
                f"Unsupported resolved operation: {operation_type}"
            )

    return result


def apply_edits(
    lines: list[str],
    operations: list[dict],
) -> tuple[list[str], int, int]:
    resolved_operations, skipped = resolve_operations(
        lines,
        operations,
    )

    result = apply_resolved_operations(
        lines,
        resolved_operations,
    )

    applied = len(resolved_operations)

    return result, applied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply per-line character-level edit operations created by "
            "create_edits.py format version 3."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file to modify.",
    )
    parser.add_argument(
        "--edits",
        required=True,
        type=Path,
        help="JSON edit package created by create_edits.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the reproduced output file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    if not args.edits.is_file():
        raise FileNotFoundError(
            f"Edit package not found: {args.edits}"
        )

    print("Reading input file...")

    lines = args.input.read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)

    print("Reading edit package...")

    package = json.loads(
        args.edits.read_text(encoding="utf-8")
    )

    version = package.get("format_version")

    if version != SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported edit format version: {version}. "
            f"Expected version {SUPPORTED_FORMAT_VERSION}."
        )

    operations = package.get("operations")

    if not isinstance(operations, list):
        raise ValueError(
            "The edit package does not contain a valid 'operations' list."
        )

    result, applied, skipped = apply_edits(
        lines,
        operations,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        "".join(result),
        encoding="utf-8",
    )

    print()
    print(f"Created output: {args.output}")
    print(f"Applied operations: {applied:,}")
    print(f"Skipped operations: {skipped:,}")

    if skipped:
        print(
            "Some operations were skipped because their corresponding "
            "lines or insertion anchors were not found in the input."
        )


if __name__ == "__main__":
    main()