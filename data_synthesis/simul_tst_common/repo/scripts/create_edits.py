#!/usr/bin/env python3

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from tqdm import tqdm


FORMAT_VERSION = 3


def line_hash(line: str) -> str:
    """
    Create a non-readable identifier for an original line.

    The newline is excluded so that LF and CRLF differences do not affect
    matching.
    """
    normalized = line.rstrip("\r\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_line_ending(line: str) -> tuple[str, str]:
    """Separate the textual content from its line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"

    if line.endswith("\n"):
        return line[:-1], "\n"

    if line.endswith("\r"):
        return line[:-1], "\r"

    return line, ""


def create_character_edits(
    original: str,
    corrected: str,
) -> list[dict]:
    """
    Create character-level operations for one line.

    Character positions refer to the original line, excluding its newline.
    Equal spans are not stored.
    """
    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=original,
        b=corrected,
        autojunk=False,
    )

    edits: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        edit = {
            "op": tag,
            "start": i1,
            "end": i2,
        }

        if tag in {"insert", "replace"}:
            edit["text"] = corrected[j1:j2]

        edits.append(edit)

    return edits


def pair_replaced_lines(
    original_lines: list[str],
    corrected_lines: list[str],
) -> list[tuple[str | None, str | None]]:
    """
    Pair lines in a replaced block.

    When the block contains different numbers of original and corrected lines,
    unmatched lines are represented by None.
    """
    pairs: list[tuple[str | None, str | None]] = []
    shared_length = min(len(original_lines), len(corrected_lines))

    for index in range(shared_length):
        pairs.append(
            (
                original_lines[index],
                corrected_lines[index],
            )
        )

    for line in original_lines[shared_length:]:
        pairs.append((line, None))

    for line in corrected_lines[shared_length:]:
        pairs.append((None, line))

    return pairs


def create_edit_package(
    original_lines: list[str],
    corrected_lines: list[str],
    use_autojunk: bool,
) -> list[dict]:
    """
    Align files by line, then create character-level edits for each changed line.
    """
    original_hashes = [
        line_hash(line)
        for line in tqdm(
            original_lines,
            desc="Hashing original lines",
            unit="line",
        )
    ]

    corrected_hashes = [
        line_hash(line)
        for line in tqdm(
            corrected_lines,
            desc="Hashing corrected lines",
            unit="line",
        )
    ]

    print("Aligning files at line level...")

    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=original_hashes,
        b=corrected_hashes,
        autojunk=use_autojunk,
    )

    opcodes = matcher.get_opcodes()
    operations: list[dict] = []

    for tag, i1, i2, j1, j2 in tqdm(
        opcodes,
        desc="Creating per-line character edits",
        unit="block",
    ):
        if tag == "equal":
            continue

        if tag == "replace":
            original_block = original_lines[i1:i2]
            corrected_block = corrected_lines[j1:j2]

            pairs = pair_replaced_lines(
                original_block,
                corrected_block,
            )

            previous_anchor_hash: str | None = (
                original_hashes[i1 - 1] if i1 > 0 else None
            )

            for original_line, corrected_line in pairs:
                # One original line corresponds to one corrected line.
                if original_line is not None and corrected_line is not None:
                    original_text, original_ending = split_line_ending(
                        original_line
                    )
                    corrected_text, corrected_ending = split_line_ending(
                        corrected_line
                    )

                    character_edits = create_character_edits(
                        original_text,
                        corrected_text,
                    )

                    operation = {
                        "op": "edit_line",
                        "line_hash": line_hash(original_line),
                        "character_edits": character_edits,
                    }

                    # Store a newline change only when necessary.
                    if original_ending != corrected_ending:
                        operation["line_ending"] = corrected_ending

                    operations.append(operation)
                    previous_anchor_hash = line_hash(original_line)

                # Original line has been removed.
                elif original_line is not None:
                    operations.append(
                        {
                            "op": "delete_line",
                            "line_hash": line_hash(original_line),
                        }
                    )
                    previous_anchor_hash = line_hash(original_line)

                # Corrected file contains an additional line.
                elif corrected_line is not None:
                    if previous_anchor_hash is not None:
                        operations.append(
                            {
                                "op": "insert_line_after",
                                "anchor_hash": previous_anchor_hash,
                                "text": corrected_line,
                            }
                        )
                    elif i1 < len(original_lines):
                        operations.append(
                            {
                                "op": "insert_line_before",
                                "anchor_hash": original_hashes[i1],
                                "text": corrected_line,
                            }
                        )
                    else:
                        operations.append(
                            {
                                "op": "append_line",
                                "text": corrected_line,
                            }
                        )

        elif tag == "delete":
            for original_line in original_lines[i1:i2]:
                operations.append(
                    {
                        "op": "delete_line",
                        "line_hash": line_hash(original_line),
                    }
                )

        elif tag == "insert":
            inserted_lines = corrected_lines[j1:j2]

            if i1 > 0:
                anchor_hash = original_hashes[i1 - 1]

                for inserted_line in inserted_lines:
                    operations.append(
                        {
                            "op": "insert_line_after",
                            "anchor_hash": anchor_hash,
                            "text": inserted_line,
                        }
                    )

            elif i1 < len(original_lines):
                anchor_hash = original_hashes[i1]

                for inserted_line in inserted_lines:
                    operations.append(
                        {
                            "op": "insert_line_before",
                            "anchor_hash": anchor_hash,
                            "text": inserted_line,
                        }
                    )

            else:
                for inserted_line in inserted_lines:
                    operations.append(
                        {
                            "op": "append_line",
                            "text": inserted_line,
                        }
                    )

        else:
            raise ValueError(f"Unsupported line-level operation: {tag}")

    return operations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create character-level edit operations independently for each "
            "changed line."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
        type=Path,
        help="Original input file.",
    )
    parser.add_argument(
        "--corrected",
        required=True,
        type=Path,
        help="Corrected file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON edit package.",
    )
    parser.add_argument(
        "--no-autojunk",
        action="store_true",
        help=(
            "Disable SequenceMatcher's autojunk heuristic during line-level "
            "alignment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.original.is_file():
        raise FileNotFoundError(
            f"Original file not found: {args.original}"
        )

    if not args.corrected.is_file():
        raise FileNotFoundError(
            f"Corrected file not found: {args.corrected}"
        )

    print("Reading input files...")

    original_lines = args.original.read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)

    corrected_lines = args.corrected.read_text(
        encoding="utf-8"
    ).splitlines(keepends=True)

    print(f"Original lines:  {len(original_lines):,}")
    print(f"Corrected lines: {len(corrected_lines):,}")

    operations = create_edit_package(
        original_lines=original_lines,
        corrected_lines=corrected_lines,
        use_autojunk=not args.no_autojunk,
    )

    package = {
        "format_version": FORMAT_VERSION,
        "encoding": "utf-8",
        "matching_unit": "line",
        "editing_unit": "character",
        "operations": operations,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Writing edit package...")

    args.output.write_text(
        json.dumps(
            package,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    character_edit_count = sum(
        len(operation.get("character_edits", []))
        for operation in operations
    )

    print(f"Created edit package: {args.output}")
    print(f"Line operations: {len(operations):,}")
    print(f"Character operations: {character_edit_count:,}")


if __name__ == "__main__":
    main()