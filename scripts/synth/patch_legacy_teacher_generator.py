#!/usr/bin/env python3
"""Create a Qwen3.6-compatible runtime copy of a legacy vLLM generator."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_IMPORT = "from vllm.sampling_params import GuidedDecodingParams"
NEW_IMPORT = "from vllm.sampling_params import StructuredOutputsParams"
OLD_SCHEMA_ARGUMENT = "guided_decoding=GuidedDecodingParams(json="
NEW_SCHEMA_ARGUMENT = "structured_outputs=StructuredOutputsParams(json="
MEMORY_ARGUMENT = "        gpu_memory_utilization=0.90,\n"
SEQUENCE_LIMIT = "        max_num_seqs=64,\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one {label} occurrence, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    source = args.input.read_text(encoding="utf-8")
    source = replace_once(source, OLD_IMPORT, NEW_IMPORT, "guided-decoding import")
    source = replace_once(
        source,
        OLD_SCHEMA_ARGUMENT,
        NEW_SCHEMA_ARGUMENT,
        "guided-decoding argument",
    )
    source = replace_once(
        source,
        MEMORY_ARGUMENT,
        MEMORY_ARGUMENT + SEQUENCE_LIMIT,
        "gpu-memory argument",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(f"Wrote Qwen3.6 vLLM runtime copy: {args.output}")


if __name__ == "__main__":
    main()
