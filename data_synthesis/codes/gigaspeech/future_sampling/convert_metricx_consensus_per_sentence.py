#!/usr/bin/env python3
"""Convert consensus decoding JSON outputs to per-sentence MetricX QE input.

Each utterance is split into its `src_text_full` sub-sentences; the Chinese
prediction is re-grouped by chunk->unit alignment (chunks are assigned to
units by their midpoint in the joined source text). One jsonl row is emitted
per (utt_id, sentence_idx). Downstream MetricX runs QE on each row, and the
companion filter keeps an instance only when EVERY sentence passes threshold.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def append_text_continuation(prefix: str, continuation: str) -> str:
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if prefix[-1].isspace() or continuation[0].isspace():
        return prefix + continuation
    if continuation[0] in ",.!?;:)]}\"'":
        return prefix + continuation
    return prefix + " " + continuation


def parse_source_units(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x or "") for x in raw]
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(x or "") for x in parsed]
    return [str(parsed)]


def parse_trajectory(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x or "") for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(x or "") for x in parsed]
    return []


def assign_chunks_to_units(chunks: List[str], source_units: List[str]) -> List[int]:
    """Map each chunk index to the src_text_full unit it belongs to.

    Uses midpoint position in the joined source text so boundary-straddling
    chunks are assigned to whichever unit holds the majority of their chars.
    """
    if not source_units:
        return [0] * len(chunks)

    unit_ends: List[int] = []
    running = ""
    for unit in source_units:
        running = append_text_continuation(running, str(unit or ""))
        unit_ends.append(len(running))

    chunk_to_unit: List[int] = []
    running_chunks = ""
    for chunk in chunks:
        before = len(running_chunks)
        running_chunks = append_text_continuation(running_chunks, str(chunk or ""))
        after = len(running_chunks)
        midpoint = (before + after) / 2.0
        unit_idx = len(source_units) - 1
        for i, u_end in enumerate(unit_ends):
            if midpoint <= u_end:
                unit_idx = i
                break
        chunk_to_unit.append(unit_idx)
    return chunk_to_unit


def collect_json_files(root: str) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".json"):
                out.append(Path(dirpath) / fn)
    return sorted(out)


def iter_per_sentence_examples(root: str) -> Iterable[Dict[str, object]]:
    for path in collect_json_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or "error" in data:
            continue

        utt_id = str(data.get("utt_id", "")).strip()
        source_units = parse_source_units(data.get("src_text_full"))
        chunks = parse_trajectory(data.get("src_trajectory"))
        target_deltas = data.get("target_trajectory") or []
        if not utt_id or not source_units or not chunks:
            continue
        if len(target_deltas) != len(chunks):
            continue

        chunk_to_unit = assign_chunks_to_units(chunks, source_units)

        per_unit_parts: List[List[str]] = [[] for _ in source_units]
        for i, assigned in enumerate(chunk_to_unit):
            delta = str(target_deltas[i] or "")
            if delta:
                per_unit_parts[assigned].append(delta)

        total_sentences = len(source_units)
        for u_idx, unit in enumerate(source_units):
            hypothesis = "".join(per_unit_parts[u_idx]).strip()
            yield {
                "source": str(unit or "").strip(),
                "hypothesis": hypothesis,
                "reference": "",
                "metadata": {
                    "utt_id": utt_id,
                    "sentence_idx": u_idx,
                    "total_sentences": total_sentences,
                    "latency": "future_sampling",
                    "stream_json": str(path),
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert consensus JSON outputs to per-sentence MetricX QE input.jsonl."
    )
    parser.add_argument("--input-dir", required=True, help="Consensus experiment directory.")
    parser.add_argument("--output", required=True, help="Output MetricX input jsonl.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    utts = set()
    empty_hyp = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for item in iter_per_sentence_examples(args.input_dir):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows += 1
            utts.add(item["metadata"]["utt_id"])
            if not item["hypothesis"]:
                empty_hyp += 1

    print("===== Convert Consensus -> Per-Sentence MetricX Input =====")
    print(f"Input dir          : {args.input_dir}")
    print(f"Output             : {output_path}")
    print(f"Utterances         : {len(utts)}")
    print(f"Sentence rows      : {rows}")
    print(f"Empty-hypothesis   : {empty_hyp}")


if __name__ == "__main__":
    main()
