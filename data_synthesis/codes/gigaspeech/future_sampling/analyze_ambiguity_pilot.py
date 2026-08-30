#!/usr/bin/env python3
"""Build a case-by-case review report from ambiguity pilot outputs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


CHUNK_RE = re.compile(r"^Chunk (\d+)/(\d+)$")
FUTURE_RE = re.compile(r"^future\[\d+\] model=(\S+) mode=(\S+): (.+)$")
LEGACY_FUTURE_RE = re.compile(r"^future\[\d+\] \(([^)]+)\): (.+)$")
SELECTED_HEADER_RE = re.compile(
    r"^\[Selected candidates\].*\| model=(\S+) mode=(\S+) count=\d+$"
)
NUMBERED_FUTURE_RE = re.compile(r"^\d+\.\s+(.+)$")
META_MARKERS = (
    "the prompt", "user prompt", "assistant turn", "sampling mode",
    "unfinished english prefix", "let's think", "4-15 words", "<think>",
)


def parse_repr(raw: str) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw.strip()
    return str(value)


def lexical_diversity(texts: list[str]) -> float:
    if len(texts) < 2:
        return float("nan")
    distances: list[float] = []
    token_sets = [set(re.findall(r"[a-z0-9']+", text.lower())) for text in texts]
    for i, left in enumerate(token_sets):
        for right in token_sets[i + 1:]:
            union = left | right
            distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return mean(distances)


def parse_verbose(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    selected_group: tuple[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = CHUNK_RE.match(line)
        if match:
            current = {
                "index": int(match.group(1)),
                "total": int(match.group(2)),
                "futures": [],
                "commit": "",
                "action": "",
            }
            chunks.append(current)
            selected_group = None
            continue
        if current is None:
            continue
        if line.startswith("future_source_prefix: "):
            current["source_prefix"] = parse_repr(line.split(": ", 1)[1])
        elif line.startswith("committed_before: "):
            current["committed_before"] = parse_repr(line.split(": ", 1)[1])
        else:
            selected_match = SELECTED_HEADER_RE.match(line)
            if selected_match:
                selected_group = (selected_match.group(1), selected_match.group(2))
                continue
            if line.startswith("[Raw candidates]") or line.startswith("[Step "):
                selected_group = None
            numbered_match = NUMBERED_FUTURE_RE.match(line)
            if selected_group and numbered_match:
                current["futures"].append({
                    "model": selected_group[0],
                    "mode": selected_group[1],
                    "text": parse_repr(numbered_match.group(1)),
                })
                continue
            future_match = FUTURE_RE.match(line)
            if future_match:
                current["futures"].append({
                    "model": future_match.group(1),
                    "mode": future_match.group(2),
                    "text": parse_repr(future_match.group(3)),
                })
            else:
                legacy_match = LEGACY_FUTURE_RE.match(line)
                if legacy_match:
                    current["futures"].append({
                        "model": "unknown",
                        "mode": legacy_match.group(1).removeprefix("targeted_prefill_"),
                        "text": parse_repr(legacy_match.group(2)),
                    })
            if "commit_after_trim=" in line:
                current["commit"] = parse_repr(line.split("commit_after_trim=", 1)[1])
            elif line.startswith("-> "):
                current["action"] = line.split()[1]
                if "too few futures" in line:
                    current["too_few_futures"] = True
    return chunks


def analyze_case(result_path: Path, verbose_path: Path | None) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    chunks = parse_verbose(verbose_path) if verbose_path and verbose_path.is_file() else []
    futures = [future for chunk in chunks for future in chunk["futures"]]
    texts = [future["text"] for future in futures]
    normalized = [re.sub(r"\s+", " ", text.strip().lower()) for text in texts]
    meta = [text for text in texts if any(marker in text.lower() for marker in META_MARKERS)]
    overlong = [text for text in texts if len(text.split()) > 20]
    per_mode: dict[str, list[str]] = {}
    for future in futures:
        per_mode.setdefault(future["mode"], []).append(future["text"])
    metrics = result.get("metrics") or {}
    actions = result.get("actions") or []
    return {
        "utt_id": result.get("utt_id", result_path.stem),
        "source": result.get("source_full_text", ""),
        "prediction": result.get("prediction", ""),
        "reference": result.get("reference_text", ""),
        "metrics": metrics,
        "writes": actions.count("WRITE"),
        "reads": actions.count("READ"),
        "nonfinal_writes": sum(bool(x) for x in (result.get("target_trajectory") or [])[:-1]),
        "future_count": len(texts),
        "unique_future_ratio": len(set(normalized)) / len(normalized) if normalized else float("nan"),
        "mode_diversity": {mode: lexical_diversity(values) for mode, values in per_mode.items()},
        "meta_leakage": meta,
        "overlong": overlong,
        "too_few_chunks": sum(bool(chunk.get("too_few_futures")) for chunk in chunks),
        "chunks": chunks,
        "result_path": str(result_path),
        "verbose_path": str(verbose_path) if verbose_path else "",
    }


def fmt(value: Any) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.3f}"


def render_markdown(cases: list[dict[str, Any]], root: Path) -> str:
    bleu = [float(c["metrics"].get("bleu_char")) for c in cases
            if isinstance(c["metrics"].get("bleu_char"), (int, float))]
    laal = [float(c["metrics"].get("laal_text")) for c in cases
            if isinstance(c["metrics"].get("laal_text"), (int, float))]
    lines = [
        "# Ambiguity Pilot Review",
        "",
        f"Root: `{root}`",
        f"Cases: {len(cases)}",
        f"Mean char-BLEU: {fmt(mean(bleu) if bleu else float('nan'))}",
        f"Mean LAAL: {fmt(mean(laal) if laal else float('nan'))}",
        f"Cases with meta leakage: {sum(bool(c['meta_leakage']) for c in cases)}",
        f"Cases with too-few-future chunks: {sum(c['too_few_chunks'] > 0 for c in cases)}",
        "",
        "| Case | BLEU | LAAL | WRITE | Futures | Unique | Meta | Too few |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['utt_id']} | {fmt(case['metrics'].get('bleu_char'))} | "
            f"{fmt(case['metrics'].get('laal_text'))} | {case['writes']} | "
            f"{case['future_count']} | {fmt(case['unique_future_ratio'])} | "
            f"{len(case['meta_leakage'])} | {case['too_few_chunks']} |"
        )
    for case in cases:
        lines.extend([
            "", f"## {case['utt_id']}", "",
            f"Source: {case['source']}", "",
            f"Prediction: {case['prediction']}", "",
            f"Reference: {case['reference']}", "",
            f"Metrics: `{json.dumps(case['metrics'], ensure_ascii=False)}`",
        ])
        for chunk in case["chunks"]:
            lines.extend([
                "", f"### Chunk {chunk['index']}/{chunk['total']}", "",
                f"Source prefix: `{chunk.get('source_prefix', '')}`", "",
                "| Model | Mode | Future |", "|---|---|---|",
            ])
            for future in chunk["futures"]:
                text = future["text"].replace("|", "\\|")
                lines.append(f"| {future.get('model', 'unknown')} | {future['mode']} | {text} |")
            lines.append("")
            lines.append(f"Decision: `{chunk.get('action', '')}`; commit: `{chunk.get('commit', '')}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    root = Path(args.pilot_root)
    output = Path(args.output_dir) if args.output_dir else root / "review"
    output.mkdir(parents=True, exist_ok=True)
    verbose_by_id = {
        path.stem.removeprefix("verbose_"): path
        for path in root.glob("task_*/verbose/verbose_*.log")
    }
    cases = []
    for result_path in sorted(root.glob("task_*/per_utt/*.json")):
        cases.append(analyze_case(result_path, verbose_by_id.get(result_path.stem)))
    (output / "pilot_review.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "pilot_review.md").write_text(render_markdown(cases, root), encoding="utf-8")
    print(f"cases={len(cases)}")
    print(f"json={output / 'pilot_review.json'}")
    print(f"markdown={output / 'pilot_review.md'}")


if __name__ == "__main__":
    main()
