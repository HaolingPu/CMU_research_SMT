#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ASCII_RE = re.compile(r"[A-Za-z]{2,}")


def iter_result_jsons(experiment_dir: Path) -> Iterable[Path]:
    task_paths = sorted(experiment_dir.glob("task_*/*.json"))
    if task_paths:
        yield from task_paths
        return
    flat_paths = sorted(
        path
        for path in experiment_dir.glob("*.json")
        if path.name != "summary.json"
    )
    yield from flat_paths


def iter_verbose_logs(experiment_dir: Path) -> Iterable[Path]:
    task_logs = sorted(experiment_dir.glob("task_*/verbose/verbose_*.log"))
    if task_logs:
        yield from task_logs
        return
    flat_logs = sorted(experiment_dir.glob("verbose/verbose_*.log"))
    yield from flat_logs


def _extract_nonempty_list_blocks(text: str, key: str) -> List[str]:
    pattern = re.compile(
        rf'"{re.escape(key)}": \[(.*?)\]',
        flags=re.DOTALL,
    )
    blocks: List[str] = []
    for match in pattern.finditer(text):
        inner = match.group(1).strip()
        if inner:
            blocks.append(inner)
    return blocks


def analyze_predictions(json_paths: Iterable[Path]) -> Dict[str, int]:
    stats = {
        "total_results": 0,
        "prediction_has_ascii": 0,
        "prediction_has_replacement_char": 0,
    }
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prediction = str(data.get("prediction", ""))
        stats["total_results"] += 1
        if ASCII_RE.search(prediction):
            stats["prediction_has_ascii"] += 1
        if "\ufffd" in prediction or "�" in prediction:
            stats["prediction_has_replacement_char"] += 1
    return stats


def analyze_verbose_logs(log_paths: Iterable[Path]) -> Dict[str, int]:
    stats = {
        "total_verbose_logs": 0,
        "logs_with_filtered_disallowed_tokens": 0,
        "logs_with_removed_disallowed_tokens": 0,
        "logs_with_removed_tail_tokens": 0,
        "total_filtered_disallowed_occurrences": 0,
        "total_removed_disallowed_occurrences": 0,
        "total_removed_tail_occurrences": 0,
        "legacy_logs_with_filtered_special_tokens": 0,
        "legacy_logs_with_removed_token_ids": 0,
    }

    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        stats["total_verbose_logs"] += 1

        filtered_blocks = _extract_nonempty_list_blocks(text, "filtered_disallowed_tokens")
        if filtered_blocks:
            stats["logs_with_filtered_disallowed_tokens"] += 1
            stats["total_filtered_disallowed_occurrences"] += len(filtered_blocks)

        removed_disallowed_blocks = _extract_nonempty_list_blocks(text, "removed_disallowed_tokens")
        if removed_disallowed_blocks:
            stats["logs_with_removed_disallowed_tokens"] += 1
            stats["total_removed_disallowed_occurrences"] += len(removed_disallowed_blocks)

        removed_tail_blocks = _extract_nonempty_list_blocks(text, "removed_tail_token_ids")
        if removed_tail_blocks:
            stats["logs_with_removed_tail_tokens"] += 1
            stats["total_removed_tail_occurrences"] += len(removed_tail_blocks)

        # Backward-compatible counters for older runs before the new hard constraint patch.
        legacy_filtered = _extract_nonempty_list_blocks(text, "filtered_special_tokens")
        if legacy_filtered:
            stats["legacy_logs_with_filtered_special_tokens"] += 1

        legacy_removed = _extract_nonempty_list_blocks(text, "removed_token_ids")
        if legacy_removed:
            stats["legacy_logs_with_removed_token_ids"] += 1

    return stats


def fmt_ratio(num: int, denom: int) -> str:
    if denom <= 0:
        return "nan"
    return f"{num / denom:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize artifact patterns for one consensus decoding experiment."
    )
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    json_paths = list(iter_result_jsons(exp_dir))
    log_paths = list(iter_verbose_logs(exp_dir))

    pred_stats = analyze_predictions(json_paths)
    log_stats = analyze_verbose_logs(log_paths)

    print(f"Experiment                       : {exp_dir}")
    print(f"Result json count                : {pred_stats['total_results']}")
    print(f"Verbose log count                : {log_stats['total_verbose_logs']}")
    print(
        "Predictions with ASCII          : "
        f"{pred_stats['prediction_has_ascii']} "
        f"({fmt_ratio(pred_stats['prediction_has_ascii'], pred_stats['total_results'])})"
    )
    print(
        "Predictions with replacement    : "
        f"{pred_stats['prediction_has_replacement_char']} "
        f"({fmt_ratio(pred_stats['prediction_has_replacement_char'], pred_stats['total_results'])})"
    )
    print(
        "Logs with filtered disallowed   : "
        f"{log_stats['logs_with_filtered_disallowed_tokens']} "
        f"({fmt_ratio(log_stats['logs_with_filtered_disallowed_tokens'], log_stats['total_verbose_logs'])})"
    )
    print(
        "Logs with removed disallowed    : "
        f"{log_stats['logs_with_removed_disallowed_tokens']} "
        f"({fmt_ratio(log_stats['logs_with_removed_disallowed_tokens'], log_stats['total_verbose_logs'])})"
    )
    print(
        "Logs with removed tail tokens   : "
        f"{log_stats['logs_with_removed_tail_tokens']} "
        f"({fmt_ratio(log_stats['logs_with_removed_tail_tokens'], log_stats['total_verbose_logs'])})"
    )
    print(
        "Filtered disallowed occurrences : "
        f"{log_stats['total_filtered_disallowed_occurrences']}"
    )
    print(
        "Removed disallowed occurrences  : "
        f"{log_stats['total_removed_disallowed_occurrences']}"
    )
    print(
        "Removed tail occurrences        : "
        f"{log_stats['total_removed_tail_occurrences']}"
    )
    if log_stats["legacy_logs_with_filtered_special_tokens"] or log_stats["legacy_logs_with_removed_token_ids"]:
        print(
            "Legacy logs detected            : "
            f"filtered_special={log_stats['legacy_logs_with_filtered_special_tokens']} "
            f"removed_token_ids={log_stats['legacy_logs_with_removed_token_ids']}"
        )


if __name__ == "__main__":
    main()
