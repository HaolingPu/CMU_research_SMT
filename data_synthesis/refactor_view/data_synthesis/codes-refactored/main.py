#!/usr/bin/env python3
"""Gemini future-sampling simultaneous translation pipeline.

For each utterance in a GigaSpeech manifest TSV:
  1. Iterate over trajectory chunks (960ms windows).
  2. Sample N future English continuations from a base LM.
  3. Ask Gemini for the next safe Chinese delta.
  4. At the last chunk, force-complete the remaining translation.
  5. Save per-utterance JSON with source chunks, target deltas, and actions.

Usage:
    python main.py \\
      --input-tsv /path/to/manifest.tsv \\
      --output-root /path/to/output \\
      --base-model-path /path/to/Qwen3-4B-Base \\
      --thinking-model-name gemini-3.1-pro-preview \\
      --thinking-reasoning-effort low \\
      --max-rows 100 --overwrite
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from future_sampling import create_base_client, sample_futures
from gemini_translation import GeminiConfig, GeminiTranslator


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------

def parse_list_column(raw: Any) -> List[str]:
    """Parse a Python list literal stored in a TSV cell."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


def iter_rows(
    input_tsv: str,
    task_id: int = 0,
    num_tasks: int = 1,
    max_rows: Optional[int] = None,
) -> Iterator[Tuple[int, Dict[str, str]]]:
    """Yield (row_idx, row_dict) for rows assigned to this task shard."""
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row
            count += 1
            if max_rows is not None and count >= max_rows:
                break


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _normalize_zh(text: str) -> str:
    """Strip whitespace from Chinese text for comparison."""
    import unicodedata
    text = unicodedata.normalize("NFC", (text or "").strip())
    return re.sub(r"\s+", "", text)


def process_utterance(
    base_llm: Any,
    translator: GeminiTranslator,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    *,
    num_futures: int = 5,
    future_tokens: int = 10,
    sample_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run the thinking-policy pipeline for one utterance.

    Returns a dict with utt_id, source chunks, target deltas, actions, etc.
    """
    full_source = " ".join(sentences)
    n_chunks = len(trajectory)

    source_chunks: List[str] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    committed = ""
    accumulated_source = ""

    for chunk_idx, chunk in enumerate(trajectory):
        chunk_str = (chunk or "").strip()
        if chunk_str:
            accumulated_source = (accumulated_source + " " + chunk_str).strip()
        source_chunks.append(chunk_str)

        is_last = chunk_idx == n_chunks - 1
        print(f"\n--- [{utt_id}] chunk {chunk_idx}/{n_chunks-1} ---")
        print(f"  accumulated_source: {accumulated_source}")
        print(f"  committed: {committed!r}")

        # --- Last chunk: force-complete remaining translation ---
        if is_last:
            full_translation = translator.complete_translation(full_source, committed)
            committed_norm = _normalize_zh(committed)
            full_norm = _normalize_zh(full_translation)
            if len(full_norm) > len(committed_norm):
                remaining = full_norm[len(committed_norm):]
                target_deltas.append(remaining)
                actions.append("WRITE")
                print(f"  [LAST] full_translation: {full_translation!r}")
                print(f"  [LAST] remaining delta: {remaining!r}")
            else:
                target_deltas.append("")
                actions.append("READ")
                print(f"  [LAST] no remaining delta")
            continue

        # --- Future sampling ---
        futures = sample_futures(
            base_llm,
            accumulated_source,
            num_futures=num_futures,
            future_tokens=future_tokens,
            temperature=sample_temperature,
        )
        print(f"  futures ({len(futures)}):")
        for i, fut in enumerate(futures):
            print(f"    [{i}] {fut}")

        if len(futures) < 2:
            target_deltas.append("")
            actions.append("READ")
            print(f"  action: READ (too few futures)")
            continue

        # --- Gemini: get safe delta ---
        delta = translator.get_safe_delta(accumulated_source, futures, committed)
        print(f"  delta: {delta!r}")

        if delta:
            target_deltas.append(delta)
            actions.append("WRITE")
            committed = (committed or "") + delta
            print(f"  action: WRITE -> committed: {committed!r}")
        else:
            target_deltas.append("")
            actions.append("READ")
            print(f"  action: READ")

    return {
        "utt_id": utt_id,
        "original_text": full_source,
        "source_future_sampling": source_chunks,
        "target_future_sampling": target_deltas,
        "actions": actions,
        "system_output_text": "".join(d for d in target_deltas if d),
        "config": {
            "num_futures": num_futures,
            "future_tokens": future_tokens,
            "thinking_model": translator._config.model,
            "reasoning_effort": translator._config.reasoning_effort,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini future-sampling simultaneous translation pipeline."
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-url", default="http://localhost:8000/v1",
                   help="Base URL of the vLLM server (OpenAI-compatible API).")
    p.add_argument("--base-model-name", default="Qwen3-4B-Base",
                   help="Model name as registered on the vLLM server.")
    p.add_argument("--thinking-model-name", default="gemini-3-flash-preview")
    p.add_argument(
        "--thinking-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="low",
    )
    p.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY",
                   help="Name of the environment variable holding the Gemini API key.")
    p.add_argument("--gemini-timeout", type=float, default=600.0)

    p.add_argument("--num-futures", type=int, default=5)
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-column", default="id")

    return p.parse_args()


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return (safe or "unknown")[:200]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Gemini API key ---
    api_key = os.environ.get(args.gemini_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"ERROR: env var {args.gemini_api_key_env} is not set. "
            "Export your Gemini API key before running."
        )

    os.makedirs(args.output_root, exist_ok=True)

    # --- Connect to vLLM server ---
    print(f"[Base] Connecting to vLLM server at {args.base_model_url} (model={args.base_model_name}) ...")
    base_llm = create_base_client(args.base_model_url, args.base_model_name)
    print("[Base] Client ready.")

    config = GeminiConfig(
        api_key=api_key,
        model=args.thinking_model_name,
        reasoning_effort=args.thinking_reasoning_effort,
        timeout=args.gemini_timeout,
    )
    translator = GeminiTranslator(config)
    print(f"[Gemini] Using model={config.model} reasoning_effort={config.reasoning_effort}")

    # --- Collect rows ---
    row_list = list(iter_rows(
        args.input_tsv, args.task_id, args.num_tasks, args.max_rows
    ))
    if not row_list:
        print("No rows to process.")
        return

    print(f"[Task {args.task_id}] Processing {len(row_list)} rows")

    written = 0
    skipped = 0
    failed = 0

    for row_idx, row in tqdm(row_list, desc=f"task_{args.task_id}"):
        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx}"
        out_path = os.path.join(args.output_root, f"{_sanitize_filename(utt_id)}.json")

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        sentences = parse_list_column(row.get("src_text_full"))
        trajectory = parse_list_column(row.get("src_trajectory"))
        if not sentences or not trajectory:
            json.dump(
                {"utt_id": utt_id, "error": "empty src_text_full or src_trajectory"},
                open(out_path, "w", encoding="utf-8"),
                ensure_ascii=False,
            )
            failed += 1
            continue

        try:
            result = process_utterance(
                base_llm,
                translator,
                utt_id,
                sentences,
                trajectory,
                num_futures=args.num_futures,
                future_tokens=args.future_tokens,
                sample_temperature=args.sample_temperature,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": str(e)}, f, ensure_ascii=False)
            failed += 1

    print(
        f"[Task {args.task_id}] Done. written={written}, skipped={skipped}, "
        f"failed={failed} -> {args.output_root}"
    )


if __name__ == "__main__":
    main()
