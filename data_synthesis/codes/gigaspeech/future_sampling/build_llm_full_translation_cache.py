#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline full-translation cache with local vLLM batching.")
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument(
        "--model-path",
        default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
    )
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--id-column", default="id")
    p.add_argument("--src-column", default="src_text_full")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default="")
    return p.parse_args()


def setup_env() -> None:
    os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_list_column(raw: Any) -> List[str]:
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


def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text).strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    text = text.strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    text = text.strip('"').strip("'")
    return text


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return safe[:200] if safe else "unknown"


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f) - 1
    if total <= task_id:
        return 0
    return int(math.ceil((total - task_id) / num_tasks))


def iter_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> Iterator[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def get_one_row_by_id(input_tsv: str, utt_id: str, id_column: str = "id") -> Optional[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if str(row.get(id_column, "")).strip() == str(utt_id).strip():
                return row_idx, row
    return None


def build_prompt(full_source: str) -> str:
    return (
        "[TASK]\n"
        "Translate the [INPUT] text into Chinese.\n"
        "Output only the Chinese translation. No explanation.\n\n"
        f"[INPUT]\n{full_source}"
    )


def run_translation_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    messages = [[{"role": "user", "content": build_prompt(r["source_text"])}] for r in rows]
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    if len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(rows)} inputs.")

    results: List[Dict[str, Any]] = []
    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text if out.outputs else ""
        zh = normalize_zh(clean_llm_output(raw))
        results.append({
            "utt_id": row["utt_id"],
            "row_index": row["row_index"],
            "source_text": row["source_text"],
            "llm_full_translation": zh,
            "reference_chars": len(zh),
        })
    return results


def main() -> None:
    args = parse_args()
    setup_env()
    os.makedirs(args.output_root, exist_ok=True)

    out_jsonl = os.path.join(args.output_root, f"task_{args.task_id:03d}.jsonl")
    err_jsonl = os.path.join(args.output_root, f"task_{args.task_id:03d}.errors.jsonl")
    if args.overwrite:
        for p in [out_jsonl, err_jsonl]:
            if os.path.exists(p):
                os.remove(p)

    print(f"[vLLM] Loading model {args.model_path} (TP={args.tp}) ...")
    llm = LLM(
        model=args.model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    print("[vLLM] Model loaded.")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    if args.test_one:
        if args.utt_id:
            one = get_one_row_by_id(args.input_tsv, args.utt_id, args.id_column)
            row_iter: Any = [one] if one is not None else []
        else:
            row_iter = list(iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks))[:1]
        total = len(row_iter)
    else:
        row_iter = iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        total = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        if args.max_rows is not None:
            total = min(total, args.max_rows)

    print(
        f"[Task {args.task_id}] offline full-translation cache\n"
        f"  rows={total}, batch_size={args.batch_size}, max_tokens={args.max_tokens}\n"
        f"  model={args.model_path}\n"
        f"  out={out_jsonl}"
    )

    processed = written = failed = 0
    pbar = tqdm(total=total, desc=f"task_{args.task_id}")
    batch: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal written, failed
        if not batch:
            return
        try:
            results = run_translation_batch(llm, sampling_params, batch)
            with open(out_jsonl, "a", encoding="utf-8") as fo:
                for rec in results:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += len(results)
        except Exception as e:
            with open(err_jsonl, "a", encoding="utf-8") as fe:
                for item in batch:
                    fe.write(json.dumps({
                        "utt_id": item["utt_id"],
                        "row_index": item["row_index"],
                        "source_text": item["source_text"],
                        "error": str(e),
                    }, ensure_ascii=False) + "\n")
            failed += len(batch)
        batch.clear()

    for row_idx, row in row_iter:
        if args.max_rows is not None and processed >= args.max_rows:
            break
        processed += 1

        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx:09d}"
        sentences = parse_list_column(row.get(args.src_column))
        source_text = " ".join(s for s in sentences if str(s).strip()).strip()
        if not source_text:
            failed += 1
            with open(err_jsonl, "a", encoding="utf-8") as fe:
                fe.write(json.dumps({
                    "utt_id": utt_id,
                    "row_index": row_idx,
                    "error": f"Empty {args.src_column}",
                }, ensure_ascii=False) + "\n")
            pbar.update(1)
            continue

        batch.append({
            "utt_id": utt_id,
            "row_index": row_idx,
            "source_text": source_text,
        })
        if len(batch) >= args.batch_size:
            flush_batch()
        pbar.update(1)

    flush_batch()
    pbar.close()

    print(
        f"[Task {args.task_id}] Done. processed={processed} written={written} failed={failed}\n"
        f"  out={out_jsonl}\n"
        f"  err={err_jsonl}"
    )


if __name__ == "__main__":
    main()
