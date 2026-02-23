#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import os
import re
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Salami LLM output for GigaSpeech TSV.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-root", required=True)

    parser.add_argument("--model-path", default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)

    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--id-column", default="id")
    parser.add_argument("--src-text-full-column", default="src_text_full")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ============================================================
# Env
# ============================================================
def setup_env():
    os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Prompt (Salami)
# ============================================================
SYSTEM_PROMPT = (
    "You will be provided with a sentence in English, and your task is to interpret it into Chinese.\n"
    "Always answer in the following JSON format:{'segmented_pairs':List[Tuple[English, Language]], 'output':Language}"
)

USER_PROMPT = (
    "Instructions: 'Salami technique' in simultaneous interpretation refers to a technique where the interpreter breaks down the source\n"
    "language input into smaller, manageable segments that each contain enough information to be accurately interpreted.\n"
    "1. Break down the following sentence into smaller segments for easier simultaneous interpretation.\n"
    "2. Translate each segment into Chinese.\n"
    "3. Connect the translated segments.\n"
    "----------Inputs: {text}----------\n"
    "\n"
    "Example Text\n"
    "Almost every way we make electricity today\n"
    "except for the emerging renewables and nuclear puts out CO2.\n"
    "\n"
    "Output\n"
    "Language = Chinese\n"
    "{'segmented_pairs':[\n"
    "[\"Almost every way\",\"几乎每一种方式\"],\n"
    "[\"we make electricity today\",\"我们今天发电的方式\"],\n"
    "[\"except for the emerging renewables and nuclear\",\"除了新兴的可再生能源和核能\"],\n"
    "[\"puts out CO2\",\"会排放二氧化碳\"]],\n"
    "'output':\"几乎每一种我们今天发电的方式，除了新兴的可再生能源和核能，都会排放二氧化碳。\"}\n"
)


def build_user_prompt(text: str) -> str:
    return USER_PROMPT.replace("{text}", text)


# ============================================================
# TSV helpers
# ============================================================
def parse_src_text_full(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    raw = str(raw_value).strip()
    if raw == "":
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = raw

    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    s = str(parsed).strip()
    return [s] if s else []


def iter_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> Iterator[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total_rows = sum(1 for _ in f) - 1
    if total_rows <= task_id:
        return 0
    return int(math.ceil((total_rows - task_id) / num_tasks))


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:200] if safe else "unknown_id"


# ============================================================
# JSON schema
# ============================================================
json_schema_salami = {
    "type": "object",
    "properties": {
        "segmented_pairs": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "output": {"type": "string"},
    },
    "required": ["segmented_pairs", "output"],
}


def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(s[l:r + 1])
        raise


def run_llm_batch_raw(llm, sampling_params, texts):
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(t)},
        ]
        for t in texts
    ]
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    if len(outputs) != len(texts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(texts)} inputs.")
    return outputs


def flush_global_batch(llm, sampling_params, batch_tasks, pending_utts):
    texts = [t["text"] for t in batch_tasks]
    try:
        outputs = run_llm_batch_raw(llm, sampling_params, texts)
    except Exception as batch_err:
        for task in batch_tasks:
            st = pending_utts[task["utt_id"]]
            try:
                out = run_llm_batch_raw(llm, sampling_params, [task["text"]])[0]
                st["results"][task["sentence_index"]] = safe_json_loads(out.outputs[0].text.strip())
            except Exception as single_err:
                st["errors"].append(
                    {
                        "sentence_index": task["sentence_index"],
                        "input": task["text"],
                        "error": f"batch_error={batch_err}; single_error={single_err}",
                    }
                )
            finally:
                st["done_sentences"] += 1
        return

    for task, out in zip(batch_tasks, outputs):
        st = pending_utts[task["utt_id"]]
        try:
            st["results"][task["sentence_index"]] = safe_json_loads(out.outputs[0].text.strip())
        except Exception as parse_err:
            st["errors"].append(
                {
                    "sentence_index": task["sentence_index"],
                    "input": task["text"],
                    "error": f"batch_parse_error={parse_err}",
                }
            )
        finally:
            st["done_sentences"] += 1


def finalize_utt_state(state: Dict[str, Any]) -> Dict[str, Any]:
    utt_id = state["utt_id"]
    sentences = state["sentences"]

    if state["errors"] or len(state["results"]) != len(sentences):
        return {
            "utt_id": utt_id,
            "input": " ".join(sentences).strip(),
            "error": "Some sentences failed during LLM inference.",
            "errors": state["errors"],
        }

    # 关键：输出结构和你旧版一致（只保留这4个字段）
    all_pairs: List[List[str]] = []
    outputs_text: List[str] = []
    for i in range(len(sentences)):
        salami = state["results"][i]
        pairs = salami.get("segmented_pairs", [])
        out_text = salami.get("output", "")
        if isinstance(pairs, list):
            all_pairs.extend(pairs)
        outputs_text.append(str(out_text).strip())

    return {
        "segmented_pairs": all_pairs,
        "output": "".join(outputs_text).strip(),
        "input": " ".join(sentences).strip(),
        "utt_id": utt_id,
    }


def main():
    args = parse_args()
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise ValueError("Invalid task-id/num-tasks.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    os.makedirs(args.output_root, exist_ok=True)
    setup_env()

    print(f"[Task {args.task_id}] Loading model TP={args.tp} ...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        repetition_penalty=1.1,
        guided_decoding=GuidedDecodingParams(json=json_schema_salami),
    )

    total_assigned = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
    if args.max_rows is not None:
        total_assigned = min(total_assigned, args.max_rows)

    pending_utts: Dict[str, Dict[str, Any]] = {}
    sentence_queue: Deque[Dict[str, Any]] = deque()

    written = 0
    skipped_existing = 0
    failed = 0
    processed_rows = 0

    pbar = tqdm(total=total_assigned, desc=f"task_{args.task_id}")

    for row_idx, row in iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks):
        if args.max_rows is not None and processed_rows >= args.max_rows:
            break
        processed_rows += 1

        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx:09d}"
        out_path = os.path.join(args.output_root, f"{sanitize_filename(utt_id)}.json")

        if os.path.exists(out_path) and not args.overwrite:
            skipped_existing += 1
            pbar.update(1)
            continue

        sentences = parse_src_text_full(row.get(args.src_text_full_column))
        if len(sentences) == 0:
            failed += 1
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utt_id": utt_id,
                        "input": "",
                        "error": f"Empty/invalid {args.src_text_full_column}",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            pbar.update(1)
            continue

        pending_utts[utt_id] = {
            "utt_id": utt_id,
            "sentences": sentences,
            "out_path": out_path,
            "results": {},
            "errors": [],
            "done_sentences": 0,
        }

        for sent_idx, text in enumerate(sentences):
            sentence_queue.append(
                {
                    "utt_id": utt_id,
                    "sentence_index": sent_idx,
                    "text": text,
                }
            )

        while len(sentence_queue) >= args.batch_size:
            batch_tasks = [sentence_queue.popleft() for _ in range(args.batch_size)]
            flush_global_batch(llm, sampling_params, batch_tasks, pending_utts)

            done_ids = [u for u, st in pending_utts.items() if st["done_sentences"] == len(st["sentences"])]
            for u in done_ids:
                st = pending_utts[u]
                out_obj = finalize_utt_state(st)
                with open(st["out_path"], "w", encoding="utf-8") as f:
                    json.dump(out_obj, f, ensure_ascii=False, indent=2)
                if "error" in out_obj:
                    failed += 1
                else:
                    written += 1
                del pending_utts[u]

        pbar.update(1)

    if sentence_queue:
        batch_tasks = list(sentence_queue)
        sentence_queue.clear()
        flush_global_batch(llm, sampling_params, batch_tasks, pending_utts)

    for u in list(pending_utts.keys()):
        st = pending_utts[u]
        out_obj = finalize_utt_state(st)
        with open(st["out_path"], "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        if "error" in out_obj:
            failed += 1
        else:
            written += 1
        del pending_utts[u]

    pbar.close()
    print(f"[Task {args.task_id}] Done. written={written}, skipped_existing={skipped_existing}, failed={failed}")


if __name__ == "__main__":
    main()
