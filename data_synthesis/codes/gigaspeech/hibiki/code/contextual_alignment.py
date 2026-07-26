#!/usr/bin/env python3
"""
Standalone greedy argmax contextual alignment checker.

Goals:
- no jieba
- no sibling-file imports
- target units are tokenizer tokens
- focus on whether greedy argmax alignment produces out-of-order jumps
- write outputs under output/argmax/

Core dependencies:
- transformers
- vllm
- numpy
- pandas
- tqdm
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


CODE_DIR = Path(__file__).resolve().parent
HIBIKI_DIR = CODE_DIR.parent
OUTPUT_DIR = HIBIKI_DIR / "output"
ARGMAX_DIR = OUTPUT_DIR / "argmax"
DEFAULT_INPUT_TSV = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
)
DEFAULT_TOKENIZER_PATH = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_OUTPUT_JSONL = ARGMAX_DIR / "contextual_alignment_argmax.jsonl"
DEFAULT_OUTPUT_TXT = ARGMAX_DIR / "contextual_alignment_argmax.txt"
DEFAULT_OUTPUT_PRETTY_JSON = ARGMAX_DIR / "contextual_alignment_argmax.pretty.json"
_TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the English source into "
    "Chinese. Output only the Chinese translation."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run standalone greedy argmax contextual alignment and check for out-of-order jumps."
    )
    p.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--base-model-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--max-rows", type=int, default=5)
    p.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    p.add_argument("--output-txt", default=str(DEFAULT_OUTPUT_TXT))
    p.add_argument("--output-pretty-json", default=str(DEFAULT_OUTPUT_PRETTY_JSON))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return p.parse_args()


def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def maybe_add_space(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if left[-1].isspace() or right[0].isspace():
        return ""
    if left[-1].isalnum() and right[0].isalnum():
        return " "
    return ""


def build_cumulative_prefixes(chunks: List[str]) -> List[str]:
    prefixes: List[str] = []
    running = ""
    for chunk in chunks:
        chunk = str(chunk)
        running = running + maybe_add_space(running, chunk) + chunk
        prefixes.append(running)
    return prefixes


def iter_rows(tsv_path: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    df = pd.read_csv(tsv_path, sep="\t")
    if max_rows > 0:
        df = df.head(max_rows)
    return df.to_dict("records")


def tokenize_target(
    tokenizer: Any,
    text: str,
) -> Tuple[List[int], List[str], List[str]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return [int(x) for x in token_ids], list(raw_tokens), decoded_tokens


def build_translation_prompt_prefix_ids(
    tokenizer: Any,
    source_prefix: str,
) -> List[int]:
    messages = [
        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "English source:\n"
            f"{source_prefix}\n\nChinese translation:",
        },
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return [int(x) for x in prompt_ids]


def compute_next_token_logprob_batched(
    base_llm: Any,
    prompt_token_id_prefixes: Sequence[Sequence[int]],
    target_token_ids: List[int],
) -> List[float]:
    from vllm import SamplingParams, TokensPrompt

    all_full_ids: List[List[int]] = []
    for prompt_ids, target_token_id in zip(prompt_token_id_prefixes, target_token_ids):
        all_full_ids.append(list(prompt_ids) + [int(target_token_id)])

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )

    outputs = base_llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in all_full_ids],
        params,
    )

    results: List[float] = []
    for i, output in enumerate(outputs):
        target_token_id = int(target_token_ids[i])
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or prompt_logprobs[-1] is None:
            results.append(float("-inf"))
            continue
        last_entry = prompt_logprobs[-1]
        if target_token_id not in last_entry:
            results.append(float("-inf"))
            continue
        logprob_obj = last_entry[target_token_id]
        if hasattr(logprob_obj, "logprob"):
            results.append(float(logprob_obj.logprob))
        else:
            results.append(float(logprob_obj))
    return results


def compute_token_scores_over_prefixes(
    base_llm: Any,
    tokenizer: Any,
    source_prefixes: List[str],
    target_token_ids: List[int],
) -> np.ndarray:
    num_target = len(target_token_ids)
    num_source = len(source_prefixes)
    scores = np.full((num_target, num_source), float("-inf"), dtype=np.float64)

    translation_prompt_prefixes = [
        build_translation_prompt_prefix_ids(tokenizer, source_prefix)
        for source_prefix in source_prefixes
    ]

    for j in tqdm(range(num_target), desc="target_tokens", leave=False):
        target_prefix_ids = target_token_ids[:j]
        current_token_id = target_token_ids[j]
        prompt_token_id_prefixes = [
            prefix_ids + target_prefix_ids
            for prefix_ids in translation_prompt_prefixes
        ]
        token_ids_for_batch = [current_token_id] * num_source
        logprobs = compute_next_token_logprob_batched(
            base_llm, prompt_token_id_prefixes, token_ids_for_batch
        )
        for k in range(num_source):
            scores[j, k] = logprobs[k]

    return scores


def assign_chunks_from_deltas_greedy(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_target, num_source = scores.shape
    deltas = np.full_like(scores, float("-inf"))

    for j in range(num_target):
        deltas[j, 0] = scores[j, 0]
        for k in range(1, num_source):
            deltas[j, k] = scores[j, k] - scores[j, k - 1]

    if num_target == 0 or num_source == 0:
        return deltas, np.zeros((num_target,), dtype=np.int64)

    assigned_chunks = np.argmax(deltas, axis=1).astype(np.int64)
    return deltas, assigned_chunks


def build_step_aligned_target_trajectory(
    tokenizer: Any,
    target_token_ids: List[int],
    assigned_chunks: Sequence[int],
    num_source_steps: int,
) -> List[str]:
    trajectory: List[str] = [""] * num_source_steps
    if len(target_token_ids) == 0:
        return trajectory

    groups: List[Tuple[int, List[int]]] = []
    current_chunk = int(assigned_chunks[0])
    current_ids = [target_token_ids[0]]
    for j in range(1, len(target_token_ids)):
        chunk = int(assigned_chunks[j])
        if chunk == current_chunk:
            current_ids.append(target_token_ids[j])
        else:
            groups.append((current_chunk, current_ids))
            current_chunk = chunk
            current_ids = [target_token_ids[j]]
    groups.append((current_chunk, current_ids))

    for chunk_idx, group_ids in groups:
        decoded = tokenizer.decode(group_ids)
        safe_idx = max(0, min(int(chunk_idx), num_source_steps - 1))
        if trajectory[safe_idx]:
            trajectory[safe_idx] += decoded
        else:
            trajectory[safe_idx] = decoded
    return trajectory


def find_non_monotonic_jumps(
    tokenizer: Any,
    target_token_ids: List[int],
    assigned_chunks: Sequence[int],
) -> List[Dict[str, Any]]:
    jumps: List[Dict[str, Any]] = []
    for j in range(1, len(assigned_chunks)):
        prev_chunk = int(assigned_chunks[j - 1])
        curr_chunk = int(assigned_chunks[j])
        if curr_chunk < prev_chunk:
            jumps.append(
                {
                    "from_token_idx": j - 1,
                    "to_token_idx": j,
                    "from_chunk": prev_chunk,
                    "to_chunk": curr_chunk,
                    "from_token_id": int(target_token_ids[j - 1]),
                    "to_token_id": int(target_token_ids[j]),
                    "from_token_decoded": tokenizer.decode([target_token_ids[j - 1]]),
                    "to_token_decoded": tokenizer.decode([target_token_ids[j]]),
                    "decoded_prefix_through_to_token": tokenizer.decode(target_token_ids[: j + 1]),
                }
            )
    return jumps


def repair_chunks_monotonic(assigned_chunks: Sequence[int]) -> np.ndarray:
    if len(assigned_chunks) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.maximum.accumulate(np.asarray(assigned_chunks, dtype=np.int64))


def find_repair_events(
    tokenizer: Any,
    target_token_ids: List[int],
    greedy_chunks: Sequence[int],
    monotonic_chunks: Sequence[int],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for j, (greedy_chunk, monotonic_chunk) in enumerate(zip(greedy_chunks, monotonic_chunks)):
        greedy_chunk = int(greedy_chunk)
        monotonic_chunk = int(monotonic_chunk)
        if monotonic_chunk != greedy_chunk:
            events.append(
                {
                    "token_idx": j,
                    "token_id": int(target_token_ids[j]),
                    "token_decoded": tokenizer.decode([target_token_ids[j]]),
                    "greedy_chunk": greedy_chunk,
                    "monotonic_chunk": monotonic_chunk,
                    "decoded_prefix_through_token": tokenizer.decode(target_token_ids[: j + 1]),
                }
            )
    return events


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "source_text_full": record["source_text_full"],
        "target_full_zh": record["target_full_zh"],
        "src_trajectory": record["src_trajectory"],
        "target_trajectory": record["target_trajectory_from_monotonic_alignment"],
    }


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(summarize_record(record), ensure_ascii=False) + "\n")


def write_pretty_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    payload = [summarize_record(record) for record in records]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_demo_text(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec_idx, record in enumerate(records):
            summary = summarize_record(record)
            f.write(f"=== CASE {rec_idx + 1}: {summary['id']} ===\n")
            f.write(f"id: {summary['id']}\n")
            f.write(f"source_text_full: {summary['source_text_full']}\n")
            f.write(f"target_full_zh: {summary['target_full_zh']}\n")
            f.write(f"src_trajectory: {json.dumps(summary['src_trajectory'], ensure_ascii=False)}\n")
            f.write(f"target_trajectory: {json.dumps(summary['target_trajectory'], ensure_ascii=False)}\n\n")


def build_record_contextual_debug(
    row: Dict[str, Any],
    tokenizer: Any,
    base_llm: Any,
) -> Dict[str, Any]:
    utt_id = str(row.get("id", "")).strip()
    source_chunks = parse_list_column(row.get("src_trajectory"))
    source_prefixes = build_cumulative_prefixes(source_chunks)
    target_text = str(row.get("llm_reference_text", "")).strip()
    source_text_full = str(row.get("src_text", "")).strip()

    token_ids, raw_tokens, decoded_tokens = tokenize_target(tokenizer, target_text)

    if len(token_ids) == 0 or len(source_prefixes) == 0:
        return {
            "id": utt_id,
            "source_text_full": source_text_full,
            "target_full_zh": target_text,
            "src_trajectory": source_chunks,
            "source_prefixes": source_prefixes,
            "target_token_ids": token_ids,
            "target_token_raw": raw_tokens,
            "target_token_decoded": decoded_tokens,
            "per_token_scores": [],
            "per_token_deltas": [],
            "assigned_chunk_per_token_greedy": [],
            "assigned_chunk_per_token_monotonic": [],
            "non_monotonic_jumps": [],
            "repair_events": [],
            "target_trajectory_from_greedy_token_alignment": [""] * len(source_chunks),
            "target_trajectory_from_monotonic_alignment": [""] * len(source_chunks),
        }

    scores = compute_token_scores_over_prefixes(base_llm, tokenizer, source_prefixes, token_ids)
    deltas, assigned_chunks = assign_chunks_from_deltas_greedy(scores)
    monotonic_chunks = repair_chunks_monotonic(assigned_chunks)
    non_monotonic_jumps = find_non_monotonic_jumps(tokenizer, token_ids, assigned_chunks)
    repair_events = find_repair_events(tokenizer, token_ids, assigned_chunks, monotonic_chunks)
    greedy_trajectory = build_step_aligned_target_trajectory(
        tokenizer, token_ids, assigned_chunks, len(source_chunks)
    )
    monotonic_trajectory = build_step_aligned_target_trajectory(
        tokenizer, token_ids, monotonic_chunks, len(source_chunks)
    )

    return {
        "id": utt_id,
        "source_text_full": source_text_full,
        "target_full_zh": target_text,
        "src_trajectory": source_chunks,
        "source_prefixes": source_prefixes,
        "target_token_ids": token_ids,
        "target_token_raw": raw_tokens,
        "target_token_decoded": decoded_tokens,
        "per_token_scores": scores.tolist(),
        "per_token_deltas": deltas.tolist(),
        "assigned_chunk_per_token_greedy": [int(x) for x in assigned_chunks],
        "assigned_chunk_per_token_monotonic": [int(x) for x in monotonic_chunks],
        "non_monotonic_jumps": non_monotonic_jumps,
        "repair_events": repair_events,
        "target_trajectory_from_greedy_token_alignment": greedy_trajectory,
        "target_trajectory_from_monotonic_alignment": monotonic_trajectory,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("[Mode] Standalone greedy argmax contextual alignment", flush=True)
    print(f"[Tokenizer] {args.tokenizer_path}", flush=True)
    print(f"[BaseModel] {args.base_model_path}", flush=True)
    if args.tokenizer_path != args.base_model_path:
        print("[Warning] tokenizer-path and base-model-path differ.", flush=True)

    output_jsonl = Path(args.output_jsonl)
    output_txt = Path(args.output_txt)
    output_pretty_json = Path(args.output_pretty_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_pretty_json.parent.mkdir(parents=True, exist_ok=True)

    from vllm import LLM

    base_llm = LLM(
        model=args.base_model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    rows = list(iter_rows(args.input_tsv, args.max_rows))
    records = []
    for row in tqdm(rows, desc="cases"):
        records.append(build_record_contextual_debug(row, tokenizer, base_llm))

    write_jsonl(output_jsonl, records)
    write_pretty_json(output_pretty_json, records)
    write_demo_text(output_txt, records)

    print(f"[Done] Wrote {len(records)} records to: {output_jsonl}", flush=True)
    print(f"[Done] Wrote pretty json to: {output_pretty_json}", flush=True)
    print(f"[Done] Wrote readable demo to: {output_txt}", flush=True)


if __name__ == "__main__":
    main()



# 方案 A：先 token-level argmax，再 cumulative max

# 最简单，能跑通，适合作 baseline。

# 方案 B：先把 token 合成 word / span，再对 span 做单调化

# 更自然，但实现复杂些。

# 方案 C：直接做带单调约束的全局优化
