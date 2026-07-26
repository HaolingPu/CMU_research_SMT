#!/usr/bin/env python3
"""
Standalone contextual alignment with sentence-local token alignment.

整体 Pipeline 概览:
===================
目标: 给定英文 source 的分步 trajectory（若干 chunk），为中文 target 也生成
      对应的分步 trajectory，使得 target 的每一步翻译内容和 source 的每一步
      对齐（contextual alignment）。

Step 1 - 构建 source prefix 序列:
    把 source 的 trajectory chunk 逐步拼接，得到累积前缀列表。
    例如 ["Hello", "world"] → ["Hello", "Hello world"]

Step 2 - 计算 token 级别的 logprob 矩阵 (score matrix):
    对每个 target token j 和每个 source prefix k，用 LLM 做 teacher-forcing:
    把 (system prompt + source_prefix_k + target_token_0..j-1) 拼成 prompt，
    然后看模型在下一个位置给出 target_token_j 的 logprob。
    得到 scores[j, k] = logP(target_token_j | source_prefix_k, target_0..j-1)

Step 3 - Argmax 求每个 target token 应该对齐到哪个 source chunk:
    计算 delta 矩阵: deltas[j, k] = scores[j, k] - scores[j, k-1]
    含义: 当 source 从 prefix_{k-1} 扩展到 prefix_k 时，target_token_j
          的 logprob 增益最大的那个 k，就是该 token 最依赖的 source chunk。
    assigned_chunk[j] = argmax_k deltas[j, k]
    即: 每个 target token 被分配到让它 logprob 提升最多的那个 source chunk。

Step 4 - 单调性修复 (monotonic repair):
    翻译是从左到右的，所以 chunk 分配必须单调递增。
    用 cumulative max 修复: chunk[j] = max(chunk[0], ..., chunk[j])

Step 5 - 直接按对齐后的 token 构建 target trajectory:
    不再做任何词级聚合。
    直接把单调修复后的 token chunk 分配结果按 chunk 分组，
    再按原始 token 顺序精确 decode，把每组 token 还原成文本片段。

Step 6 - 构建 target trajectory:
    把每个 token group 放到对应的 source step 位置，
    生成和 source trajectory 等长的 target trajectory 列表。
"""

import argparse
import ast
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


CODE_DIR = Path(__file__).resolve().parent
HIBIKI_DIR = CODE_DIR.parent
DEFAULT_INPUT_TSV = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
)
DEFAULT_TOKENIZER_PATH = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
_TARGET_LANG_SPECS = {
    "zh": {
        "language_name": "Chinese",
        "user_label": "Chinese translation",
        "output_dir_name": "hibiki-100",
        "sentence_end_chars": "。！？.!?",
    },
    "de": {
        "language_name": "German",
        "user_label": "German translation",
        "output_dir_name": "hibiki-de-100",
        "sentence_end_chars": ".!?",
    },
    "ja": {
        "language_name": "Japanese",
        "user_label": "Japanese translation",
        "output_dir_name": "hibiki-ja-100",
        "sentence_end_chars": "。！？.!?",
    },
}


def get_lang_spec(target_lang: str) -> Dict[str, str]:
    if target_lang not in _TARGET_LANG_SPECS:
        raise ValueError(f"Unsupported target_lang={target_lang!r}")
    return _TARGET_LANG_SPECS[target_lang]


def build_translation_system_prompt(target_lang: str) -> str:
    spec = get_lang_spec(target_lang)
    return (
        "You are a professional translator. Translate the English source into "
        f"{spec['language_name']}. Output only the {spec['language_name']} translation."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run contextual alignment and build target trajectories directly from aligned tokens."
    )
    p.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--base-model-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--max-rows", type=int, default=5)
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-lang", choices=["zh", "de", "ja"], default="zh")
    p.add_argument("--target-list-column", default="")
    p.add_argument("--target-full-column", default="")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--debug-id", default="")
    p.add_argument("--debug-log", default="")
    return p.parse_args()


def get_target_list_column(args: argparse.Namespace) -> str:
    if args.target_list_column:
        return args.target_list_column
    return "llm_reference_text_list" if args.target_lang == "zh" else f"llm_reference_text_list_{args.target_lang}"


def get_target_full_column(args: argparse.Namespace) -> str:
    if args.target_full_column:
        return args.target_full_column
    if args.target_lang == "zh":
        return "target_full_zh"
    return f"target_full_{args.target_lang}"


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


def maybe_add_space_target(left: str, right: str, target_lang: str) -> str:
    if not left or not right:
        return ""
    if target_lang in {"zh", "ja"}:
        return ""
    if left[-1].isspace() or right[0].isspace():
        return ""
    no_space_before = set(",.!?:;%)]}\"'»”’")
    no_space_after = set("([{\"'«“‘")
    if right[0] in no_space_before:
        return ""
    if left[-1] in no_space_after:
        return ""
    return " "


def join_target_pieces(pieces: Sequence[str], target_lang: str) -> str:
    if target_lang in {"zh", "ja"}:
        return "".join(str(x) for x in pieces)
    running = ""
    for piece in pieces:
        piece = str(piece)
        running = running + maybe_add_space_target(running, piece, target_lang) + piece
    return running


def build_cumulative_prefixes(chunks: List[str]) -> List[str]:
    prefixes: List[str] = []
    running = ""
    for chunk in chunks:
        chunk = str(chunk)
        running = running + maybe_add_space(running, chunk) + chunk
        prefixes.append(running)
    return prefixes


def iter_rows(tsv_path: str, max_rows: int, task_id: int = 0, num_tasks: int = 1) -> Iterable[Dict[str, Any]]:
    df = pd.read_csv(tsv_path, sep="\t")
    if max_rows > 0:
        df = df.head(max_rows)
    if num_tasks > 1:
        df = df.iloc[list(range(task_id, len(df), num_tasks))]
    return df.to_dict("records")


def tokenize_target_with_offsets(
    tokenizer: Any,
    text: str,
) -> Tuple[List[int], List[str], List[str], List[Tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_tokens = [decode_token_ids_exact(tokenizer, [tid]) for tid in token_ids]
    return token_ids, list(raw_tokens), decoded_tokens, offsets


def decode_token_ids_exact(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        [int(tid) for tid in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_translation_prompt_prefix_ids(
    tokenizer: Any,
    source_prefix: str,
    target_lang: str,
) -> List[int]:
    spec = get_lang_spec(target_lang)
    messages = [
        {"role": "system", "content": build_translation_system_prompt(target_lang)},
        {
            "role": "user",
            "content": "English source:\n"
            f"{source_prefix}\n\n{spec['user_label']}:",
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
        last_entry = prompt_logprobs[-1] # get the last token's logprob
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
    target_lang: str,
) -> np.ndarray:
    """
    [Step 2] 构建 logprob score 矩阵。

    返回 scores[j, k]:
      对于第 j 个 target token, 在给定 source_prefix_k 的条件下,
      模型 teacher-forced 输出该 token 的 log 概率。

    具体做法: 对每个 target token j, 构造 batch:
      prompt = chat_template(source_prefix_k) + target_token_0..j-1
      然后用 vLLM 的 prompt_logprobs 拿到 target_token_j 的 logprob。
      一次 batch 覆盖所有 source prefix, 所以填满 scores[j, :] 一整行。
    """
    num_target = len(target_token_ids)
    num_source = len(source_prefixes)
    scores = np.full((num_target, num_source), float("-inf"), dtype=np.float64)

    # 为每个 source prefix 预先构建 chat template prompt 的 token ids
    translation_prompt_prefixes = [
        build_translation_prompt_prefix_ids(tokenizer, source_prefix, target_lang)
        for source_prefix in source_prefixes
    ]

    for j in tqdm(range(num_target), desc="target_tokens", leave=False):
        # target_prefix_ids = 已经生成的 target tokens (0..j-1), 作为 teacher-forcing 的上文
        target_prefix_ids = target_token_ids[:j]
        current_token_id = target_token_ids[j]
        # 拼接: [chat_template(src_prefix_k)] + [target_0..j-1], 对每个 k
        prompt_token_id_prefixes = [
            prefix_ids + target_prefix_ids
            for prefix_ids in translation_prompt_prefixes
        ]
        token_ids_for_batch = [current_token_id] * num_source
        # 一次 batch 调用, 拿到 scores[j, 0..num_source-1]
        logprobs = compute_next_token_logprob_batched(
            base_llm, prompt_token_id_prefixes, token_ids_for_batch
        )
        for k in range(num_source):
            scores[j, k] = logprobs[k]

    return scores


def assign_chunks_from_deltas_greedy(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    [Step 3] 通过 delta (logprob 增益) 的 argmax 来分配 chunk。

    deltas[j, k] = scores[j, k] - scores[j, k-1]
    含义: source 从 prefix_{k-1} 扩展到 prefix_k 时, target token j 的 logprob 变化量。
    argmax_k deltas[j, k] 就是对 target token j 影响最大的那个 source chunk。

    直觉: 如果某个 target token 在看到 source chunk k 之后 logprob 突然变高,
          说明这个 token 的翻译最依赖于 chunk k 带来的信息。
    """
    num_target, num_source = scores.shape
    deltas = np.full_like(scores, float("-inf"))

    for j in range(num_target):
        # k=0 时没有前一个 prefix, 直接用 scores[j, 0]
        deltas[j, 0] = scores[j, 0]
        for k in range(1, num_source):
            deltas[j, k] = scores[j, k] - scores[j, k - 1]

    if num_target == 0 or num_source == 0:
        return deltas, np.zeros((num_target,), dtype=np.int64)

    # 每个 target token 取 delta 最大的 source chunk index
    assigned_chunks = np.argmax(deltas, axis=1).astype(np.int64)
    return deltas, assigned_chunks


def repair_chunks_monotonic(assigned_chunks: Sequence[int]) -> np.ndarray:
    """
    [Step 4] 单调性修复: 用 cumulative max 保证 chunk 分配递增。
    例如 [0, 2, 1, 3] → [0, 2, 2, 3], 防止翻译"倒回去"。
    """
    if len(assigned_chunks) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.maximum.accumulate(np.asarray(assigned_chunks, dtype=np.int64))


def split_text_into_sentences(text: str, sentence_end_chars: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in sentence_end_chars:
            end = i + 1
            while end < len(text) and text[end] in "\"'”’)]} ":
                end += 1
            sent = text[start:end].strip()
            if sent:
                real_start = text.find(sent, start, end)
                spans.append(
                    {"text": sent, "char_span": [int(real_start), int(real_start + len(sent))]}
                )
            start = end
        i += 1
    tail = text[start:].strip()
    if tail:
        real_start = text.find(tail, start)
        spans.append({"text": tail, "char_span": [int(real_start), int(real_start + len(tail))]})
    return spans


def sentence_text_list(text: str, sentence_end_chars: str) -> List[str]:
    return [str(x["text"]) for x in split_text_into_sentences(text, sentence_end_chars)]


def build_sentence_spans_from_list(
    source_text_full: str,
    source_sentences: Sequence[str],
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for sent in source_sentences:
        sent = str(sent).strip()
        if not sent:
            continue
        start = source_text_full.find(sent, cursor)
        if start < 0:
            start = source_text_full.find(sent)
        if start < 0:
            raise ValueError(f"Cannot locate source sentence in src_text: {sent!r}")
        end = start + len(sent)
        spans.append({"text": sent, "char_span": [int(start), int(end)]})
        cursor = end
    return spans


def assign_chunks_to_source_sentences(
    source_text_full: str,
    source_sentences: Sequence[str],
    source_chunks: Sequence[str],
) -> List[int]:
    sentence_spans = build_sentence_spans_from_list(source_text_full, source_sentences)
    if not sentence_spans:
        raise ValueError("Empty source sentence list: src_text_full")

    chunk_sentence_ids: List[int] = []
    cursor = 0
    sent_idx = 0
    for chunk in source_chunks:
        chunk = str(chunk)
        chunk_for_match = chunk
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            chunk_sentence_ids.append(sent_idx)
            continue

        start = source_text_full.find(chunk_for_match, cursor)
        if start < 0:
            start = source_text_full.find(chunk_for_match)
        if start < 0 and chunk_stripped != chunk_for_match:
            start = source_text_full.find(chunk_stripped, cursor)
        if start < 0 and chunk_stripped != chunk_for_match:
            start = source_text_full.find(chunk_stripped)
        if start < 0:
            raise ValueError(f"Cannot locate source chunk in src_text: {chunk!r}")
        end = start + len(chunk_stripped if chunk_stripped != chunk_for_match and source_text_full[start:start + len(chunk_stripped)] == chunk_stripped else chunk_for_match)
        mid = (start + end) / 2.0
        while sent_idx + 1 < len(sentence_spans) and mid >= sentence_spans[sent_idx]["char_span"][1]:
            sent_idx += 1
        chunk_sentence_ids.append(sent_idx)
        cursor = end
    return chunk_sentence_ids


def group_contiguous_indices(ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not ids:
        return []
    groups: List[Tuple[int, int, int]] = []
    start = 0
    current = int(ids[0])
    for i in range(1, len(ids)):
        if int(ids[i]) != current:
            groups.append((current, start, i))
            start = i
            current = int(ids[i])
    groups.append((current, start, len(ids)))
    return groups


def adjust_target_boundary(target_text: str, boundary: int) -> int:
    """
    当 token 分组边界落在一个词内部时，把边界往右推到词尾，
    避免德语等空格语言在重建时出现 `ein st` 这类伪切分。
    """
    n = len(target_text)
    boundary = max(0, min(int(boundary), n))
    while 0 < boundary < n and target_text[boundary - 1].isalnum() and target_text[boundary].isalnum():
        boundary += 1
    return min(boundary, n)


def build_token_trajectory(
    target_text: str,
    target_token_offsets: Sequence[Tuple[int, int]],
    assigned_chunks: Sequence[int],
    num_source_steps: int,
) -> List[str]:
    """
    [Step 5/6] 基于 offset_mapping 直接从原始 target text 切分。
    这样可以保留 tokenizer 在分块 decode 时容易丢掉的空格和词内部边界。
    """
    trajectory: List[str] = [""] * num_source_steps
    if len(assigned_chunks) == 0:
        return trajectory

    groups: List[Tuple[int, int, int]] = []
    current_chunk = int(assigned_chunks[0])
    group_start = 0
    for j in range(1, len(assigned_chunks)):
        chunk = int(assigned_chunks[j])
        if chunk == current_chunk:
            continue
        else:
            groups.append((current_chunk, group_start, j))
            current_chunk = chunk
            group_start = j
    groups.append((current_chunk, group_start, len(assigned_chunks)))

    total_len = len(target_text)
    raw_boundaries = [0]
    for _, start_tok, _ in groups[1:]:
        raw_start = int(target_token_offsets[start_tok][0]) if start_tok < len(target_token_offsets) else total_len
        raw_boundaries.append(adjust_target_boundary(target_text, raw_start))
    raw_boundaries.append(total_len)

    for group_idx, (chunk_idx, _, _) in enumerate(groups):
        char_start = raw_boundaries[group_idx]
        char_end = raw_boundaries[group_idx + 1]
        decoded = target_text[char_start:char_end]
        safe_idx = max(0, min(int(chunk_idx), num_source_steps - 1))
        trajectory[safe_idx] += decoded
    return trajectory


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    sentence_end_chars = get_lang_spec(record.get("target_lang", "zh"))["sentence_end_chars"]
    return {
        "id": record["id"],
        "src_text_full": sentence_text_list(record["source_text_full"], ".!?"),
        "source_text_full": record["source_text_full"],
        "target_lang": record.get("target_lang", "zh"),
        "original_target_full": record.get("original_target_full", ""),
        "target_full": record["target_full"],
        "target_sentences": sentence_text_list(record["target_full"], sentence_end_chars),
        "src_trajectory": record["src_trajectory"],
        "target_trajectory": record["target_trajectory"],
    }


def write_pretty_json(output_dir: Path, records: Sequence[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_single_pretty_json(output_dir, record)


def sanitize_filename(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw).strip())


def get_pretty_json_path(output_dir: Path, record_id: str) -> Path:
    file_stem = sanitize_filename(record_id) or "unknown_id"
    return output_dir / f"{file_stem}.json"


def write_single_pretty_json(output_dir: Path, record: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_record(record)
    output_path = get_pretty_json_path(output_dir, str(summary["id"]))
    tmp_path = output_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    return output_path


def is_valid_pretty_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            return False
        return isinstance(item, dict) and bool(str(item.get("id", "")).strip())
    except Exception:
        return False


def _fmt_float(x: float) -> str:
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    return f"{float(x):.4f}"


def write_debug_log(
    path: Path,
    record: Dict[str, Any],
    scores: np.ndarray,
    deltas: np.ndarray,
    assigned_chunks: Sequence[int],
    monotonic_chunks: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_chunks = record["src_trajectory"]
    source_prefixes = record["source_prefixes"]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"id: {record['id']}\n")
        f.write(f"target_lang: {record.get('target_lang', 'zh')}\n")
        f.write(f"source_text_full: {record['source_text_full']}\n")
        f.write(f"target_full: {record['target_full']}\n\n")

        f.write("== Source Steps ==\n")
        for k, (chunk, prefix) in enumerate(zip(source_chunks, source_prefixes)):
            f.write(f"[{k:03d}] chunk={repr(chunk)}\n")
            f.write(f"      prefix={repr(prefix)}\n")
        f.write("\n")

        f.write("== Token-Level Assignment ==\n")
        for j, (tid, raw_tok, dec_tok, offset) in enumerate(
            zip(
                record["target_token_ids"],
                record["target_token_raw"],
                record["target_token_decoded"],
                record["target_token_offsets"],
            )
        ):
            best = int(assigned_chunks[j])
            mono = int(monotonic_chunks[j])
            f.write(
                f"[token {j:03d}] id={tid:<8d} raw={repr(raw_tok)} decoded={repr(dec_tok)} "
                f"offset={tuple(offset)} best_chunk={best} mono_chunk={mono}\n"
            )
            score_items = "  ".join(
                f"{k}:{_fmt_float(scores[j, k])}" for k in range(scores.shape[1])
            )
            delta_items = "  ".join(
                f"{k}:{_fmt_float(deltas[j, k])}" for k in range(deltas.shape[1])
            )
            f.write(f"  scores: {score_items}\n")
            f.write(f"  deltas: {delta_items}\n")
        f.write("\n")

        f.write("== Chunk-Level Token Groups ==\n")
        token_ids = record["target_token_ids"]
        mono_chunks = record["assigned_chunk_per_token_monotonic"]
        if token_ids:
            start = 0
            current_chunk = int(mono_chunks[0])
            for idx in range(1, len(token_ids) + 1):
                boundary = idx == len(token_ids) or int(mono_chunks[idx]) != current_chunk
                if not boundary:
                    continue
                group_ids = token_ids[start:idx]
                group_decoded = decode_token_ids_exact(tokenizer, group_ids)
                f.write(
                    f"[group {start:03d}-{idx - 1:03d}] chunk={current_chunk} "
                    f"text={repr(group_decoded)}\n"
                )
                if idx < len(token_ids):
                    start = idx
                    current_chunk = int(mono_chunks[idx])
        f.write("\n")

        f.write("== Final Trajectory ==\n")
        for step_idx, (src_chunk, tgt_chunk) in enumerate(
            zip(record["src_trajectory"], record["target_trajectory"])
        ):
            f.write(
                f"[step {step_idx:03d}] src={repr(src_chunk)}  tgt={repr(tgt_chunk)}\n"
            )


def ensure_vllm_cuda_platform() -> None:
    """
    vLLM 0.11 may occasionally fail to auto-detect CUDA in interactive shells
    even when torch can see GPUs. When that happens, patch the cached platform
    object to CudaPlatform before constructing LLM().
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return

        import vllm.platforms as vllm_platforms
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", ""):
            return

        from vllm.platforms.cuda import CudaPlatform
        from vllm.engine import arg_utils as vllm_arg_utils

        cuda_platform = CudaPlatform()
        vllm_platforms._current_platform = cuda_platform
        vllm_arg_utils.current_platform = cuda_platform
        print("[Info] Patched vLLM platform detection to CUDA via torch fallback.", flush=True)
    except Exception as e:
        print(f"[Warn] Failed to patch vLLM CUDA platform fallback: {e}", flush=True)


def build_local_alignment_record(
    tokenizer: Any,
    base_llm: Any,
    source_chunks: Sequence[str],
    target_text: str,
    target_lang: str,
) -> Dict[str, Any]:
    source_chunks = [str(x) for x in source_chunks]
    source_prefixes = build_cumulative_prefixes(source_chunks)
    token_ids, raw_tokens, decoded_tokens, offsets = tokenize_target_with_offsets(tokenizer, target_text)

    if len(token_ids) == 0 or len(source_prefixes) == 0:
        return {
            "source_prefixes": source_prefixes,
            "target_token_ids": token_ids,
            "target_token_raw": raw_tokens,
            "target_token_decoded": decoded_tokens,
            "target_token_offsets": [[int(s), int(e)] for s, e in offsets],
            "assigned_chunk_per_token_greedy": [],
            "assigned_chunk_per_token_monotonic": [],
            "target_trajectory": [""] * len(source_chunks),
            "scores": np.zeros((0, len(source_chunks)), dtype=np.float64),
            "deltas": np.zeros((0, len(source_chunks)), dtype=np.float64),
        }

    scores = compute_token_scores_over_prefixes(
        base_llm, tokenizer, source_prefixes, token_ids, target_lang
    )
    deltas, assigned_chunks = assign_chunks_from_deltas_greedy(scores)
    monotonic_chunks = repair_chunks_monotonic(assigned_chunks)
    target_trajectory = build_token_trajectory(
        target_text,
        offsets,
        monotonic_chunks,
        len(source_chunks),
    )
    return {
        "source_prefixes": source_prefixes,
        "target_token_ids": token_ids,
        "target_token_raw": raw_tokens,
        "target_token_decoded": decoded_tokens,
        "target_token_offsets": [[int(s), int(e)] for s, e in offsets],
        "assigned_chunk_per_token_greedy": [int(x) for x in assigned_chunks],
        "assigned_chunk_per_token_monotonic": [int(x) for x in monotonic_chunks],
        "target_trajectory": target_trajectory,
        "scores": scores,
        "deltas": deltas,
    }


def build_record_contextual_final(
    row: Dict[str, Any],
    tokenizer: Any,
    base_llm: Any,
    target_lang: str,
    target_list_column: str,
    target_full_column: str,
    debug_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    逐 sub-sentence 对齐，无 fallback。
    要求 TSV 同时有:
      - src_text_full: 源句列表 (e.g. ["sent1.", "sent2."])
      - llm_reference_text_list: 逐句翻译列表 (e.g. ["译文1。", "译文2。"])
      - src_trajectory: source chunk 列表
    src_text_full 和 llm_reference_text_list 长度必须一致（1:1 逐句对应）。
    """
    utt_id = str(row.get("id", "")).strip()
    source_chunks = parse_list_column(row.get("src_trajectory"))
    source_text_full = str(row.get("src_text", "")).strip()
    original_target_text = str(
        row.get(target_full_column)
        or row.get("target_full")
        or row.get("target_text")
        or row.get("target_full_zh")
        or row.get("target_full_de")
        or row.get("target_full_ja")
        or ""
    ).strip()

    # 解析逐句 source 和 target
    source_sentences = parse_list_column(row.get("src_text_full"))
    target_sentences = parse_list_column(row.get(target_list_column))

    if not source_sentences:
        raise ValueError(f"{utt_id}: empty src_text_full sentence list")
    if not target_sentences:
        raise ValueError(f"{utt_id}: empty {target_list_column}")
    if len(source_sentences) != len(target_sentences):
        raise ValueError(
            f"{utt_id}: len(src_text_full)={len(source_sentences)} != "
            f"len({target_list_column})={len(target_sentences)}"
        )

    # 拼接完整 target（用于输出）
    target_text = join_target_pieces(target_sentences, target_lang)

    if len(source_chunks) == 0:
        return {
            "id": utt_id,
            "target_lang": target_lang,
            "source_text_full": source_text_full,
            "original_target_full": original_target_text,
            "target_full": target_text,
            "src_trajectory": source_chunks,
            "target_trajectory": [],
        }

    # 把 source_chunks 按句子分组
    sentence_chunk_ids = assign_chunks_to_source_sentences(
        source_text_full,
        source_sentences,
        source_chunks,
    )
    sentence_chunk_groups = group_contiguous_indices(sentence_chunk_ids)
    if len(sentence_chunk_groups) != len(source_sentences):
        raise ValueError(
            f"{utt_id}: len(source sentence groups)={len(sentence_chunk_groups)} != "
            f"len(src_text_full)={len(source_sentences)}"
        )

    # 逐组对齐
    target_trajectory = [""] * len(source_chunks)
    merged_source_prefixes: List[str] = []
    merged_target_token_ids: List[int] = []
    merged_target_token_raw: List[str] = []
    merged_target_token_decoded: List[str] = []
    merged_target_token_offsets: List[List[int]] = []
    merged_mono_chunks: List[int] = []
    merged_greedy_chunks: List[int] = []
    merged_scores_rows: List[List[float]] = []
    merged_deltas_rows: List[List[float]] = []

    tgt_char_cursor = 0
    for group_idx, (_, start_idx, end_idx) in enumerate(sentence_chunk_groups):
        local_source_chunks = source_chunks[start_idx:end_idx]
        target_sentence = target_sentences[group_idx]

        if not target_sentence.strip() or not local_source_chunks:
            tgt_char_cursor += len(target_sentence)
            continue

        local_record = build_local_alignment_record(
            tokenizer,
            base_llm,
            local_source_chunks,
            target_sentence,
            target_lang,
        )
        for local_step, text_piece in enumerate(local_record["target_trajectory"]):
            target_trajectory[start_idx + local_step] += text_piece

        local_offsets = local_record["target_token_offsets"]
        for off in local_offsets:
            merged_target_token_offsets.append([off[0] + tgt_char_cursor, off[1] + tgt_char_cursor])

        merged_source_prefixes.extend(local_record["source_prefixes"])
        merged_target_token_ids.extend(local_record["target_token_ids"])
        merged_target_token_raw.extend(local_record["target_token_raw"])
        merged_target_token_decoded.extend(local_record["target_token_decoded"])
        merged_greedy_chunks.extend([start_idx + int(x) for x in local_record["assigned_chunk_per_token_greedy"]])
        merged_mono_chunks.extend([start_idx + int(x) for x in local_record["assigned_chunk_per_token_monotonic"]])
        for row_vals in local_record["scores"].tolist():
            full = [float("-inf")] * len(source_chunks)
            for local_k, val in enumerate(row_vals):
                full[start_idx + local_k] = val
            merged_scores_rows.append(full)
        for row_vals in local_record["deltas"].tolist():
            full = [float("-inf")] * len(source_chunks)
            for local_k, val in enumerate(row_vals):
                full[start_idx + local_k] = val
            merged_deltas_rows.append(full)
        tgt_char_cursor += len(target_sentence)

    record = {
        "id": utt_id,
        "target_lang": target_lang,
        "source_text_full": source_text_full,
        "original_target_full": original_target_text,
        "target_full": target_text,
        "src_trajectory": source_chunks,
        "source_prefixes": merged_source_prefixes,
        "target_token_ids": merged_target_token_ids,
        "target_token_raw": merged_target_token_raw,
        "target_token_decoded": merged_target_token_decoded,
        "target_token_offsets": merged_target_token_offsets,
        "assigned_chunk_per_token_greedy": merged_greedy_chunks,
        "assigned_chunk_per_token_monotonic": merged_mono_chunks,
        "target_trajectory": target_trajectory,
        "source_sentence_groups": sentence_chunk_groups,
        "source_sentences_used": source_sentences,
        "target_sentences_used": target_sentences,
    }
    if debug_log_path is not None:
        write_debug_log(
            debug_log_path,
            record,
            np.asarray(merged_scores_rows, dtype=np.float64),
            np.asarray(merged_deltas_rows, dtype=np.float64),
            merged_greedy_chunks,
            merged_mono_chunks,
        )
    return record


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    target_list_column = get_target_list_column(args)
    target_full_column = get_target_full_column(args)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = HIBIKI_DIR / "output" / get_lang_spec(args.target_lang)["output_dir_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM
    ensure_vllm_cuda_platform()

    base_llm = LLM(
        model=args.base_model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    rows = list(iter_rows(args.input_tsv, args.max_rows, args.task_id, args.num_tasks))
    written = 0
    skipped_existing = 0
    for row in tqdm(rows, desc="cases"):
        utt_id = str(row.get("id", "")).strip()
        output_path = get_pretty_json_path(output_dir, utt_id)
        if output_path.exists() and is_valid_pretty_json(output_path):
            skipped_existing += 1
            continue
        debug_log_path = None
        if args.debug_id and args.debug_log and utt_id == args.debug_id:
            debug_log_path = Path(args.debug_log)
        try:
            record = build_record_contextual_final(
                row,
                tokenizer,
                base_llm,
                args.target_lang,
                target_list_column,
                target_full_column,
                debug_log_path=debug_log_path,
            )
            write_single_pretty_json(output_dir, record)
            written += 1
        except ValueError as e:
            print(f"[Skip] {e}", flush=True)

    print(
        f"[Done] Wrote {written} per-id pretty json files to: {output_dir} "
        f"(skipped_existing={skipped_existing})",
        flush=True,
    )


if __name__ == "__main__":
    main()
