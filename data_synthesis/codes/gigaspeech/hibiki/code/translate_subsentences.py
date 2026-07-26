#!/usr/bin/env python3
"""
逐句翻译 source sub-sentences，生成带目标语言列的 shard TSV。

支持 --task-id / --num-tasks 分片，每个 shard 独立输出一个 TSV。
最后用 merge 模式合并所有 shard。
"""

import argparse
import ast
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any, List

DEFAULT_INPUT_TSV = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
)
DEFAULT_MODEL_PATH = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

_LANG_SPECS = {
    "zh": {
        "language_name": "Chinese",
        "output_list_column": "llm_reference_text_list",
        "output_full_column": "llm_reference_text",
        "output_root_name": "subsentence_ref_shards",
    },
    "de": {
        "language_name": "German",
        "output_list_column": "llm_reference_text_list_de",
        "output_full_column": "target_full_de",
        "output_root_name": "subsentence_ref_shards_de",
    },
    "ja": {
        "language_name": "Japanese",
        "output_list_column": "llm_reference_text_list_ja",
        "output_full_column": "target_full_ja",
        "output_root_name": "subsentence_ref_shards_ja",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    p.add_argument("--output-root", default="",
                   help="Output directory for shard TSVs")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--target-lang", choices=["zh", "de", "ja"], default="zh")
    p.add_argument("--output-list-column", default="")
    p.add_argument("--output-full-column", default="")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--merge", action="store_true",
                   help="Merge all shard TSVs into one final TSV (no GPU needed)")
    return p.parse_args()


def get_lang_spec(target_lang: str):
    if target_lang not in _LANG_SPECS:
        raise ValueError(f"Unsupported target_lang={target_lang!r}")
    return _LANG_SPECS[target_lang]


def get_output_list_column(args) -> str:
    if args.output_list_column:
        return args.output_list_column
    return get_lang_spec(args.target_lang)["output_list_column"]


def get_output_full_column(args) -> str:
    if args.output_full_column:
        return args.output_full_column
    return get_lang_spec(args.target_lang)["output_full_column"]


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


def join_pieces(pieces: List[str], target_lang: str) -> str:
    if target_lang in {"zh", "ja"}:
        return "".join(pieces)
    running = ""
    for piece in pieces:
        running = running + maybe_add_space_target(running, piece, target_lang) + piece
    return running


def build_translation_messages(source_sentence: str, target_lang: str, context: str = ""):
    language_name = get_lang_spec(target_lang)["language_name"]
    system_prompt = (
        "You are a professional translator. Translate the English source into "
        f"{language_name}. Output only the {language_name} translation, nothing else."
    )
    if context:
        user_content = (
            f"Context (for reference, do NOT translate this):\n{context}\n\n"
            f"Translate ONLY the following sentence into {language_name}:\n{source_sentence}"
        )
    else:
        user_content = f"Translate into {language_name}:\n{source_sentence}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def run_shard(args):
    # 确定输出路径
    if not args.output_root:
        inp = Path(args.input_tsv)
        args.output_root = str(inp.parent / get_lang_spec(args.target_lang)["output_root_name"])

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{args.task_id:03d}.tsv"

    # 读取数据 + 分片
    df = pd.read_csv(args.input_tsv, sep="\t")
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    # 按 task_id 分片
    shard_indices = list(range(args.task_id, len(df), args.num_tasks))
    df_shard = df.iloc[shard_indices].copy()
    df_shard = df_shard.reset_index(drop=True)

    print(f"[Shard {args.task_id}/{args.num_tasks}] {len(df_shard)} rows")

    # 收集翻译任务
    tasks = []
    row_sentence_counts = []
    output_list_column = get_output_list_column(args)
    output_full_column = get_output_full_column(args)

    for local_idx, (_, row) in enumerate(df_shard.iterrows()):
        sentences = parse_list_column(row.get("src_text_full"))
        row_sentence_counts.append(len(sentences))
        context = ""
        for sent_idx, sent in enumerate(sentences):
            msgs = build_translation_messages(sent.strip(), args.target_lang, context)
            tasks.append((local_idx, sent_idx, msgs))
            context += (" " if context else "") + sent.strip()

    print(f"[Shard {args.task_id}] Total sub-sentences: {len(tasks)}")

    # 加载模型
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    # 构建 prompt
    prompts = []
    for _, _, msgs in tasks:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # 推理
    all_outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size),
                  desc=f"Shard {args.task_id}"):
        batch = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            all_outputs.append(out.outputs[0].text.strip())

    # 组装结果
    result_lists = {}
    for (local_idx, sent_idx, _), translation in zip(tasks, all_outputs):
        if local_idx not in result_lists:
            result_lists[local_idx] = {}
        result_lists[local_idx][sent_idx] = translation

    ref_list_col = []
    ref_full_col = []
    for local_idx in range(len(df_shard)):
        n = row_sentence_counts[local_idx]
        trans = result_lists.get(local_idx, {})
        ordered = [trans.get(i, "") for i in range(n)]
        ref_list_col.append(str(ordered))
        ref_full_col.append(join_pieces(ordered, args.target_lang))

    df_shard[output_list_column] = ref_list_col
    df_shard[output_full_column] = ref_full_col

    # 保存原始全局 index 用于合并时排序
    df_shard["_global_idx"] = shard_indices

    df_shard.to_csv(output_path, sep="\t", index=False)
    print(f"[Shard {args.task_id}] Wrote {len(df_shard)} rows to {output_path}")


def run_merge(args):
    if not args.output_root:
        inp = Path(args.input_tsv)
        args.output_root = str(inp.parent / get_lang_spec(args.target_lang)["output_root_name"])

    output_dir = Path(args.output_root)
    shard_files = sorted(output_dir.glob("shard_*.tsv"))
    print(f"Found {len(shard_files)} shards in {output_dir}")

    dfs = [pd.read_csv(f, sep="\t") for f in shard_files]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("_global_idx").reset_index(drop=True)
    merged = merged.drop(columns=["_global_idx"])

    suffix = "" if args.target_lang == "zh" else f"_{args.target_lang}"
    final_path = output_dir.parent / (
        Path(args.input_tsv).stem + f"_subsentence_ref{suffix}.tsv"
    )
    merged.to_csv(final_path, sep="\t", index=False)
    print(f"[Merge] Wrote {len(merged)} rows to {final_path}")

    # 打印几个示例
    output_list_column = get_output_list_column(args)
    for i in range(min(3, len(merged))):
        row = merged.iloc[i]
        print(f"\n--- {row['id']} ---")
        src_list = parse_list_column(row["src_text_full"])
        tgt_list = parse_list_column(row[output_list_column])
        for j, (s, t) in enumerate(zip(src_list, tgt_list)):
            print(f"  [{j}] src: {s}")
            print(f"      tgt: {t}")


def main():
    args = parse_args()
    if args.merge:
        run_merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
