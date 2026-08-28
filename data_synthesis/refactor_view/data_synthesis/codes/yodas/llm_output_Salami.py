#!/usr/bin/env python3
import os
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # Data-parallel worker args
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)

    # vLLM args
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)

    # dataset selection
    parser.add_argument("--num-en", type=str, default="1")        # "1" or "all"
    parser.add_argument("--num-parquets", type=str, default="all") # not used heavily; kept for compatibility
    parser.add_argument("--num-samples", type=str, default="all")  # "all" or int

    # Debug: only run one parquet (8 digits e.g. 00000000)
    parser.add_argument("--only-parquet", type=str, default=None)

    # Paths
    parser.add_argument(
        "--output-root",
        type=str,
        default="/data/user_data/haolingp/outputs/llm_output_SALAMI",
        help="Output root directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Local model path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/yodas-granary/data",
        help="Yodas-granary data root",
    )

    return parser.parse_args()


args = parse_args()


# ============================================================
# Env (HF cache)
# ============================================================
os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Lang selection
# ============================================================
AVAILABLE_EN = ["en000", "en001", "en002", "en003", "en004"]
if args.num_en == "all":
    en_list = AVAILABLE_EN
else:
    en_list = AVAILABLE_EN[: int(args.num_en)]

num_samples = None if args.num_samples == "all" else int(args.num_samples)


# ============================================================
# Output dir
# ============================================================
output_root = args.output_root
os.makedirs(output_root, exist_ok=True)


# ============================================================
# Load model
# ============================================================
print(f"[Task {args.task_id}] Loading model TP={args.tp} ...")
llm = LLM(
    model=args.model_path,
    dtype="bfloat16",
    tensor_parallel_size=args.tp,
    max_model_len=16384,
    gpu_memory_utilization=0.90,
)


# ============================================================
# Salami JSON schema (guided decoding)
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

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    repetition_penalty=1.1,
    guided_decoding=GuidedDecodingParams(json=json_schema_salami),
)


# ============================================================
# Prompt (match screenshot style; only Chinese)
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
# DP split helper
# ============================================================
def split_list(lst: List[str], num_parts: int, part_id: int) -> List[str]:
    chunk = (len(lst) + num_parts - 1) // num_parts
    start = part_id * chunk
    end = min(len(lst), (part_id + 1) * chunk)
    return lst[start:end]


# ============================================================
# Batch processing
# ============================================================
def process_batch(batch_texts: List[str], batch_metas: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    batch_texts: list of English sentences
    batch_metas: list of dicts: {utt_id, out_json_path, text}
    """
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(t)},
        ]
        for t in batch_texts
    ]

    try:
        outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)
    except Exception as e:
        # whole batch failed
        for meta in batch_metas:
            with open(meta["out_json_path"], "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utt_id": meta["utt_id"],
                        "input": meta["text"],
                        "error": f"Batch error: {str(e)}",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return 0, len(batch_metas)

    success = 0
    fail = 0

    for i, out in enumerate(outputs):
        meta = batch_metas[i]
        utt_id = meta["utt_id"]
        out_json_path = meta["out_json_path"]
        text = meta["text"]

        try:
            response = out.outputs[0].text.strip()
            parsed = json.loads(response)  # guided decoding should guarantee JSON

            # attach metadata to make downstream easier
            parsed["input"] = text
            parsed["utt_id"] = utt_id

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)

            success += 1
        except Exception as e:
            fail += 1
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"utt_id": utt_id, "input": text, "error": str(e)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    return success, fail


# ============================================================
# Main
# ============================================================
global_success = 0
global_fail = 0

for lang_id in en_list:
    lang_root = os.path.join(output_root, lang_id)
    os.makedirs(lang_root, exist_ok=True)

    parquet_dir = f"{args.data_root}/{lang_id}/asr_only"
    all_parquets = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    # Optional: only one parquet
    if args.only_parquet is not None:
        all_parquets = [
            p for p in all_parquets
            if os.path.basename(p).startswith(args.only_parquet)
        ]

    # DP split parquets
    my_parquets = split_list(all_parquets, args.num_tasks, args.task_id)
    print(f"[Task {args.task_id}] {lang_id}: assigned {len(my_parquets)} parquets.")

    for pq_path in my_parquets:
        pq_name = os.path.basename(pq_path).replace(".parquet", "")
        pq_out_dir = os.path.join(lang_root, pq_name)
        os.makedirs(pq_out_dir, exist_ok=True)

        df = pd.read_parquet(pq_path)
        if num_samples is not None:
            df = df.iloc[:num_samples]

        batch_texts: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        pbar = tqdm(total=len(df), desc=f"{lang_id}_{pq_name}")

        for idx, row in df.iterrows():
            text = row["text"]
            utt_id = f"utt_{lang_id}_{pq_name}_{idx:04d}"
            out_json_path = os.path.join(pq_out_dir, f"{utt_id}.json")

            if os.path.exists(out_json_path):
                pbar.update(1)
                continue

            batch_texts.append(text)
            batch_metas.append({"utt_id": utt_id, "out_json_path": out_json_path, "text": text})

            if len(batch_texts) >= args.batch_size:
                s, f = process_batch(batch_texts, batch_metas)
                global_success += s
                global_fail += f

                pbar.update(len(batch_texts))
                batch_texts = []
                batch_metas = []

        # last batch
        if len(batch_texts) > 0:
            s, f = process_batch(batch_texts, batch_metas)
            global_success += s
            global_fail += f
            pbar.update(len(batch_texts))

        pbar.close()

print(f"[Task {args.task_id}] Success={global_success}, Fail={global_fail}")
