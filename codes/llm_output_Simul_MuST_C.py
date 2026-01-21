import os
import json
import glob
import pandas as pd
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from __future__ import annotations
from typing import Dict, List, Any

# ============================================================
# Arguments for DP worker
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)  # NEW: batch size

    parser.add_argument("--num-en", type=str, default="1")
    parser.add_argument("--num-parquets", type=str, default="all")
    parser.add_argument("--num-samples", type=str, default="all")

    return parser.parse_args()

args = parse_args()

# ============================================================
# Config
# ============================================================
os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

AVAILABLE_EN = ["en000", "en001", "en002", "en003", "en004"]

if args.num_en == "all":
    en_list = AVAILABLE_EN
else:
    en_list = AVAILABLE_EN[:int(args.num_en)]

num_samples = None if args.num_samples == "all" else int(args.num_samples)

output_root = "/data/user_data/haolingp/outputs/llm_output_EAST"
os.makedirs(output_root, exist_ok=True)

# ============================================================
# Load model
# ============================================================
print(f"[Task {args.task_id}] Loading model TP={args.tp} ...")

model_path = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    tensor_parallel_size=args.tp,
    max_model_len=16384,
    gpu_memory_utilization=0.90
)

# ============================================================
# JSON schema
# ============================================================
json_schema = {
    "type": "object",
    "properties": {
        "low_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["English", "Chinese"],
        },
        "medium_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["English", "Chinese"],
        },
        "high_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["English", "Chinese"],
        },
    },
    "required": ["low_latency", "medium_latency", "high_latency"],
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    repetition_penalty=1.1,
    guided_decoding=GuidedDecodingParams(json=json_schema),
)

# ============================================================
# Build prompt
# ============================================================
_EXAMPLE_TEXT = (
    "Almost every way we make electricity today except for the emerging renewables and nuclear puts out CO2."
)

_EXAMPLE_OUTPUTS: Dict[str, Dict[str, Any]] = {
    "Japanese": {
        "segmented_pairs": [
            ["Almost every way", "ほとんどすべての方法は"],
            ["we make electricity today", "私たちが今日電気を作る"],
            ["except for the emerging renewables and nuclear", "新興の再生可能エネルギーと原子力を除いて"],
            ["puts out CO2", "CO2を排出します"],
        ],
        "output": "ほとんどすべての方法で、私たちが今日電気を作るのは、新興の再生可能エネルギーと原子力を除いて、CO2を排出します",
    },
    "Chinese": {
        "segmented_pairs": [
            ["Almost every way", "几乎每一种方式"],
            ["we make electricity today", "我们今天发电的方式"],
            ["except for the emerging renewables and nuclear", "除了新兴的可再生能源和核能"],
            ["puts out CO2", "会排放二氧化碳"],
        ],
        "output": "几乎每一种我们今天发电的方式，除了新兴的可再生能源和核能，都会排放二氧化碳。",
    },
    "German": {
        "segmented_pairs": [
            ["Almost every way we make electricity today", "Fast jede Art, wie wir heute Strom erzeugen,"],
            ["except for the emerging renewables and nuclear", "außer den aufkommenden erneuerbaren Energien und der Kernenergie,"],
            ["puts out CO2", "stößt CO2 aus."],
        ],
        "output": "Fast jede Art, wie wir heute Strom erzeugen, außer den aufkommenden erneuerbaren Energien und der Kernenergie, stößt CO2 aus.",
    },
}


def build_simul_must_baseline_prompt(text: str, target_language: str) -> List[Dict[str, str]]:
    """
    Paper baseline prompt for Simul-MuST-C style segmentation + translation.
    Assumes ONLY three target languages: Chinese, Japanese, German.
    Always includes the one-shot example from the paper figure.

    Returns chat messages: [{"role":"system","content":...}, {"role":"user","content":...}]
    """
    if target_language not in _EXAMPLE_OUTPUTS:
        raise ValueError(f"Unsupported target_language={target_language}. Must be one of: {list(_EXAMPLE_OUTPUTS)}")

    system_msg = (
        f"You will be provided with a sentence in English, and your task is to interpret it into {target_language}.\n"
        "Always answer in the following JSON format:\n"
        "{'segmented_pairs': List[Tuple[English, Language]], 'output': Language}"
    )

    example_output = _EXAMPLE_OUTPUTS[target_language]

    user_msg = (
        "Instructions: 'Salami technique' in simultaneous interpretation refers to a technique where the interpreter "
        "breaks down the source language input into smaller, manageable segments that each contain enough information "
        "to be accurately interpreted.\n"
        "1. Break down the following sentence into smaller segments for easier simultaneous interpretation.\n"
        f"2. Translate each segment into {target_language}.\n"
        "3. Connect the translated segments.\n\n"
        "Example Text:\n"
        f"{_EXAMPLE_TEXT}\n\n"
        "Example Output:\n"
        f"{example_output}\n\n"
        "Inputs:\n"
        f"{text}\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

# ============================================================
# Worker DP Split Logic
# ============================================================
def split_list(lst, num_parts):
    chunk = (len(lst) + num_parts - 1) // num_parts
    start = args.task_id * chunk
    end = min(len(lst), (args.task_id + 1) * chunk)
    return lst[start:end]


# ============================================================
# NEW: Batch processing function
# ============================================================
def process_batch(batch_data, batch_metadata):
    """
    batch_data: list of text strings to process
    batch_metadata: list of dicts with {utt_id, out_json_path, text}
    """
    # Build all prompts
    prompts = [build_prompt_EAST(text) for text in batch_data]
    
    # Prepare messages for vLLM chat API (list of conversations)
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    
    try:
        # Batch inference
        outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)
        
        success_count = 0
        fail_count = 0
        
        # Process each output
        for i, output in enumerate(outputs):
            meta = batch_metadata[i]
            utt_id = meta["utt_id"]
            out_json_path = meta["out_json_path"]
            text = meta["text"]
            
            try:
                response = output.outputs[0].text.strip()
                parsed = json.loads(response)
                parsed["input"] = text
                parsed["utt_id"] = utt_id
                
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                with open(out_json_path, "w") as f:
                    json.dump({
                        "utt_id": utt_id,
                        "input": text,
                        "error": str(e)
                    }, f, indent=2)
        
        return success_count, fail_count
        
    except Exception as e:
        # Entire batch failed
        print(f"Batch processing failed: {e}")
        fail_count = len(batch_metadata)
        
        # Save error for each sample in batch
        for meta in batch_metadata:
            with open(meta["out_json_path"], "w") as f:
                json.dump({
                    "utt_id": meta["utt_id"],
                    "input": meta["text"],
                    "error": f"Batch error: {str(e)}"
                }, f, indent=2)
        
        return 0, fail_count


# ============================================================
# Run segmentation with batching
# ============================================================
global_success = 0
global_fail = 0

for lang_id in en_list:
    # create en000/
    lang_root = os.path.join(output_root, lang_id)
    os.makedirs(lang_root, exist_ok=True)

    parquet_dir = f"/data/group_data/li_lab/siqiouya/datasets/yodas-granary/data/{lang_id}/asr_only"
    all_parquets = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    # DP split here
    my_parquets = split_list(all_parquets, args.num_tasks)

    print(f"[Task {args.task_id}] {lang_id}: assigned {len(my_parquets)} parquets.")

    for pq_path in my_parquets:
        pq_name = os.path.basename(pq_path).replace(".parquet", "")
        pq_out_dir = os.path.join(output_root, pq_name)
        os.makedirs(pq_out_dir, exist_ok=True)

        df = pd.read_parquet(pq_path)
        if num_samples is not None:
            df = df.iloc[:num_samples]

        # Collect samples for batching
        batch_data = []
        batch_metadata = []
        
        # Progress bar
        pbar = tqdm(total=len(df), desc=f"{lang_id}_{pq_name}")
        
        for idx, row in df.iterrows():
            text = row["text"]
            utt_id = f"utt_{lang_id}_{pq_name}_{idx:04d}"
            out_json_path = os.path.join(pq_out_dir, f"{utt_id}.json")

            # Skip if already exists
            if os.path.exists(out_json_path):
                pbar.update(1)
                continue

            # Add to batch
            batch_data.append(text)
            batch_metadata.append({
                "utt_id": utt_id,
                "out_json_path": out_json_path,
                "text": text
            })

            # Process when batch is full
            if len(batch_data) >= args.batch_size:
                success, fail = process_batch(batch_data, batch_metadata)
                global_success += success
                global_fail += fail
                
                pbar.update(len(batch_data))
                
                # Clear batch
                batch_data = []
                batch_metadata = []

        # Process remaining samples in the last batch
        if len(batch_data) > 0:
            success, fail = process_batch(batch_data, batch_metadata)
            global_success += success
            global_fail += fail
            pbar.update(len(batch_data))
        
        pbar.close()

print(f"[Task {args.task_id}] Success={global_success}, Fail={global_fail}")