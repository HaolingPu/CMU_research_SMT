import os
import json
import glob
import pandas as pd
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# ============================================================
# 1. 配置
# ============================================================
os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="LLM segmentation pipeline")

    parser.add_argument("--num-en", type=str, default="1",
                    help="Number of enXXX datasets to process (e.g., 1,2,3 or 'all')")
                                                              
    parser.add_argument("--num-parquets", type=int, default=1,
                        help="Number of parquet files to process. "
                             "Use 'all' to process everything.")

    parser.add_argument("--num-samples", type=str, default="all",
                        help="Samples per parquet: 100, 200, or 'all' (default: all)")

    return parser.parse_args()


# ============================================================
# 2️⃣ Decide which parquet files to load
# ============================================================

args = parse_args()


AVAILABLE_EN = ["en000", "en001", "en002", "en003", "en004"]

if args.num_en == "all":
    en_list = AVAILABLE_EN
else:
    en_list = AVAILABLE_EN[:int(args.num_en)]

if args.num_samples == "all":
    num_samples = None
else:
    num_samples = int(args.num_samples)

print("\n===========================================")
print("🔹 Will process EN datasets:", en_list)
print("===========================================\n")


# ============================================================
# 3. Output Root
# ============================================================
output_root = "/data/user_data/haolingp/outputs/llm_segmentation_json"
os.makedirs(output_root, exist_ok=True)

# ============================================================
# 4. Load LLM Model
# ============================================================
model_path = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
max_new_tokens = 2048

print(f"🚀 Loading model {model_path}")
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=16384,
    gpu_memory_utilization=0.90
)
print("✅ Model loaded.\n")




# ============================================================
# 5. JSON schema
# ============================================================
json_schema = {
    "type": "object",
    "properties": {
        "low_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["English", "Chinese"]
        },
        "medium_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["English", "Chinese"]
        },
        "high_latency": {
            "type": "object",
            "properties": {
                "English": {"type": "array", "items": {"type": "string"}},
                "Chinese": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["English", "Chinese"]
        }
    },
    "required": ["low_latency", "medium_latency", "high_latency"]
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=max_new_tokens,
    repetition_penalty=1.1,
    guided_decoding=GuidedDecodingParams(json=json_schema)
)

print("✅ Schema loaded.\n")

# ============================================================
# 4️⃣ Prompt 构建
# ============================================================
def build_prompt(english_sentence):
    return f"""You are a professional English-to-Chinese simultaneous interpreter.

Task: Segment the English sentence into THREE different granularities and translate each segment to Chinese.

**CRITICAL RULES:**
For EVERY granularity (low_latency, medium_latency, high_latency):
- The English array MUST have EXACTLY the same number of items as the Chinese array
- If English has N segments, Chinese MUST have N segments (no more, no less)
- Each English[i] MUST correspond to Chinese[i] at the SAME index
- DO NOT merge multiple English segments into one Chinese segment
- DO NOT skip any segments

==================== NEW CHUNK QUALITY RULES ====================

Each English chunk MUST satisfy ALL of the following:

A. Minimum length requirement:
   - A chunk MUST contain at least TWO meaningful English words.
   - Single-word chunks are NOT allowed.

B. No punctuation chunks:
   - A chunk CANNOT be a standalone symbol such as:
     ".", "?", "!", ",", "...", "-", ":", ";", "。", "，"
   - A chunk composed ONLY of punctuation is strictly forbidden.

C. No empty chunks
   - Segments must contain actual semantic content.

   
**Granularity Definitions:**

**low_latency** (FINEST grain):
- Split into MANY SHORT segments (1-3 words each)
- Break at natural phrase boundaries
- Keep functional words separate when possible

**medium_latency** (MEDIUM grain):
- Split into FEWER MEDIUM segments (3-8 words each)
- Combine related phrases
- Break at major clause boundaries

**high_latency** (COARSE grain):
- Split ONLY at major punctuation (periods, semicolons, commas between independent clauses)
- Each segment = one complete clause or sentence
- DO NOT output the entire input as one segment if it contains multiple clauses

---

Example 1:
Input: "Houston issued a series of tornado warnings on the evening of the 16th."

{{
  "low_latency": {{
    "English": ["Houston", "issued", "a series of", "tornado warnings", "on the evening", "of the 16th."],
    "Chinese": ["休斯敦", "发出了", "一系列", "龙卷风警报", "在傍晚", "16日。"]
  }},
  "medium_latency": {{
    "English": ["Houston issued", "a series of tornado warnings", "on the evening of the 16th."],
    "Chinese": ["休斯敦发出了", "一系列龙卷风警报", "在16日傍晚。"]
  }},
  "high_latency": {{
    "English": ["Houston issued a series of tornado warnings on the evening of the 16th."],
    "Chinese": ["休斯敦在16日傍晚发出了一系列龙卷风警报。"]
  }}
}}

Example 2:
Input: "The company announced new features and improved performance yesterday."

{{
  "low_latency": {{
    "English": ["The company", "announced", "new features", "and", "improved performance", "yesterday."],
    "Chinese": ["该公司", "宣布了", "新功能", "以及", "改进的性能", "昨天。"]
  }},
  "medium_latency": {{
    "English": ["The company announced", "new features and improved performance", "yesterday."],
    "Chinese": ["该公司宣布了", "新功能和改进的性能", "昨天。"]
  }},
  "high_latency": {{
    "English": ["The company announced new features and improved performance yesterday."],
    "Chinese": ["该公司昨天宣布了新功能和改进的性能。"]
  }}
}}

Example 3 (Multiple clauses with punctuation):
Input: "The weather was sunny in the morning, but it started raining in the afternoon, and the temperature dropped significantly."

{{
  "low_latency": {{
    "English": ["The weather", "was sunny", "in the morning,", "but", "it started", "raining", "in the afternoon,", "and", "the temperature", "dropped", "significantly."],
    "Chinese": ["天气", "是晴朗的", "在早上，", "但是", "开始", "下雨", "在下午，", "并且", "温度", "下降了", "显著。"]
  }},
  "medium_latency": {{
    "English": ["The weather was sunny in the morning,", "but it started raining in the afternoon,", "and the temperature dropped significantly."],
    "Chinese": ["早上天气晴朗，", "但下午开始下雨，", "温度显著下降。"]
  }},
  "high_latency": {{
    "English": ["The weather was sunny in the morning,", "but it started raining in the afternoon,", "and the temperature dropped significantly."],
    "Chinese": ["早上天气晴朗，", "但下午开始下雨，", "温度显著下降。"]
  }}
}}

Example 4 (Short sentence - no punctuation to split):
Input: "She loves reading books."

{{
  "low_latency": {{
    "English": ["She", "loves", "reading", "books."],
    "Chinese": ["她", "喜欢", "阅读", "书籍。"]
  }},
  "medium_latency": {{
    "English": ["She loves", "reading books."],
    "Chinese": ["她喜欢", "阅读书籍。"]
  }},
  "high_latency": {{
    "English": ["She loves reading books."],
    "Chinese": ["她喜欢阅读书籍。"]
  }}
}}

---

Now process this input:
Input: "{english_sentence}"

**REMEMBER:**
- For high_latency: Split at commas, periods, semicolons that separate clauses
- NEVER output the entire sentence as one segment if it has multiple clauses
- English and Chinese arrays MUST have the same length
- Output ONLY the JSON object, no explanations"""


# ============================================================
# 5️⃣ 执行 LLM segmentation
# ============================================================
global_success = 0
global_fail = 0


for lang_id in en_list:
    print(f"\n===============================")
    print(f"🌍 Processing dataset: {lang_id}")
    print("===============================\n")

    parquet_dir = f"/data/group_data/li_lab/siqiouya/datasets/yodas-granary/data/{lang_id}/asr_only"
    all_parquets = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    if args.num_parquets == "all":
        parquet_files = all_parquets
    else:
        parquet_files = all_parquets[:int(args.num_parquets)]

    print(f"🔹 Total parquet files: {len(all_parquets)}")
    print(f"🔹 Will process: {len(parquet_files)}")

    for pq_path in parquet_files:

        pq_name = os.path.basename(pq_path).replace(".parquet", "")
        pq_output_dir = os.path.join(output_root, f"{lang_id}/{pq_name}")
        os.makedirs(pq_output_dir, exist_ok=True)

        df = pd.read_parquet(pq_path)
        if num_samples is not None:
            df = df.iloc[:num_samples]

        print(f"\n📌 {lang_id} / {pq_name}: {len(df)} rows")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{lang_id}_{pq_name}"):

          text = row["text"]

          # === Unified uttid naming ===
          utt_id = f"utt_{lang_id}_{pq_name}_{idx:04d}"
          out_json_path = os.path.join(pq_output_dir, f"{utt_id}.json")

          # skip existing (auto resume)
          if os.path.exists(out_json_path):
              continue

          prompt = build_prompt(text)

          try:
              outputs = llm.chat(
                  messages=[{"role": "user", "content": prompt}],
                  sampling_params=sampling_params
              )
              response = outputs[0].outputs[0].text.strip()

              parsed = json.loads(response)
              parsed["input"] = text
              parsed["utt_id"] = utt_id  # <-- store utt_id inside JSON

              with open(out_json_path, "w", encoding="utf-8") as f:
                  json.dump(parsed, f, ensure_ascii=False, indent=2)

              global_success += 1

          except Exception as e:
              global_fail += 1
              with open(out_json_path, "w", encoding="utf-8") as f:
                  json.dump({
                      "utt_id": utt_id,
                      "input": text,
                      "error": str(e),
                      "raw_output": response if "response" in locals() else None
                  }, f, ensure_ascii=False, indent=2)

# ============================================================
# 6️⃣ Summary
# ============================================================
print("\n=== DONE ===")
print(f"✅ Success: {global_success}")
print(f"❌ Failed : {global_fail}")
print(f"📂 Output root: {output_root}\n")