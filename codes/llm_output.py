import os
import json
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# ============================================================
# 1️⃣ 配置
# ============================================================
os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"

parquet_path = "/data/hf_cache/yodas-granary/data/en000/asr_only/00000000.parquet"
num_samples = 100
output_dir = "/data/user_data/haolingp/outputs/llm_segmentation_json"
output_reference_dir = "/data/user_data/haolingp/outputs/llm_reference_json"
os.makedirs(output_dir, exist_ok=True)

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


# ============================================================
# 2️⃣ JSON schema
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

sampling_params_reference = SamplingParams(
    temperature=0.0,
    max_tokens=512,  # 参考翻译不需要那么多tokens
    repetition_penalty=1.1
    # ✅ 没有 guided_decoding！
)

print("✅ Schema loaded.\n")

# ============================================================
# 3️⃣ 读取 parquet 文件
# ============================================================
df = pd.read_parquet(parquet_path)
df = df.iloc[:num_samples]

print(f"📖 Loaded {len(df)} rows from parquet.\n")

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



def build_reference_prompt(english_sentence):
    return f"""
        Translate the following English sentence into fluent, high-quality Chinese:
        {english_sentence}

        Rules:
        - Do NOT segment.
        - Produce ONE single Chinese sentence.
        - No explanations.
        """.strip()



# ============================================================
# 5️⃣ 执行 LLM segmentation
# ============================================================
successful, failed = 0, 0

for new_id, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing")):

    utt_id = f"utt_{new_id:06d}" 
    text = row["text"]

    out_json_path = os.path.join(output_dir, f"{utt_id}.json")
    raw_path = os.path.join(output_dir, f"{utt_id}_raw.txt")

    prompt = build_prompt(text)
    ref_prompt = build_reference_prompt(text) # build offline reference

    try:
        # streaming segmentation
        outputs = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=sampling_params
        )
        result = outputs[0].outputs[0]
        response = result.text.strip()

        parsed = json.loads(response)
        parsed["input"] = text

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)


        # offline reference
        ref_outputs = llm.chat(
            messages=[{"role": "user", "content": ref_prompt}],
            sampling_params=sampling_params_reference
        )
        ref_result = ref_outputs[0].outputs[0]
        offline_translation = ref_result.text.strip()

        ref_json = {
            "utt_id": utt_id,
            "input": text,
            "offline_reference": offline_translation,
        }


        os.makedirs(output_reference_dir, exist_ok=True)
        second_path = os.path.join(output_reference_dir, f"{utt_id}.json")
        with open(second_path, "w", encoding="utf-8") as f2:
            json.dump(ref_json, f2, ensure_ascii=False, indent=2)



        successful += 1

    except Exception as e:
        failed += 1
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "utt_id": utt_id,
                "input": text,
                "error": str(e),
                "raw_output": response if "response" in locals() else None
            }, f, ensure_ascii=False, indent=2)

# ============================================================
# 6️⃣ 总结
# ============================================================
print("\n=== DONE ===")
print(f"Success: {successful}/{len(df)}")
print(f"Failed:  {failed}/{len(df)}")
print(f"Saved results in: {output_dir}")
