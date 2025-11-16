import os
import re
import json
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ============================================================
# 1️⃣ 参数设置
# ============================================================
os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"

tsv_path = "/data/user_data/haolingp/datasets/granary_en_1000h_manifest.tsv"
num_sentences = 10
output_dir = "/data/user_data/haolingp/segment_results_json_vllm_chat_template"
os.makedirs(output_dir, exist_ok=True)

# 指定模型路径
model_path = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
max_new_tokens = 2048

print(f"🚀 Loading vLLM model from {model_path}")
llm = LLM(
    model=model_path,
    dtype="bfloat16", 
    tensor_parallel_size=1,
    max_model_len=16384,              # ✅ 降低上下文长度
    gpu_memory_utilization=0.90  
    )
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=max_new_tokens,
    repetition_penalty=1.1
)
print("✅ vLLM engine initialized successfully.\n")

# ============================================================
# 2️⃣ 读取 TSV 文件
# ============================================================
print(f"📖 Loading TSV: {tsv_path}")
df = pd.read_csv(tsv_path, sep="\t", header=None, 
                 names=["audio", "text", "duration","source"], 
                 dtype=str)
sentences = df["text"].dropna().tolist()[:num_sentences]
print(f"✅ Loaded {len(sentences)} samples.\n")

# ============================================================
# 3️⃣ Prompt 构建函数
# ============================================================
def build_prompt(english_sentence):
    return f"""You are a professional English-to-Chinese simultaneous interpreter.

Task: Segment the English sentence into THREE different granularities and translate each segment to Chinese.

**IMPORTANT - Granularity Definition:**
- **low_latency** (FINEST grain) → Many SHORT segments, each 1-3 words
- **medium_latency** (MEDIUM grain) → Fewer MEDIUM segments, each 3-8 words  
- **high_latency** (COARSEST grain) → Very few LONG segments or one complete sentence

Output ONLY valid JSON in this exact format:

Example 1:
Input: "Houston issued a series of tornado warnings on the evening of the 16th."

{{
  "low_latency": {{
    "English": ["Houston", "issued", "a series of", "tornado warnings", "on the evening", "of the 16th."],
    "Chinese": ["休斯敦", "发出了", "一系列", "龙卷风警报", "在傍晚", "16日。"]
  }},
  "medium_latency": {{
    "English": ["Houston issued a series of", "tornado warnings", "on the evening of the 16th."],
    "Chinese": ["休斯敦发出了一系列", "龙卷风警报", "在16日傍晚。"]
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

Now process this input:
Input: "{english_sentence}"

Output ONLY the JSON object, no explanations:"""




# ============================================================
# 4️⃣ JSON 提取函数
# ============================================================
def extract_json_from_response(response_text):
    response_text = response_text.strip()
    response_text = re.sub(r"```(?:json)?", "", response_text)
    start = response_text.find('{')
    if start == -1:
        return None
    count = 0
    for i in range(start, len(response_text)):
        if response_text[i] == '{':
            count += 1
        elif response_text[i] == '}':
            count -= 1
            if count == 0:
                json_str = response_text[start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None

# ============================================================
# 5️⃣ 主循环（vLLM生成）
# ============================================================
successful, failed = 0, 0

for i, sentence in enumerate(tqdm(sentences, desc="Processing", ncols=100)):
    json_path = os.path.join(output_dir, f"{i:05d}.json")
    raw_path = os.path.join(output_dir, f"{i:05d}_raw.txt")

    prompt = build_prompt(sentence)
    messages = [[{"role": "user", "content": prompt}]]

    # ✅ 使用 llm.chat (自动应用 chat template)
    outputs = llm.chat(messages, sampling_params)
    response = outputs[0].outputs[0].text.strip()

    num_tokens = len(output.outputs[0].token_ids)
    print(f"📊 Generated {num_tokens} tokens")

    # 查看生成的文本
    text = output.outputs[0].text
    print(f"📝 Text length: {len(text)} characters")

    print("\n" + "="*80)
    print(f"📝 INPUT: {sentence[:80]}...")
    print("-"*80)
    print(f"🤖 OUTPUT ({len(response)} chars):")
    print(response[:500] + "..." if len(response) > 500 else response)
    print("="*80 + "\n")

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"Input:\n{sentence}\n\nOutput:\n{response}\n")

    result_json = extract_json_from_response(response)
    if result_json:
        result_json["input"] = sentence
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        successful += 1
    else:
        error_obj = {
            "input": sentence,
            "error": "Failed to parse JSON",
            "raw_output": response[:800],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(error_obj, f, ensure_ascii=False, indent=2)
        failed += 1

# ============================================================
# 6️⃣ 汇总统计
# ============================================================
print("\n" + "="*60)
print(f"✅ Processing complete")
print(f"📊 Success: {successful}/{len(sentences)}")
print(f"❌ Failed:  {failed}/{len(sentences)}")
print(f"📁 Results saved in: {output_dir}")
print("="*60)
