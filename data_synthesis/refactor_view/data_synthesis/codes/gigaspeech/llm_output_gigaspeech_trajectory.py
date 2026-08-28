#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import os
import re
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Set, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


LATENCY_LEVELS = ["low_latency", "medium_latency", "high_latency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EAST baseline sentence segmentation on GigaSpeech manifest TSV. "
            "Input is src_text_full (list of sentence strings), and output is one JSON per utt_id "
            "with flattened low/medium/high latency segment lists."
        )
    )
    parser.add_argument("--input-tsv", required=True, help="Path to manifest TSV file.")
    parser.add_argument("--output-root", required=True, help="Directory to write per-utt JSON results.")
    parser.add_argument(
        "--model-path",
        default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Local model path for vLLM.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sentence batch size for one vLLM chat call.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Data-parallel worker id (0-indexed).")
    parser.add_argument("--num-tasks", type=int, default=1, help="Total number of workers.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process first N assigned rows (debug).",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column name for utterance id in TSV.",
    )
    parser.add_argument(
        "--src-text-full-column",
        default="src_text_full",
        help="Column that stores list-like sentence strings.",
    )
    parser.add_argument(
        "--save-per-sentence",
        action="store_true",
        help="Also save sentence-level LLM outputs inside each JSON (bigger files, better debugging).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files. Default is resume (skip existing).",
    )
    parser.add_argument(
        "--strict-pair-check",
        action="store_true",
        help=(
            "Strictly require English/Chinese list lengths to match. "
            "Default is lenient mode: truncate to min length and continue."
        ),
    )
    parser.add_argument(
        "--retry-single-on-fail",
        action="store_true",
        help=(
            "If set, retry failed prompts with single-sentence calls. "
            "Default is throughput-first: do not retry singles."
        ),
    )
    return parser.parse_args()


def setup_env() -> None:
    os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_prompt_refined_east(english_sentence: str) -> str:
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



def build_prompt_east(english_sentence: str) -> str:
    prompt = """
As a professional simultaneous interpreter, your task is to segment sentences into independent
semantic chunks and provide corresponding Chinese translations.
You will use three different granularities for segmentation:
1. For low latency, the chunks would be fragmented into brief, coherent phrases that convey a complete thought.
2. For medium latency, the chunks would be longer, possibly clause or sentence-long segments.
3. For high latency, the chunks would be the longest, likely to cover complete clauses or full sentences.
You also need to provide corresponding simultaneous translation for each segment by performing
the translation monotonically while making the translation grammatically tolerable.

Input:
English: Houston issued a series of tornado and severe thunderstorm warnings on the evening of the 16th.

Output:
{
  "low_latency": {
    "English": ["Houston", "on the evening of the 16th", "issued a series of", "tornado", "and severe thunderstorm", "warnings."],
    "Chinese": ["休斯敦", "16日晚", "发出一系列", "龙卷风", "和严重雷暴", "警报。"]
  },
  "medium_latency": {
    "English": ["On the evening of the 16th, Houston", "issued a series of", "tornado and severe thunderstorm warnings."],
    "Chinese": ["休斯敦16日晚", "发出一系列", "龙卷风和严重雷暴警报。"]
  },
  "high_latency": {
    "English": ["On the evening of the 16th, Houston", "issued a series of tornado and severe thunderstorm warnings."],
    "Chinese": ["休斯敦16日晚", "发出一系列龙卷风和严重雷暴警报。"]
  }
}
"""
    return (
        prompt
        + "\nNow process the following input:\n"
        + f"English: {english_sentence}\n"
        + "Output:\n"
    )


JSON_SCHEMA = {
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
    "required": LATENCY_LEVELS,
}


def parse_src_text_full(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []

    raw = str(raw_value).strip()
    if raw == "":
        return []

    parsed: Any
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = raw

    if isinstance(parsed, list):
        out = []
        for item in parsed:
            sentence = str(item).strip()
            if sentence:
                out.append(sentence)
        return out

    sentence = str(parsed).strip()
    return [sentence] if sentence else []


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:200] if safe else "unknown_id"


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total_rows = sum(1 for _ in f) - 1
    if total_rows <= task_id:
        return 0
    return int(math.ceil((total_rows - task_id) / num_tasks))


def iter_assigned_rows(
    input_tsv: str,
    task_id: int,
    num_tasks: int,
) -> Iterator[Tuple[int, Dict[str, str]]]:
    # Data-parallel split:
    # task k processes rows where (row_idx % num_tasks == k).
    # This guarantees no overlap across array workers.
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def normalize_one_level(
    level_obj: Dict[str, Any],
    level_name: str,
    strict_pair_check: bool,
) -> Dict[str, List[str]]:
    if not isinstance(level_obj, dict):
        if strict_pair_check:
            raise ValueError(f"{level_name} must be a dict.")
        return {"English": [], "Chinese": []}

    english = level_obj.get("English", [])
    chinese = level_obj.get("Chinese", [])

    # Fast path for generation stage: do not truncate/skip; keep raw model output shape.
    if not strict_pair_check:
        if not isinstance(english, list):
            english = [english]
        if not isinstance(chinese, list):
            chinese = [chinese]
        return {
            "English": [str(x) for x in english],
            "Chinese": [str(x) for x in chinese],
        }

    if not isinstance(english, list) or not isinstance(chinese, list):
        raise ValueError(f"{level_name}.English and {level_name}.Chinese must be lists.")
    if len(english) != len(chinese):
        raise ValueError(
            f"{level_name} English/Chinese length mismatch: {len(english)} vs {len(chinese)}"
        )

    eng_clean, zh_clean = [], []
    for i, (e, z) in enumerate(zip(english, chinese)):
        e_text = str(e).strip()
        z_text = str(z).strip()
        if not e_text:
            raise ValueError(f"{level_name}.English[{i}] is empty.")
        if not z_text:
            raise ValueError(f"{level_name}.Chinese[{i}] is empty.")
        eng_clean.append(e_text)
        zh_clean.append(z_text)

    return {"English": eng_clean, "Chinese": zh_clean}


def normalize_response(
    parsed: Dict[str, Any],
    strict_pair_check: bool,
) -> Dict[str, Dict[str, List[str]]]:
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")

    normalized = {}
    for level in LATENCY_LEVELS:
        if level not in parsed:
            if strict_pair_check:
                raise ValueError(f"Missing level: {level}")
            normalized[level] = {"English": [], "Chinese": []}
            continue
        normalized[level] = normalize_one_level(
            parsed[level], level, strict_pair_check=strict_pair_check
        )
    return normalized


def run_llm_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    sentences: List[str],
    strict_pair_check: bool,
) -> List[Dict[str, Dict[str, List[str]]]]:
    # One vLLM call handles multiple prompts.
    # With global batching, these prompts can come from different utt_ids.
    messages = [[{"role": "user", "content": build_prompt_refined_east(s)}] for s in sentences]
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    if len(outputs) != len(sentences):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(sentences)} inputs.")

    parsed_list = []
    for out in outputs:
        response = out.outputs[0].text.strip()
        parsed = json.loads(response)
        parsed_list.append(normalize_response(parsed, strict_pair_check=strict_pair_check))
    return parsed_list


def parse_output_obj(
    output_obj: Any,
    strict_pair_check: bool,
) -> Dict[str, Dict[str, List[str]]]:
    response = output_obj.outputs[0].text.strip()
    parsed = json.loads(response)
    return normalize_response(parsed, strict_pair_check=strict_pair_check)


def flush_global_sentence_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    sentence_tasks: List[Dict[str, Any]],
    pending_utts: Dict[str, Dict[str, Any]],
    strict_pair_check: bool,
    retry_single_on_fail: bool,
) -> Set[str]:
    """
    全局批处理的核心函数。

    关键设计：sentence_tasks 里的句子可能来自不同的 utt_id。
    例如一个 batch 可能是 [utt_A的第2句, utt_A的第3句, utt_B的第1句, utt_C的第5句]。
    这样做的好处是：无论单个 utterance 有多少句子，GPU 始终以 batch_size 满载运行。

    vLLM 内部处理流程：
    1. 收到 32 条 messages 后，tokenize 成 32 组 input_ids
    2. Prefill 阶段：32 组 input_ids 并行计算 attention（GPU 矩阵运算高度并行）
    3. Decode 阶段：32 个请求共享 GPU 算力，逐 token 生成
       - 因为 JSON schema 约束了输出格式，decode 长度可控
       - PagedAttention 按需分配 KV Cache 显存，避免预留浪费
    4. 返回 32 个 output 对象，顺序与输入一一对应

    Returns: 本次 flush 涉及到的 utt_id 集合（用于后续检查哪些 utt 已全部完成）
    """
    touched: Set[str] = set()

    # ---- 第一步：构造 batch 输入 ----
    # 从 sentence_tasks 提取纯文本，构造 chat messages
    # 每个 task 记录了它属于哪个 utt_id、是第几句（sentence_index）
    texts = [t["text"] for t in sentence_tasks]
    messages = [[{"role": "user", "content": build_prompt_refined_east(s)}] for s in texts]

    try:
        # ---- 第二步：一次性把整个 batch 送进 vLLM ----
        # 这是速度的关键！32 个 prompt 在 GPU 上并行推理，
        # 比逐条调用快得多（减少了 31 次 GPU kernel launch 开销 + 提高了计算密度）
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        if len(outputs) != len(sentence_tasks):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(sentence_tasks)} inputs."
            )

        # ---- 第三步：拆包结果，把每个输出归还给对应的 utt ----
        # outputs[i] 对应 sentence_tasks[i]，所以用 zip 配对
        for task, output_obj in zip(sentence_tasks, outputs):
            utt_id = task["utt_id"]
            sent_idx = task["sentence_index"]
            touched.add(utt_id)
            state = pending_utts[utt_id]  # 找到这个句子所属 utt 的状态容器
            text = task["text"]
            try:
                # 解析 LLM 输出的 JSON，存入 state["results"][sent_idx]
                # 这样同一个 utt 的不同句子结果会按 sentence_index 归位
                parsed = parse_output_obj(output_obj, strict_pair_check=strict_pair_check)
                state["results"][sent_idx] = parsed
            except Exception as parse_err:
                # 单条解析失败的容错：可选择逐句重试或直接记录错误
                if retry_single_on_fail:
                    try:
                        parsed = run_llm_batch(
                            llm,
                            sampling_params,
                            [text],
                            strict_pair_check=strict_pair_check,
                        )[0]
                        state["results"][sent_idx] = parsed
                    except Exception as single_err:
                        state["errors"].append(
                            {
                                "sentence_index": sent_idx,
                                "input": text,
                                "error": f"batch_parse_error={parse_err}; single_error={single_err}",
                            }
                        )
                else:
                    state["errors"].append(
                        {
                            "sentence_index": sent_idx,
                            "input": text,
                            "error": f"batch_parse_error={parse_err}",
                        }
                    )
            finally:
                # 无论成功失败，计数器 +1，用于判断该 utt 是否所有句子都处理完了
                state["done_sentences"] += 1

    except Exception as batch_err:
        # ---- 整个 batch 调用失败的兜底 ----
        # 如果 llm.chat 本身抛异常（比如 OOM），所有句子都需要处理
        for task in sentence_tasks:
            utt_id = task["utt_id"]
            sent_idx = task["sentence_index"]
            text = task["text"]
            touched.add(utt_id)
            state = pending_utts[utt_id]
            if retry_single_on_fail:
                # 逐句重试：batch_size=1，虽然慢但可能绕过 OOM
                try:
                    parsed = run_llm_batch(
                        llm,
                        sampling_params,
                        [text],
                        strict_pair_check=strict_pair_check,
                    )[0]
                    state["results"][sent_idx] = parsed
                except Exception as single_err:
                    state["errors"].append(
                        {
                            "sentence_index": sent_idx,
                            "input": text,
                            "error": f"batch_error={batch_err}; single_error={single_err}",
                        }
                    )
            else:
                state["errors"].append(
                    {
                        "sentence_index": sent_idx,
                        "input": text,
                        "error": f"batch_error={batch_err}",
                    }
                )
            state["done_sentences"] += 1

    return touched


def aggregate_sentence_results(
    sentence_results: List[Dict[str, Dict[str, List[str]]]]
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[List[int]]]]:
    """
    把句子级的 LLM 输出拼接成 utterance 级的完整结果。

    举例：一个 utterance 有 2 个句子，low_latency 下：
      句子0 的 LLM 输出: English=["The cat", "sat"], Chinese=["猫", "坐了"]
      句子1 的 LLM 输出: English=["on the", "mat."], Chinese=["在", "垫子上。"]

    聚合后：
      English = ["The cat", "sat", "on the", "mat."]   (直接拼接)
      Chinese = ["猫", "坐了", "在", "垫子上。"]
      spans   = [[0, 2], [2, 4]]  (句子0 占 index 0~1，句子1 占 index 2~3)

    spans 的作用：记录每个原始句子在拼接后数组中的位置范围，
    方便后续如果需要追溯某个 segment 来自哪个原始句子。
    """
    aggregated: Dict[str, Dict[str, List[str]]] = {}
    spans: Dict[str, List[List[int]]] = {}

    for level in LATENCY_LEVELS:
        english_all: List[str] = []
        chinese_all: List[str] = []
        level_spans: List[List[int]] = []
        cursor = 0  # 当前写入位置指针

        for sent in sentence_results:
            eng = sent[level]["English"]
            zh = sent[level]["Chinese"]
            # 记录这个句子的 segments 在最终数组中的 [start, end) 范围
            level_spans.append([cursor, cursor + len(eng)])
            english_all.extend(eng)  # 追加到总数组
            chinese_all.extend(zh)
            cursor += len(eng)

        aggregated[level] = {"English": english_all, "Chinese": chinese_all}
        spans[level] = level_spans

    return aggregated, spans


def finalize_utt_state(
    state: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Build final JSON object for one utt.
    Returns (is_success, out_obj).
    """
    utt_id = state["utt_id"]
    row_idx = state["row_index"]
    sentences = state["sentences"]

    if state["errors"] or len(state["results"]) != len(sentences):
        return False, {
            "utt_id": utt_id,
            "row_index": row_idx,
            "input_sentences": sentences,
            "error": "Some sentences failed during LLM inference.",
            "errors": state["errors"],
            "num_success_sentences": len(state["results"]),
            "num_total_sentences": len(sentences),
        }

    sentence_results = [state["results"][i] for i in range(len(sentences))]
    aggregated, sentence_spans = aggregate_sentence_results(sentence_results)

    out_obj: Dict[str, Any] = {
        "utt_id": utt_id,
        "row_index": row_idx,
        "input": " ".join(sentences).strip(),
        "input_sentences": sentences,
        "sentence_spans": sentence_spans,
        "source_manifest": args.input_tsv,
    }
    for level in LATENCY_LEVELS:
        out_obj[level] = aggregated[level]

    for k, v in state["meta_fields"].items():
        out_obj[k] = v

    if args.save_per_sentence:
        out_obj["per_sentence"] = []
        for sent_idx, sent_text in enumerate(sentences):
            item = {
                "sentence_index": sent_idx,
                "input": sent_text,
            }
            for level in LATENCY_LEVELS:
                item[level] = sentence_results[sent_idx][level]
            out_obj["per_sentence"].append(item)

    return True, out_obj


def main() -> None:
    args = parse_args()
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise ValueError(f"Invalid task split: task-id={args.task_id}, num-tasks={args.num_tasks}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    os.makedirs(args.output_root, exist_ok=True)
    setup_env()

    print(f"[Task {args.task_id}] Loading model TP={args.tp} from {args.model_path}")
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
        guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
    )

    total_assigned = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
    if args.max_rows is not None:
        total_assigned = min(total_assigned, args.max_rows)

    print(
        f"[Task {args.task_id}] Start processing: input={args.input_tsv}, "
        f"assigned_rows={total_assigned}, output={args.output_root}"
    )

    written = 0
    skipped_existing = 0
    failed = 0
    processed_rows = 0

    pending_utts: Dict[str, Dict[str, Any]] = {}
    sentence_queue: Deque[Dict[str, Any]] = deque()

    pbar = tqdm(total=total_assigned, desc=f"task_{args.task_id}")

    for row_idx, row in iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks):
        if args.max_rows is not None and processed_rows >= args.max_rows:
            break

        processed_rows += 1

        utt_id = str(row.get(args.id_column, "")).strip()
        if not utt_id:
            utt_id = f"row_{row_idx:09d}"

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
                        "row_index": row_idx,
                        "error": f"Empty/invalid {args.src_text_full_column}",
                        "raw_value": row.get(args.src_text_full_column),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            pbar.update(1)
            continue

        # Register one utt state, then push all its sentences into global queue.
        meta_fields = {}
        for k in ["audio", "n_frames", "speaker", "src_lang", "tgt_lang", "asr", "src_text"]:
            if k in row:
                meta_fields[k] = row[k]
        pending_utts[utt_id] = {
            "utt_id": utt_id,
            "row_index": row_idx,
            "sentences": sentences,
            "out_path": out_path,
            "meta_fields": meta_fields,
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

        # Global batching: mix sentences from different utt_ids into one llm.chat call.
        while len(sentence_queue) >= args.batch_size:
            batch_tasks = [sentence_queue.popleft() for _ in range(args.batch_size)]
            touched_utts = flush_global_sentence_batch(
                llm=llm,
                sampling_params=sampling_params,
                sentence_tasks=batch_tasks,
                pending_utts=pending_utts,
                strict_pair_check=args.strict_pair_check,
                retry_single_on_fail=args.retry_single_on_fail,
            )

            for touched_utt in touched_utts:
                state = pending_utts.get(touched_utt)
                if state is None:
                    continue
                if state["done_sentences"] != len(state["sentences"]):
                    continue
                ok, out_obj = finalize_utt_state(state, args)
                with open(state["out_path"], "w", encoding="utf-8") as f:
                    json.dump(out_obj, f, ensure_ascii=False, indent=2)
                if ok:
                    written += 1
                else:
                    failed += 1
                del pending_utts[touched_utt]

        pbar.update(1)

    # Flush remaining sentence tasks.
    if sentence_queue:
        batch_tasks = list(sentence_queue)
        sentence_queue.clear()
        touched_utts = flush_global_sentence_batch(
            llm=llm,
            sampling_params=sampling_params,
            sentence_tasks=batch_tasks,
            pending_utts=pending_utts,
            strict_pair_check=args.strict_pair_check,
            retry_single_on_fail=args.retry_single_on_fail,
        )
        for touched_utt in touched_utts:
            state = pending_utts.get(touched_utt)
            if state is None:
                continue
            if state["done_sentences"] != len(state["sentences"]):
                continue
            ok, out_obj = finalize_utt_state(state, args)
            with open(state["out_path"], "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            if ok:
                written += 1
            else:
                failed += 1
            del pending_utts[touched_utt]

    # Safety flush: finalize any remaining unfinished state as failure.
    for utt_id in list(pending_utts.keys()):
        state = pending_utts[utt_id]
        ok, out_obj = finalize_utt_state(state, args)
        with open(state["out_path"], "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        if ok:
            written += 1
        else:
            failed += 1
        del pending_utts[utt_id]

    pbar.close()
    print(
        f"[Task {args.task_id}] Done. written={written}, "
        f"skipped_existing={skipped_existing}, failed={failed}"
    )


if __name__ == "__main__":
    main()
