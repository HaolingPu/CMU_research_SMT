#!/usr/bin/env python3
"""Local Agreement (LA) rule-based simultaneous MT baseline.

Reference: TAF paper (arXiv 2410.22499), Section 3 baselines.

LA-N with prefix constraint: at each step, generate a full translation
hypothesis (prefix-constrained to committed text) using an offline MT model.
Commit the LCP of the last N hypotheses. At the final chunk, force-complete.

Usage:
  python local_agreement.py \
    --mt-api-base http://localhost:8100 \
    --mt-api-model qwen3-instruct \
    --mt-tokenizer-path /path/to/tokenizer \
    --la-n 2 --segment-size 1 --target-lang Chinese \
    --test-one --overwrite \
    --output-jsonl output/la_n2_seg1.jsonl
"""
import argparse
import ast
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_TSV_PATH = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/"
    "train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
)


# ---------------------------------------------------------------------------
# Environment & CLI
# ---------------------------------------------------------------------------

def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LA-N simultaneous MT baseline.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # MT model
    p.add_argument("--mt-tokenizer-path", required=True)
    p.add_argument("--mt-api-base", required=True)
    p.add_argument("--mt-api-model", default=os.environ.get("MT_API_MODEL", "qwen3-instruct"))
    p.add_argument("--mt-api-timeout", type=float, default=120.0)
    # LA parameters
    p.add_argument("--la-n", type=int, default=2)
    p.add_argument("--segment-size", type=int, default=1)
    p.add_argument("--lcp-mode", choices=["char", "word"], default="char")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--target-lang", default="Chinese")
    # Output / row selection
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def parse_trajectory(raw: str) -> List[str]:
    return ast.literal_eval(raw)


def join_source_chunks(chunks: List[str]) -> str:
    text = ""
    for raw_piece in chunks:
        piece = str(raw_piece or "")
        if not piece:
            continue
        if not text:
            text = piece
            continue
        if text[-1].isspace() or piece[0].isspace():
            text += piece
        elif piece[0] in ",.!?;:)]}\"'":
            text += piece
        elif text[-1] in "([{\"'":
            text += piece
        else:
            text += " " + piece
    return text.strip()


def build_source_observed(chunks: List[str], t: int) -> str:
    return join_source_chunks(chunks[: t + 1])


def get_full_source_text(row: Dict[str, Any]) -> str:
    raw = row.get("src_text")
    if raw is None or pd.isna(raw):
        raise ValueError("src_text missing")
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("src_text is empty")
    return text


def clean_model_text(text: str) -> str:
    """清洗模型原始输出：去掉 <think> 思考块和特殊 token，只留翻译文本。"""
    text = str(text or "")
    # 去掉完整的 <think>...</think> 块（模型思考内容）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 去掉被截断的 <think>（没有闭合 </think>，说明 max_tokens 不够）
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def normalize_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def verify_api(api_base: str, timeout: float) -> List[str]:
    req = urllib.request.Request(
        f"{normalize_api_base(api_base)}/models",
        headers={"Authorization": "Bearer dummy"}, method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]


def load_tokenizer(path: str) -> Any:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Translation (MT model calls)
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: Any, source: str, target_lang: str,
                  committed_text: str = "") -> str:
    """构建翻译 prompt。

    有 committed_text 时做 prefix-constrained 生成：
      prompt 末尾拼上已提交的翻译，模型只能从这个前缀续写，
      保证新 hypothesis 不会与已提交内容矛盾。
    无 committed_text 时做自由生成：模型输出完整翻译。
    """
    has_committed = bool(str(committed_text or "").strip())
    if has_committed:
        # 有已提交翻译 → 告诉模型从这个前缀续写
        content = (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. Continue from that prefix "
            "and complete the translation. Output only the continuation."
        )
    else:
        # 无已提交翻译 → 自由生成完整翻译
        content = (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only."
        )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=False, tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    # prefix-constrained: 把已提交翻译拼进 assistant 前缀，强制模型续写
    if has_committed:
        prompt += committed_text
    return prompt


def translate_source_prefix(
    tokenizer: Any, source_observed: str, committed_text: str,
    api_base: str, api_model: str, api_timeout: float,
    max_new_tokens: int, target_lang: str,
) -> str:
    """对当前 source prefix 生成完整翻译 hypothesis。

    有 committed_text 时，模型被强制从已提交翻译续写（prefix-constrained），
    返回值 = committed_text + continuation（完整 hypothesis）。
    无 committed_text 时，返回值 = 模型自由生成的完整翻译。
    """
    has_committed = bool(str(committed_text or "").strip())
    prompt = _build_prompt(tokenizer, source_observed, target_lang, committed_text)
    data = _http_json(
        f"{normalize_api_base(api_base)}/completions",
        payload={
            "model": api_model, "prompt": prompt,
            "max_tokens": max_new_tokens, "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        },
        timeout=api_timeout,
    )
    choices = data.get("choices", [])
    if not choices:
        return committed_text if has_committed else ""
    continuation = clean_model_text(str(choices[0].get("text", "")))
    # 有 committed → 拼回完整 hypothesis; 无 committed → continuation 就是完整翻译
    return (committed_text + continuation) if has_committed else continuation


def force_complete_translation(
    tokenizer: Any, full_source: str, committed_text: str,
    api_base: str, api_model: str, api_timeout: float,
    max_new_tokens: int, target_lang: str,
) -> str:
    """最后一步：用完整 source 生成 committed_text 之后的剩余翻译。"""
    prompt = _build_prompt(tokenizer, full_source, target_lang, committed_text)
    data = _http_json(
        f"{normalize_api_base(api_base)}/completions",
        payload={
            "model": api_model, "prompt": prompt,
            "max_tokens": max_new_tokens, "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        },
        timeout=api_timeout,
    )
    choices = data.get("choices", [])
    if not choices:
        return ""
    return clean_model_text(str(choices[0].get("text", "")))


# ---------------------------------------------------------------------------
# LCP
# ---------------------------------------------------------------------------

def longest_common_prefix_chars(hypotheses: List[str]) -> str:
    """字符级 LCP：逐字符对齐所有 hypothesis，遇到第一个不一致的字符就停。
    适用于中文/日文（目标语言无空格分词）。
    """
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]
    prefix: List[str] = []
    for chars in zip(*hypotheses):  # zip 自动截断到最短 hypothesis 的长度
        if len(set(chars)) == 1:    # 所有 hypothesis 在这个位置的字符相同
            prefix.append(chars[0])
        else:
            break  # 第一个分歧点 → 停止
    return "".join(prefix)


def longest_common_prefix_words(hypotheses: List[str]) -> str:
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]
    token_lists = [h.split() for h in hypotheses]
    prefix: List[str] = []
    for tokens in zip(*token_lists):
        if len(set(tokens)) == 1:
            prefix.append(tokens[0])
        else:
            break
    return " ".join(prefix)


def compute_lcp(hypotheses: List[str], mode: str = "char") -> str:
    if mode == "word":
        return longest_common_prefix_words(hypotheses)
    return longest_common_prefix_chars(hypotheses)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_reference_text(row: Dict[str, Any], target_lang: str) -> Optional[str]:
    lang_suffix_map = {"Japanese": "ja", "German": "de", "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    keys: List[str] = []
    if lang_suffix:
        keys.extend([f"target_full_{lang_suffix}", f"tgt_text_full_{lang_suffix}",
                      f"llm_reference_text_{lang_suffix}"])
    keys.extend(["llm_reference_text", "tgt_text_full", "tgt_text",
                  "target_text", "translation", "ref_text", "reference"])
    for key in keys:
        raw = row.get(key)
        if raw is not None and not pd.isna(raw):
            text = str(raw).strip()
            if text and text.lower() != "nan":
                return text
    return None


def compute_laal(source_chunks: List[str], target_deltas: List[str],
                 actions: List[str], reference: str) -> float:
    timeline: List[int] = []
    source_read = 0
    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        source_read += len(str(chunk).strip().split()) if str(chunk).strip() else 0
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)
    y_len = len("".join(d for d in target_deltas if d))
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())
    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")
    denom = max(y_len, yref_len)
    if denom <= 0 or not timeline:
        return float("nan")
    total = sum(
        (timeline[i - 1] if i <= len(timeline) else x_len) - (i - 1) * x_len / denom
        for i in range(1, denom + 1)
    )
    return total / denom


def compute_bleu_char(hypothesis: str, reference: str,
                      max_order: int = 4, smooth: bool = True) -> float:
    hyp = [c for c in str(hypothesis) if not c.isspace()]
    ref = [c for c in str(reference) if not c.isspace()]
    hyp_len, ref_len = len(hyp), len(ref)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    eff_order = min(max_order, hyp_len, ref_len)
    if eff_order <= 0:
        return float("nan")
    precisions: List[float] = []
    for n in range(1, eff_order + 1):
        hyp_ngrams = Counter(tuple(hyp[i : i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(ref_len - n + 1))
        total = sum(hyp_ngrams.values())
        if total <= 0:
            return float("nan")
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items())
        if smooth:
            precisions.append((clipped + 1.0) / (total + 1.0))
        else:
            if clipped == 0:
                return 0.0
            precisions.append(clipped / total)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / eff_order) * 100.0


# ---------------------------------------------------------------------------
# Core: run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(row: Dict[str, Any], args: argparse.Namespace,
                      mt_tokenizer: Any) -> Dict[str, Any]:
    """LA-N 核心循环：逐步读入 source chunk，生成 hypothesis，用 LCP 决定提交。

    流程示意（LA-2, segment_size=1）：
      step 1: src="I think"       → hyp0="我觉得"
      step 2: src="I think we"    → hyp1="我觉得我们"
              window=[hyp0, hyp1]  → LCP="我觉得"  → commit delta="我觉得"
      step 3: src="I think we should" → hyp2="我觉得我们应该"（prefix-constrained: 必须以"我觉得"开头）
              window=[hyp1, hyp2]  → LCP="我觉得我们" → commit delta="我们"
      ...
      最后一步: 用完整 source 做 force_complete，把剩余翻译全部提交
    """
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = parse_trajectory(row["src_trajectory"])
    full_source_text = get_full_source_text(row)
    n_chunks = len(chunks)

    committed_text = ""                              # 已提交的翻译文本
    target_deltas: List[str] = [""] * n_chunks       # 每个 chunk 对应的翻译增量
    actions: List[str] = ["READ"] * n_chunks         # 每个 chunk 的动作：READ 或 WRITE
    recent_hypotheses: List[str] = []                # 滑动窗口：最近 N 条 hypothesis

    step_start = 0
    while step_start < n_chunks:
        step_end = min(step_start + args.segment_size, n_chunks)  # 一次消耗 segment_size 个 chunk
        is_last_step = (step_end == n_chunks)
        last_chunk_idx = step_end - 1
        source_observed = build_source_observed(chunks, last_chunk_idx)

        if is_last_step:
            # ── 最后一步：用完整 source 做 force-complete，提交所有剩余翻译 ──
            delta = force_complete_translation(
                mt_tokenizer, full_source_text, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            if delta:
                committed_text += delta
            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = "WRITE" if delta else "READ"

        elif not source_observed.strip():
            # 空 source prefix（第一个 chunk 为空）→ 跳过
            step_start = step_end
            continue

        else:
            # ── 正常步骤：生成 hypothesis → 滑动窗口 LCP → 决定提交 ──

            # 1) 调 MT model 生成当前 source prefix 的翻译 hypothesis
            #    有 committed_text 时，hypothesis 被强制以 committed_text 开头
            hyp = translate_source_prefix(
                mt_tokenizer, source_observed, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )

            # 2) 维护滑动窗口：保留最近 N 条 hypothesis
            recent_hypotheses.append(hyp)
            if len(recent_hypotheses) > args.la_n:
                recent_hypotheses.pop(0)

            # 3) 计算窗口内所有 hypothesis 的最长公共前缀 (LCP)
            lcp = compute_lcp(recent_hypotheses, mode=args.lcp_mode)

            # 4) 如果 LCP 比已提交的更长 → 提交新增部分 (delta)
            if lcp.startswith(committed_text) and len(lcp) > len(committed_text):
                delta = lcp[len(committed_text):]
                committed_text = lcp
            else:
                delta = ""

            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = "WRITE" if delta else "READ"

        step_start = step_end

    # Metrics
    reference_text = _extract_reference_text(row, args.target_lang)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    try:
        if reference_text:
            laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
            bleu_char_value = compute_bleu_char(committed_text, reference_text)
    except Exception:
        pass

    return {
        "utt_id": utt_id,
        "src_trajectory": chunks,
        "source_full_text": full_source_text,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "reference_text": reference_text or "",
        "decoder_impl": {
            "method": "local_agreement",
            "la_n": args.la_n,
            "segment_size": args.segment_size,
            "lcp_mode": args.lcp_mode,
        },
        "metrics": {"laal_text": laal_value, "bleu_char": bleu_char_value},
    }


# ---------------------------------------------------------------------------
# Row selection & main
# ---------------------------------------------------------------------------

def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_column].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected
    if args.test_one:
        return df.iloc[[args.row_idx]]
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max(1, int(args.max_rows)))
    return df.iloc[start:end]


def main() -> None:
    setup_env()
    args = parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)
    print(f"Processing {len(rows)} row(s)  LA-{args.la_n}  segment_size={args.segment_size}")

    models = verify_api(args.mt_api_base, args.mt_api_timeout)
    if args.mt_api_model not in models:
        raise RuntimeError(f"model '{args.mt_api_model}' not found; available={models}")
    print(f"[MT] model={args.mt_api_model}  api={normalize_api_base(args.mt_api_base)}")

    mt_tokenizer = load_tokenizer(args.mt_tokenizer_path)

    out_fh = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        out_fh = open(args.output_jsonl, "w" if args.overwrite else "a", encoding="utf-8")

    for _, row in rows.iterrows():
        result = run_one_utterance(row.to_dict(), args, mt_tokenizer)
        m = result["metrics"]
        print(f"  {result['utt_id']}  bleu={m['bleu_char']:.2f}  laal={m['laal_text']:.2f}")
        if out_fh:
            out_fh.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            out_fh.flush()

    if out_fh:
        out_fh.close()
    print("Done.")


if __name__ == "__main__":
    main()


# # 先跑一条测试
# python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement/local_agreement.py \
#   --mt-api-base http://localhost:8100 \
#   --mt-api-model qwen3-instruct \
#   --mt-tokenizer-path /data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
#   --la-n 2 --segment-size 1 --target-lang Chinese \
#   --test-one --overwrite \
#   --output-jsonl /data/user_data/haolingp/data_synthesis/codes/gigaspeech/rule-based-SMT/local_agreement/output/la_n2_seg1_test.jsonl
