#!/usr/bin/env python3
"""Consensus decoding with future sampling via vLLM serve API.

Two base models (e.g. gemma4-E2B + Qwen3-4B-Base) each generate future
continuations of the observed source prefix.  An instruct model
(Qwen3-30B-A3B-Instruct) provides next-token distributions for each
hypothesised full source, and the consensus intersection selects the
committed token.  Candidate sets are built via either top-k or min-p.
"""
import argparse
import ast
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_TSV_PATH = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
DEFAULT_INSTRUCT_API_BASE = os.environ.get("INSTRUCT_API_BASE", "")
DEFAULT_INSTRUCT_API_MODEL = os.environ.get("INSTRUCT_API_MODEL", "qwen3-instruct")
TOP_K = 6
MIN_P = 0.0


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus decoding with future sampling via vLLM serve.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # primary base model (future sampling)
    p.add_argument("--base-model-path", default="")
    p.add_argument("--base-api-base", required=True)
    p.add_argument("--base-api-model", required=True)
    p.add_argument("--base-api-timeout", type=float, default=120.0)
    # secondary base model (future sampling)
    p.add_argument("--secondary-base-model-path", default="")
    p.add_argument("--secondary-base-api-base", default="")
    p.add_argument("--secondary-base-api-model", default="")
    p.add_argument("--secondary-base-api-timeout", type=float, default=120.0)
    # instruct model (next-token distribution)
    p.add_argument("--instruct-tokenizer-path", required=True)
    p.add_argument("--instruct-api-base", required=True)
    p.add_argument("--instruct-api-model", default=DEFAULT_INSTRUCT_API_MODEL)
    p.add_argument("--instruct-api-timeout", type=float, default=120.0)
    # sampling / decoding
    p.add_argument("--num-futures", type=int, default=20)
    p.add_argument("--secondary-num-futures", type=int, default=10)
    p.add_argument("--future-tokens", type=int, default=20)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--max-consensus-steps", type=int, default=32)
    p.add_argument("--candidate-top-k", type=int, default=TOP_K)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.0,
                   help="Nucleus (top-p) candidate selection: keep smallest set with cumulative prob >= top-p.")
    # target language
    p.add_argument("--target-lang", default="Chinese",
                   help="Target language name for prompts (e.g. Chinese, Japanese, German)")
    # output
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-dir", default=None)
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--num-concurrent-cases", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip rows whose per-utterance output JSON already exists.")
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
        raise ValueError("src_text is missing from input row")
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("src_text is empty in input row")
    return text


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def append_text_continuation(prefix: str, continuation: str) -> str:
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if prefix[-1].isspace() or continuation[0].isspace():
        return prefix + continuation
    if continuation[0] in ",.!?;:)]}\"'":
        return prefix + continuation
    return prefix + " " + continuation


# ---------------------------------------------------------------------------
# Verbose logging (lightweight)
# ---------------------------------------------------------------------------

def _vlog(f: Optional[Any], msg: str) -> None:
    if f is None:
        return
    line = str(msg) if msg.endswith("\n") else str(msg) + "\n"
    f.write(line)
    f.flush()


class _TeeWriter:
    """Write to both a file and stdout (used only for --test-one)."""
    def __init__(self, fobj: Any):
        self._f = fobj
    def write(self, msg: str) -> None:
        self._f.write(msg); sys.stdout.write(msg)
    def flush(self) -> None:
        self._f.flush(); sys.stdout.flush()
    def close(self) -> None:
        self._f.close()


def write_pretty_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    text = clean_model_text(raw_text)
    if text.startswith(observed_source):
        text = text[len(observed_source):].lstrip()
    text = text.splitlines()[0].strip() if text else ""
    return text


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned = ["translate", "translation", "grammar analysis", "analyze", "analysis",
              "这句话", "翻译", "语法", "句子结构"]
    return not any(b in lowered for b in banned)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_future_sampling_prompt(observed_source: str) -> str:
    return observed_source


def build_translation_probe_prompt(tokenizer: Any, full_source: str, target_prefix: str, target_lang: str = "Chinese") -> str:
    if not str(target_prefix or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
            "and output only the next continuation token(s)."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. You must continue from that "
            "exact prefix and produce only the continuation."
        )}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(target_prefix or "").strip():
        prompt += target_prefix
    return prompt


def build_translation_probe_prompt_prefix_token_ids(
    tokenizer: Any,
    full_source: str,
    has_target_prefix: bool,
    target_lang: str = "Chinese",
) -> List[int]:
    if not has_target_prefix:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
            "and output only the next continuation token(s)."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. You must continue from that "
            "exact prefix and produce only the continuation."
        )}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    return list(prompt_ids) + list(assistant_prefix_ids) #返回翻译指令+assistant前缀的token ID列表，后面会拼上已提交翻译的token IDs


def build_final_completion_prompt(tokenizer: Any, full_source: str, committed_text: str, target_lang: str = "Chinese") -> str:
    if not str(committed_text or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. Continue from that prefix "
            "and output only the remaining continuation."
        )}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(committed_text or "").strip():
        prompt += committed_text
    return prompt


# ---------------------------------------------------------------------------
# Model / API helpers
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def verify_api(api_base: str, timeout: float) -> List[str]:
    data = _http_get_json(f"{normalize_api_base(api_base)}/models", timeout=timeout)
    return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _disallowed_generation_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    for frag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if frag in token_text:
            return "special_token_text"
    if re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    # NOTE: Do NOT filter replacement_char (U+FFFD) here. Byte-level BPE tokens
    # for rare characters (e.g. Japanese kanji "繰") decode to U+FFFD individually
    # but combine into valid characters when decoded together in the pending buffer.
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width_or_variation_selector"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control_or_surrogate"
    return None


def filter_distribution_token_ids(
    tokenizer: Any,
    id_distribution: Dict[int, float],
    dist_debug: Dict[str, Any],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    filtered: Dict[int, float] = {}
    removed: List[Dict[str, Any]] = []
    for tok_id, prob in id_distribution.items():
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None:
            removed.append({"token_id": int(tok_id), "token_text": _single_token_text(tokenizer, tok_id),
                            "prob": round(float(prob), 6), "reason": reason})
            continue
        filtered[int(tok_id)] = float(prob)

    out_debug = dict(dist_debug)
    out_debug["filtered_disallowed_tokens"] = removed
    if filtered:
        out_debug["topk_token_ids"] = list(filtered.keys())
        out_debug["topk_token_texts"] = [_single_token_text(tokenizer, t) for t in filtered]
        out_debug["topk_true_probs"] = [round(float(filtered[t]), 6) for t in filtered]
    elif removed and out_debug.get("reason") == "ok":
        out_debug["reason"] = "all_top_tokens_filtered_as_disallowed"
    return filtered, out_debug


# ---------------------------------------------------------------------------
# Future sampling (API only)
# ---------------------------------------------------------------------------

def sample_source_futures_api(
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    sample_temperature: float,
    api_base: str,
    api_model: str,
    api_timeout: float,
) -> List[str]:
    if not observed_source.strip():
        return []
    payload = {
        "model": api_model,
        "prompt": build_future_sampling_prompt(observed_source),
        "max_tokens": future_tokens,
        "temperature": sample_temperature,
        "top_p": 0.95,
        "n": num_futures,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call base model，用observed source作为prompt，采样n条可能的未来源语言续写
    futures: List[str] = []
    seen: set = set()
    for choice in data.get("choices", []): #每个choice是一条future续写，清洗去重后返回
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = clean_future_text(observed_source, raw)
        if cleaned and is_valid_future_text(cleaned) and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            futures.append(cleaned)
    return futures


def sample_source_futures_multi(
    base_specs: List[Dict[str, Any]],
    observed_source: str,
    future_tokens: int,
    sample_temperature: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    merged: List[str] = []
    merged_info: List[Dict[str, Any]] = []
    seen: set = set()
    for spec in base_specs:
        requested = int(spec.get("num_futures", 0) or 0)
        if requested <= 0:
            continue
        futures = sample_source_futures_api(
            observed_source=observed_source,
            num_futures=requested,
            future_tokens=future_tokens,
            sample_temperature=sample_temperature,
            api_base=spec["api_base"],
            api_model=spec["api_model"],
            api_timeout=spec["api_timeout"],
        )
        for future in futures:
            key = future.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(future)
            merged_info.append({"source": spec["name"], "path": spec["path"], "future": future})
    return merged, merged_info


# ---------------------------------------------------------------------------
# Next-token distribution (API only, batch)
# ---------------------------------------------------------------------------

def _parse_token_id_string(raw: str) -> Optional[int]:
    match = re.fullmatch(r"token_id:(\d+)", str(raw or "").strip())
    return int(match.group(1)) if match else None


def _parse_completion_top_logprobs(
    # The raw "top_logprobs" from the API is a list of dicts, 
    # one per generated token, where each dict maps
    # token_id strings like "token_id:1234" to logprobs. 
    #We only care about the first step (next token).
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not top_logprobs:
        return {}, {"reason": "missing_top_logprobs"}
    step = top_logprobs[0]
    if not step:
        return {}, {"reason": "empty_top_logprobs"}
    id_distribution: Dict[int, float] = {}
    unknown_tokens: List[str] = []
    for raw_token, logprob in step.items():
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is None:
            unknown_tokens.append(str(raw_token))
            continue
        id_distribution[tok_id] = float(math.exp(float(logprob))) #logprob是对数概率，exp转回真实概率
    token_ids = list(id_distribution.keys())
    # {
    #     123: 0.82,
    #     456: 0.21,
    #     789: 0.07
    # }
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [_single_token_text(tokenizer, t) for t in token_ids],
        "topk_true_probs": [round(float(id_distribution[t]), 6) for t in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


def batch_get_next_token_distributions(
    tokenizer: Any,
    full_sources: List[str],
    target_prefix_token_ids: List[int],
    top_k: int = TOP_K,
    min_p: float = 0.0,
    top_p: float = 0.0,
    api_base: str = "",
    api_model: str = "",
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
) -> List[Tuple[Dict[int, float], Dict[str, Any]]]:
# Step 1: 用 chat template 把翻译指令编码成 token IDs
#         → [<|im_start|>, user, \n, [TASK], Translate..., <|im_end|>]
        
# Step 2: 拼上 assistant 前缀的 token IDs
#         → 上面 + encode("<|im_start|>assistant\n")
        
# Step 3: 拼上已提交翻译的 token IDs
#         → 上面 + [32108, 45621]
    prompts = [
        build_translation_probe_prompt_prefix_token_ids(
            tokenizer,
            src,
            has_target_prefix=bool(target_prefix_token_ids),
            target_lang=target_lang,
        ) + list(target_prefix_token_ids) # Step 3
        for src in full_sources
    ]
    # [user task tokens] + [assistant prefix tokens] + [already committed target prefix tokens]
    if not prompts:
        return []
    logprobs_n = max(top_k, 100) if (min_p > 0 or top_p > 0) else top_k
    payload = {
        "model": api_model,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": logprobs_n,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，传token ID列表，只生成1个token，拿top-k logprobs分布
    choices = data.get("choices", []) #vllm做batch处理，每个choice对应一个future的next-token分布
    results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
    for i in range(len(prompts)):
        if i >= len(choices):
            results.append(({}, {"reason": "missing_choice", "raw_response": data}))
            continue
        choice = choices[i] 
        # choice = {
        #     "text": "而",
        #     "logprobs": {
        #         "tokens": ["token_id:123"],
        #         "top_logprobs": [
        #             {
        #                 "token_id:123": -0.1,
        #                 "token_id:456": -1.2,
        #                 "token_id:789": -2.0,
        #                 "token_id:999": -3.1,
        #                 "token_id:555": -4.0,
        #             }
        #         ]
        #     }
        # }
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {} #每个choice里有一个"logprobs"字段，里面的"top_logprobs"是一个列表，每个元素是一个dict，表示对应生成token的top-k logprobs分布；我们只关心第一个生成token的分布，所以取top_logprobs[0]来解析出token ID分布

        dist, dist_debug = _parse_completion_top_logprobs(logprobs.get("top_logprobs"), tokenizer=tokenizer) #解析出token ID分布，并记录debug信息
        dist_debug["api_backend"] = "vllm_completion"
        if min_p > 0:
            dist_debug["candidate_policy"] = "min_p"
            dist_debug["min_p"] = min_p
        else:
            dist_debug["candidate_policy"] = "top_k"
            dist_debug["candidate_top_k"] = top_k
        results.append(filter_distribution_token_ids(tokenizer, dist, dist_debug)) #过滤掉不合规的token ID，并在debug信息里记录被过滤掉的token ID和原因
    return results


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    # Return the top-k token IDs by probability, sorted in descending order of probability.
    return [tok_id for tok_id, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def minp_token_ids(dist: Dict[int, float], min_p: float) -> List[int]:
    """Return token IDs whose probability >= min_p (absolute threshold)."""
    if not dist:
        return []
    return [tok_id for tok_id, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True) if prob >= min_p]


def topp_token_ids(dist: Dict[int, float], top_p: float) -> List[int]:
    """Return the smallest set of token IDs whose cumulative probability >= top_p."""
    if not dist:
        return []
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    result: List[int] = []
    cumsum = 0.0
    for tok_id, prob in sorted_items:
        result.append(tok_id)
        cumsum += prob
        if cumsum >= top_p:
            break
    return result


def _select_candidates(dist: Dict[int, float], top_p: float = 0.0,
                       min_p: float = 0.0, top_k: int = TOP_K) -> List[int]:
    """Select candidate token IDs using the active policy (top-p > min-p > top-k)."""
    if top_p > 0:
        return topp_token_ids(dist, top_p)
    if min_p > 0:
        return minp_token_ids(dist, min_p)
    return topk_token_ids(dist, top_k)


def choose_consensus_token(
    distributions: List[Dict[int, float]],
    min_p: float = MIN_P,
    candidate_top_k: int = TOP_K,
    top_p: float = 0.0,
) -> Tuple[Optional[int], Dict[str, Any]]:
    if not distributions:
        return None, {"reason": "no_distributions"}
    candidate_lists = [_select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
                       for dist in distributions]
    intersection = set(candidate_lists[0]) #对所有future的候选集取交集，只保留大家都认可的token
    for clist in candidate_lists[1:]:
        intersection &= set(clist) #逐个求交集，交集为空说明没有共识
    if not intersection:
        return None, {"reason": "empty_intersection", "candidate_lists": candidate_lists}
    #就从交集里选平均概率最高的那个
    best_token = max(intersection, key=lambda tok: sum(d.get(tok, 0.0) for d in distributions) / len(distributions)) #交集里选平均概率最高的token作为共识输出
    return best_token, {
        "reason": "ok",
        "intersection": sorted(intersection),
        "avg_score": sum(d.get(best_token, 0.0) for d in distributions) / len(distributions),
        "candidate_lists": candidate_lists,
    }


# ---------------------------------------------------------------------------
# Token buffer management
# ---------------------------------------------------------------------------

def decode_token_ids_to_text(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def inspect_token_ids(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
    # Decode the entire token ID buffer to text for debugging and logging.  This is NOT used for any model input or state management, to avoid drift from re-encoding incomplete token sequences.
    decoded_text = decode_token_ids_to_text(tokenizer, token_ids)
    last_token_id = token_ids[-1] if token_ids else None
    last_token_text = _single_token_text(tokenizer, last_token_id) if last_token_id is not None else ""
    return {"decoded_text": decoded_text, "last_token_id": last_token_id, "last_token_text": last_token_text}


def has_suspicious_content(text: str) -> bool:
    if not text:
        return False
    # Check the ENTIRE decoded text for replacement characters, not just tail.
    # Byte-level BPE tokens may form broken sequences anywhere in the buffer
    # when consensus picks a byte token but fails to pick its pair.
    if "\ufffd" in text or "�" in text:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_pending_token_ids(
    tokenizer: Any, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]]]:
# Filter out any disallowed tokens from the pending token ID buffer, and log the removed tokens with reasons.  This ensures that we never commit or feed back into the model any token that could cause decoding issues or model confusion.
    kept: List[int] = []
    removed: List[Dict[str, Any]] = []
    for idx, tok_id in enumerate(pending_token_ids):
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is None:
            kept.append(tok_id)
        else:
            removed.append({"position": idx, "token_id": int(tok_id),
                            "token_text": _single_token_text(tokenizer, tok_id), "reason": reason})
    return kept, removed


def trim_pending_tokens_to_complete_boundary(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[int], Dict[str, Any]]:
    work, removed_disallowed = sanitize_pending_token_ids(tokenizer, pending_token_ids)
    if not work:
        return [], removed_disallowed, [], {
            "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
        }
    # Commit only an exact prefix of the original token-id buffer.
    # This preserves the true model state and avoids drift from re-encoding text
    # such as ["而", "且"] into a merged token like ["而且"].
    last_committable_idx = -1 #从前往后逐个decode，找到最后一个"解码后没有乱码"的位置
    for i in range(len(work)):
        partial = decode_token_ids_to_text(tokenizer, work[:i + 1])
        if not has_suspicious_content(partial): #没有U+FFFD等异常字符，说明到这里的token组合是完整的
            last_committable_idx = i
    if last_committable_idx >= 0:
        clean_ids = work[:last_committable_idx + 1]
        removed_tail = work[last_committable_idx + 1:]
        view = inspect_token_ids(tokenizer, clean_ids)
        view["full_text"] = committed_text + view["decoded_text"]
        return clean_ids, removed_disallowed, removed_tail, view
    return [], removed_disallowed, work, {
        "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
    }


def finalize_external_commit(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[str, str, List[int], Dict[str, Any]]:
    trimmed, removed_disallowed, removed_tail, view = trim_pending_tokens_to_complete_boundary(
        tokenizer, committed_text, pending_token_ids)
    commit_text = view["decoded_text"]
    return committed_text + commit_text, commit_text, trimmed, {
        "pending_before_trim": decode_token_ids_to_text(tokenizer, pending_token_ids),
        "pending_after_trim": commit_text,
        "removed_disallowed_tokens": removed_disallowed,
        "removed_tail_text": decode_token_ids_to_text(tokenizer, removed_tail),
    }


# ---------------------------------------------------------------------------
# Core consensus loop
# ---------------------------------------------------------------------------

def extend_pending_tokens(
    instruct_tokenizer: Any,
    source_observed: str,
    futures: List[str],
    committed_text: str,
    committed_token_ids: List[int],
    max_consensus_steps: int,
    candidate_top_k: int = TOP_K,
    instruct_api_base: str = "",
    instruct_api_model: str = "",
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
    top_p: float = 0.0,
    target_lang: str = "Chinese",
) -> Tuple[List[int], List[Dict[str, Any]]]:
    pending_token_ids: List[int] = []
    grow_logs: List[Dict[str, Any]] = []

    for step_idx in range(max_consensus_steps):
        target_prefix_token_ids = list(committed_token_ids) + list(pending_token_ids)
        full_sources = [append_text_continuation(source_observed, f) for f in futures]

        batch_results = batch_get_next_token_distributions(
            tokenizer=instruct_tokenizer,
            full_sources=full_sources,
            target_prefix_token_ids=target_prefix_token_ids,
            top_k=candidate_top_k,
            min_p=min_p,
            top_p=top_p,
            api_base=instruct_api_base,
            api_model=instruct_api_model,
            api_timeout=instruct_api_timeout,
            target_lang=target_lang,
        )

        distributions: List[Dict[int, float]] = []
        per_future: List[Dict[str, Any]] = []
        for i, (dist, dist_debug) in enumerate(batch_results):
            if not dist:
                grow_logs.append({"step": step_idx, "stop": "empty_distribution",
                                  "future": futures[i], "dist_debug": dist_debug})
                return pending_token_ids, grow_logs
            distributions.append(dist)
            candidate_ids = _select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
            per_future.append({
                "future": futures[i],
                "candidate_texts": [_single_token_text(instruct_tokenizer, t) for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        consensus_token_id, meta = choose_consensus_token(distributions, min_p=min_p, candidate_top_k=candidate_top_k, top_p=top_p)
        if consensus_token_id is None: #没有共识token，停止本轮生长
            grow_logs.append({"step": step_idx, "stop": "no_consensus_token",
                              "per_future": per_future, "meta": meta})
            break

        pending_token_ids.append(consensus_token_id) #共识成功，追加到pending buffer，继续下一步
        view = inspect_token_ids(instruct_tokenizer, pending_token_ids) #解码当前pending buffer，准备日志
        log_entry = {
            "step": step_idx,
            "accepted_token_id": consensus_token_id,
            "accepted_token_text": view["last_token_text"],
            "pending_text": view["decoded_text"],
            "llm_prefix": decode_token_ids_to_text(instruct_tokenizer, target_prefix_token_ids),
            "llm_prefix_token_ids": target_prefix_token_ids,
            "per_future": per_future,
            "meta": meta,
        }
        grow_logs.append(log_entry)

    return pending_token_ids, grow_logs


# ---------------------------------------------------------------------------
# Force-complete last chunk
# ---------------------------------------------------------------------------

def force_complete_translation(
    tokenizer: Any,
    full_source: str,
    committed_text: str,
    api_base: str,
    api_model: str,
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
) -> str:
    prompt = build_final_completion_prompt(tokenizer, full_source, committed_text, target_lang=target_lang)
    payload = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，带上已提交翻译前缀，让模型把剩余翻译补完
    choices = data.get("choices", [])
    if not choices:
        return ""
    return clean_model_text(str(choices[0].get("text", "")))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_reference_text_from_row(row: Dict[str, Any], target_lang: str = "Chinese") -> Optional[str]:
    lang_suffix_map = {"Japanese": "ja", "German": "de", "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    keys = []
    if lang_suffix:
        keys.extend([f"target_full_{lang_suffix}", f"tgt_text_full_{lang_suffix}", f"llm_reference_text_{lang_suffix}"])
    keys.extend(["llm_reference_text", "tgt_text_full", "tgt_text", "target_text", "translation", "ref_text", "reference"])
    for key in keys:
        raw = row.get(key)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


def compute_laal(
    source_chunks: List[str], target_deltas: List[str], actions: List[str], reference: str,
) -> float:
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
    total = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        total += d_i - (i - 1) * x_len / denom
    return total / denom


def compute_bleu_char(hypothesis: str, reference: str, max_order: int = 4, smooth: bool = True) -> float:
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
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(ref_len - n + 1))
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
# Run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = parse_trajectory(row["src_trajectory"])
    full_source_text = get_full_source_text(row)

    committed_text = ""
    committed_token_ids: List[int] = []
    target_deltas: List[str] = []
    actions: List[str] = []

    if args.top_p > 0:
        candidate_policy = f"top_p({args.top_p})"
    elif args.min_p > 0:
        candidate_policy = f"min_p({args.min_p})"
    else:
        candidate_policy = f"top_k({args.candidate_top_k})"

    # ── verbose header ──
    _vlog(verbose_log_file, "############################################################")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source_text}")
    _vlog(verbose_log_file, f"# Chunks: {len(chunks)}")
    _vlog(verbose_log_file, f"# num_futures={args.num_futures}, top_k={args.candidate_top_k}, min_p={args.min_p}")
    for si, bs in enumerate(base_specs):
        label = bs.get("name", f"spec_{si}")
        _vlog(verbose_log_file, f"# base_model[{label}]: api model={bs.get('api_model','')} base={bs.get('api_base','')} num_futures={bs.get('num_futures','')}")
    _vlog(verbose_log_file, f"# instruct_backend: vllm_completion")
    _vlog(verbose_log_file, "############################################################")

    for t in range(len(chunks)):
        source_observed = build_source_observed(chunks, t)
        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}")
        _vlog(verbose_log_file, f"source_observed: {source_observed!r}")
        _vlog(verbose_log_file, f"committed_before: {committed_text!r}")

        if t == len(chunks) - 1: #最后一个chunk，不再做共识，直接让instruct model把翻译补完
            final_delta = force_complete_translation(
                tokenizer=instruct_tokenizer,
                full_source=full_source_text,
                committed_text=committed_text,
                api_base=args.instruct_api_base,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang,
            )
            if final_delta:
                committed_text += final_delta
                target_deltas.append(final_delta)
                actions.append("WRITE")
            else:
                target_deltas.append("")
                actions.append("READ")
            _vlog(verbose_log_file, f"  [Final] delta={final_delta!r}")
            continue

        futures, future_infos = sample_source_futures_multi( #call base models，用当前observed source采样多条未来续写
            base_specs=base_specs,
            observed_source=source_observed,
            future_tokens=args.future_tokens,
            sample_temperature=args.sample_temperature,
        )

        # ── verbose: list futures ──
        _vlog(verbose_log_file, f"[Step 1-2] future_sampling total={len(futures)}")
        if verbose_log_file is not None:
            for fi, ftxt in enumerate(futures):
                info = future_infos[fi] if fi < len(future_infos) else {}
                label = info.get("source", "?")
                _vlog(verbose_log_file, f"  future[{fi}] ({label}): {ftxt!r}")

        if len(futures) < 2: #future太少无法做共识，等待更多源语言输入
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        pending_token_ids, grow_logs = extend_pending_tokens(
            instruct_tokenizer=instruct_tokenizer,
            source_observed=source_observed,
            futures=futures,
            committed_text=committed_text,
            committed_token_ids=committed_token_ids,
            max_consensus_steps=args.max_consensus_steps,
            candidate_top_k=args.candidate_top_k,
            instruct_api_base=args.instruct_api_base,
            instruct_api_model=args.instruct_api_model,
            instruct_api_timeout=args.instruct_api_timeout,
            min_p=args.min_p,
            top_p=args.top_p,
            target_lang=args.target_lang,
        )

        # ── verbose: consensus steps ──
        if verbose_log_file is not None and grow_logs:
            _vlog(verbose_log_file, f"[Step 4-5] consensus summary:")
            for gl in grow_logs:
                step = gl.get("step", "?")
                stop = gl.get("stop", "")
                meta = gl.get("meta", {})
                intersection = meta.get("intersection", [])
                if stop:
                    intersection_texts = [_single_token_text(instruct_tokenizer, tid) for tid in intersection] if intersection else []
                    _vlog(verbose_log_file, f"  step={step} stop={stop} intersection={intersection_texts} pending={decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)!r}")
                else:
                    accepted = gl.get("accepted_token_text", "?")
                    pending = gl.get("pending_text", "")
                    _vlog(verbose_log_file, f"  step={step} accepted={accepted!r} pending={pending!r}")
                per_future = gl.get("per_future", [])
                for pf in per_future:
                    texts = pf.get("candidate_texts", [])
                    probs = pf.get("candidate_probs", [])
                    pairs = ", ".join(f"{t!r}:{p:.3f}" for t, p in zip(texts, probs))
                    idx = per_future.index(pf)
                    _vlog(verbose_log_file, f"    future[{idx}] candidates={len(texts)}: [{pairs}]")

        new_committed, delta, committed_delta_token_ids, finalize_meta = finalize_external_commit( #修剪pending tokens，只提交解码后没有乱码的前缀部分
            tokenizer=instruct_tokenizer,
            committed_text=committed_text,
            pending_token_ids=pending_token_ids,
        )

        # ── verbose: finalize ──
        pending_text = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
        _vlog(verbose_log_file, f"[Step 6-7] pending_before_trim={pending_text!r}")
        _vlog(verbose_log_file, f"[Step 6-7] commit_after_trim={delta!r}")

        action = "WRITE" if delta else "READ"
        target_deltas.append(delta)
        actions.append(action)
        _vlog(verbose_log_file, f"-> {action} delta={delta!r}")
        _vlog(verbose_log_file, f"committed_after: {new_committed!r}")
        committed_text = new_committed
        committed_token_ids.extend(committed_delta_token_ids)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "src_trajectory": chunks,
        "source_full_text": full_source_text,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {"candidate_policy": candidate_policy, "backend": "vllm_completion"},
    }

    reference_text = _extract_reference_text_from_row(row, target_lang=args.target_lang)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    try:
        if not reference_text:
            raise ValueError("reference_text_unavailable")
        laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
        bleu_char_value = compute_bleu_char(committed_text, reference_text)
    except Exception:
        pass

    result["reference_text"] = reference_text or ""
    result["metrics"] = {"laal_text": laal_value, "bleu_char": bleu_char_value}
    _vlog(verbose_log_file, f"  prediction={committed_text!r} bleu={bleu_char_value:.2f} laal={laal_value:.2f}")
    return result


# ---------------------------------------------------------------------------
# Row selection
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_env()
    args = parse_args()

    primary_num_futures = args.num_futures - args.secondary_num_futures
    if primary_num_futures <= 0:
        raise ValueError("primary base model must keep at least 1 future")

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)

    # Build base model specs (API only)
    base_specs: List[Dict[str, Any]] = []
    for name, api_base, api_model, api_timeout, model_path, num_futures in [
        ("primary", args.base_api_base, args.base_api_model,
         args.base_api_timeout, args.base_model_path, primary_num_futures),
        ("secondary", args.secondary_base_api_base, args.secondary_base_api_model,
         args.secondary_base_api_timeout, args.secondary_base_model_path, args.secondary_num_futures),
    ]:
        if not api_base or num_futures <= 0:
            continue
        models = verify_api(api_base, api_timeout)
        if api_model not in models:
            raise RuntimeError(f"{name} api_model '{api_model}' not found; available={models}")
        base_specs.append({
            "name": name, "path": model_path, "num_futures": num_futures,
            "api_base": api_base, "api_model": api_model, "api_timeout": api_timeout,
        })
        print(f"[Base] {name}: model={api_model} api={normalize_api_base(api_base)} futures={num_futures}")

    # Verify instruct API
    models = verify_api(args.instruct_api_base, args.instruct_api_timeout)
    if args.instruct_api_model not in models:
        raise RuntimeError(f"instruct_api_model '{args.instruct_api_model}' not found; available={models}")
    instruct_tokenizer = load_tokenizer(args.instruct_tokenizer_path)
    print(f"[Instruct] model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")

    output_dir: Optional[str] = None
    if args.output_jsonl:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        os.makedirs(output_dir, exist_ok=True)
    if args.verbose and args.verbose_dir:
        os.makedirs(args.verbose_dir, exist_ok=True)

    def _process_one_row(row_idx: int, series_dict: Dict[str, Any]) -> Dict[str, Any]:
        row = series_dict
        utt_id = str(row.get(args.id_column, row.get("id", f"row_{row_idx}")))
        out_path = (
            os.path.join(output_dir, f"{sanitize_filename(utt_id)}.json")
            if output_dir is not None else None
        )

        if args.skip_existing and out_path is not None and os.path.exists(out_path):
            print(f"[SKIP existing] {out_path}")
            return {"utt_id": utt_id, "skipped_existing": True}

        verbose_log_file: Optional[Any] = None
        if args.verbose:
            if args.verbose_dir:
                vpath = os.path.join(args.verbose_dir, f"verbose_{sanitize_filename(utt_id)}.log")
                raw_file = open(vpath, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file) if args.test_one else raw_file
            elif args.test_one:
                verbose_log_file = sys.stdout

        try:
            result = run_one_utterance(
                row=row, args=args, base_specs=base_specs,
                instruct_tokenizer=instruct_tokenizer, verbose_log_file=verbose_log_file,
            )
        finally:
            if verbose_log_file is not None and args.verbose_dir:
                verbose_log_file.close()

        if output_dir is not None:
            write_pretty_json(out_path or os.path.join(output_dir, f"{sanitize_filename(result['utt_id'])}.json"), result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    num_concurrent = max(1, args.num_concurrent_cases)
    row_items = [(idx, s.to_dict()) for idx, (_, s) in enumerate(rows.iterrows())]

    if num_concurrent <= 1:
        for row_idx, row_dict in row_items:
            _process_one_row(row_idx, row_dict)
    else:
        print(f"[Concurrent] {len(row_items)} rows, {num_concurrent} workers")
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futs = {executor.submit(_process_one_row, ri, rd): ri for ri, rd in row_items}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[ERROR] Row {futs[fut]} raised: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
