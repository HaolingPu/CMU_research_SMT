#!/usr/bin/env python3
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency per runtime env
    LLM = None
    SamplingParams = None


DEFAULT_TSV_PATH = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv"
DEFAULT_BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/data/user_data/haolingp/models/Qwen3-4B-Base",
)
DEFAULT_INSTRUCT_MODEL_PATH = os.environ.get(
    "INSTRUCT_MODEL_PATH",
    "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
)
DEFAULT_INSTRUCT_API_BASE = os.environ.get("INSTRUCT_API_BASE", "")
DEFAULT_INSTRUCT_API_MODEL = os.environ.get("INSTRUCT_API_MODEL", "qwen3-instruct")
TOP_K = 6
MIN_P = 0.0  # 0 = disabled (use top-k); >0 = keep tokens with prob >= MIN_P


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus decoding prototype with future sampling.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    p.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    p.add_argument("--base-api-base", default="")
    p.add_argument("--base-api-model", default="")
    p.add_argument("--base-api-timeout", type=float, default=120.0)
    p.add_argument("--secondary-base-model-path", default="")
    p.add_argument("--secondary-base-api-base", default="")
    p.add_argument("--secondary-base-api-model", default="")
    p.add_argument("--secondary-base-api-timeout", type=float, default=120.0)
    p.add_argument("--instruct-model-path", default=DEFAULT_INSTRUCT_MODEL_PATH)
    p.add_argument("--instruct-tokenizer-path", default=DEFAULT_INSTRUCT_MODEL_PATH)
    p.add_argument("--instruct-api-base", default=DEFAULT_INSTRUCT_API_BASE)
    p.add_argument("--instruct-api-model", default=DEFAULT_INSTRUCT_API_MODEL)
    p.add_argument("--instruct-api-timeout", type=float, default=120.0)
    p.add_argument("--instruct-use-vllm-engine", action="store_true")
    p.add_argument("--instruct-vllm-max-model-len", type=int, default=8192)
    p.add_argument("--base-device", default="cuda:1")
    p.add_argument("--secondary-base-device", default="")
    p.add_argument("--instruct-device", default="cuda:0")
    p.add_argument("--num-futures", type=int, default=10)
    p.add_argument("--secondary-num-futures", type=int, default=0)
    p.add_argument("--future-tokens", type=int, default=20)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--max-consensus-steps", type=int, default=32)
    p.add_argument("--candidate-top-k", type=int, default=TOP_K,
                   help="Top-k candidate set size when --min-p is 0.")
    p.add_argument("--min-p", type=float, default=0.0,
                   help="True min-p threshold over the full next-token distribution. 0=use top-k.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-dir", default=None)
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--num-concurrent-cases", type=int, default=1,
                   help="Number of cases to process concurrently via threading. "
                        "Each case sends batched futures to vLLM; concurrent cases "
                        "let vLLM batch across cases too. Recommended: 4-8.")
    return p.parse_args()


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
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


class _TeeWriter:
    def __init__(self, file_obj: Any):
        self._f = file_obj

    def write(self, msg: str) -> None:
        self._f.write(msg)
        sys.stdout.write(msg)

    def flush(self) -> None:
        self._f.flush()
        sys.stdout.flush()

    def close(self) -> None:
        self._f.close()


def _vlog(verbose_log_file: Optional[Any], msg: str) -> None:
    line = str(msg)
    if not line.endswith("\n"):
        line += "\n"
    if verbose_log_file is None:
        return
    verbose_log_file.write(line)
    verbose_log_file.flush()


def _vjson(verbose_log_file: Optional[Any], title: str, obj: Any) -> None:
    _vlog(verbose_log_file, title)
    _vlog(
        verbose_log_file,
        json.dumps(obj, ensure_ascii=False, indent=2),
    )


def _short_text(text: Any, limit: int = 120) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _format_token_preview(
    tokenizer: Any,
    token_ids: List[int],
    max_items: int = 8,
) -> str:
    if not token_ids:
        return "[]"
    items: List[str] = []
    for tok_id in token_ids[:max_items]:
        tok_text = _single_token_text(tokenizer, tok_id)
        items.append(f"{tok_id}:{tok_text!r}")
    if len(token_ids) > max_items:
        items.append(f"...(+{len(token_ids) - max_items} more)")
    return "[" + ", ".join(items) + "]"


def _summarize_future_infos(future_infos: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = [f"[Step 1-2] future_sampling total={len(future_infos)}"]
    for idx, info in enumerate(future_infos):
        source = str(info.get("source", "unknown"))
        future = _short_text(info.get("future", ""), limit=240)
        lines.append(f"  future[{idx}] ({source}): {future!r}")
    return lines


def _summarize_grow_logs(
    tokenizer: Any,
    grow_logs: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = ["[Step 4-5] consensus summary:"]
    if not grow_logs:
        lines.append("  no grow steps")
        return lines

    for item in grow_logs:
        step = int(item.get("step", -1))
        meta = item.get("meta", {}) or {}
        intersection = meta.get("intersection", []) or []
        intersection_preview = _format_token_preview(tokenizer, intersection, max_items=6)

        if "accepted_token_id" in item:
            accepted_id = int(item["accepted_token_id"])
            accepted_text = item.get("accepted_token_text", _single_token_text(tokenizer, accepted_id))
            pending_text = _short_text(item.get("pending_text", ""), limit=80)
            lines.append(
                f"  step={step} intersection={intersection_preview} "
                f"accept={accepted_id}:{accepted_text!r} pending={pending_text!r}"
            )
        elif stop := item.get("stop"):
            pending_text = _short_text(item.get("pending_text", ""), limit=80)
            if stop == "empty_distribution":
                future = _short_text(item.get("future", ""), limit=80)
                reason = (item.get("dist_debug", {}) or {}).get("reason", "unknown")
                lines.append(
                    f"  step={step} stop={stop} reason={reason} future={future!r} pending={pending_text!r}"
                )
            else:
                lines.append(
                    f"  step={step} stop={stop} intersection={intersection_preview} pending={pending_text!r}"
                )

        # Print per-future candidate details for EVERY step
        per_future = item.get("per_future", [])
        if per_future:
            for fi, pf in enumerate(per_future):
                n_cand = pf.get("num_candidates", "?")
                cand_texts = pf.get("candidate_texts", [])
                cand_probs = pf.get("candidate_probs", [])
                preview = ", ".join(
                    f"{t!r}:{p:.3f}" for t, p in zip(cand_texts[:6], cand_probs[:6])
                )
                lines.append(f"    future[{fi}] candidates={n_cand}: [{preview}]")
    return lines


def _summarize_finalize_meta(finalize_meta: Dict[str, Any]) -> List[str]:
    pending_before = _short_text(finalize_meta.get("pending_before_trim", ""), limit=80)
    commit_after_trim = _short_text(finalize_meta.get("pending_after_trim", ""), limit=80)
    removed_tail = _short_text(finalize_meta.get("removed_tail_text", ""), limit=60)
    removed_disallowed = finalize_meta.get("removed_disallowed_tokens", []) or []
    removed_disallowed_texts = [str(item.get("token_text", "")) for item in removed_disallowed if item.get("token_text")]

    lines = [
        f"[Step 6-7] pending_before_trim={pending_before!r}",
        f"[Step 6-7] commit_after_trim={commit_after_trim!r}",
    ]
    if removed_tail:
        lines.append(f"[Step 6-7] removed_tail={removed_tail!r}")
    if removed_disallowed_texts:
        preview = removed_disallowed_texts[:6]
        suffix = f" ... (+{len(removed_disallowed_texts) - 6} more)" if len(removed_disallowed_texts) > 6 else ""
        lines.append(f"[Step 6-7] removed_disallowed={preview}{suffix}")
    return lines


def write_pretty_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


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


def build_future_sampling_prompt(observed_source: str) -> str:
    # Base models do next-token prediction directly on the raw text.
    # No instruction wrapping — just feed the prefix and let the model continue.
    return observed_source


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned_fragments = [
        "translate",
        "translation",
        "grammar analysis",
        "analyze",
        "analysis",
        "这句话",
        "翻译",
        "语法",
        "句子结构",
    ]
    if any(fragment in lowered for fragment in banned_fragments):
        return False
    return True


def build_translation_probe_prompt(tokenizer: Any, full_source: str, target_prefix: str) -> str:
    if not str(target_prefix or "").strip():
        messages = [{
            "role": "user",
            "content": (
                "[TASK]\n"
                "Translate the [INPUT] text into Chinese.\n\n"
                f"[INPUT]\n{full_source}\n\n"
                "[IMPORTANT]\n"
                "Start the Chinese translation from the beginning and output only the next continuation token(s)."
            ),
        }]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt += "<|im_start|>assistant\n"
        return prompt

    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] text into Chinese.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            "[IMPORTANT]\n"
            "A partial Chinese translation is already committed at the start of the assistant reply. "
            "You must continue from that exact prefix and produce only the continuation."
        ),
    }]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    prompt += target_prefix
    return prompt


def build_translation_probe_messages(full_source: str, target_prefix: str) -> List[Dict[str, str]]:
    user_content = (
        "[TASK]\n"
        "Translate the [INPUT] text into Chinese.\n\n"
        f"[INPUT]\n{full_source}\n\n"
        "[IMPORTANT]\n"
    )
    if not str(target_prefix or "").strip():
        user_content += (
            "Start the Chinese translation from the beginning and output only the next continuation token(s)."
        )
        return [{"role": "user", "content": user_content}]

    user_content += (
        "A partial Chinese translation is already committed. "
        "Continue from that exact prefix and produce only the continuation."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_prefix},
    ]


def build_final_completion_prompt(tokenizer: Any, full_source: str, committed_text: str) -> str:
    if not str(committed_text or "").strip():
        messages = [{
            "role": "user",
            "content": (
                "[TASK]\n"
                "Translate the [INPUT] text into Chinese.\n\n"
                f"[INPUT]\n{full_source}\n\n"
                "[IMPORTANT]\n"
                "Output the complete Chinese translation only."
            ),
        }]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt += "<|im_start|>assistant\n"
        return prompt

    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] text into Chinese.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            "[IMPORTANT]\n"
            "A partial Chinese translation is already committed at the start of the assistant reply. "
            "Continue from that prefix and output only the remaining continuation."
        ),
    }]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    prompt += committed_text
    return prompt


def load_causal_lm(path: str, device: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def load_vllm_engine(
    model_path: str,
    tokenizer_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> Tuple[Any, Dict[str, Any]]:
    if LLM is None or SamplingParams is None:
        raise ImportError("vLLM is not available in the current Python environment.")
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_logprobs=-1,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    return tokenizer, {"backend": "vllm_engine", "llm": llm}


def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _http_json(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer dummy"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def verify_instruct_api(api_base: str, timeout: float) -> List[str]:
    data = _http_get_json(f"{normalize_api_base(api_base)}/models", timeout=timeout)
    models = data.get("data", [])
    return [str(item.get("id", "")) for item in models if item.get("id")]


def _parse_token_id_string(raw: str) -> Optional[int]:
    text = str(raw or "").strip()
    match = re.fullmatch(r"token_id:(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _parse_completion_top_logprobs(
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
# 从 vllm completion endpoint 的返回里解析出下一个 token 的 logprobs 分布，注意它的格式和传统 LM 的 logits 输出不太一样
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
        id_distribution[tok_id] = float(math.exp(float(logprob)))

    token_ids = list(id_distribution.keys())
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [
            tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for tok_id in token_ids
        ],
        "topk_true_probs": [round(float(id_distribution[tok_id]), 6) for tok_id in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


def _parse_chat_top_logprobs(
    chat_logprobs: Optional[Dict[str, Any]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not chat_logprobs:
        return {}, {"reason": "missing_chat_logprobs"}
    content = chat_logprobs.get("content") or []
    if not content:
        return {}, {"reason": "empty_chat_logprobs"}
    first_pos = content[0] or {}
    top_logprobs = first_pos.get("top_logprobs") or []
    if not top_logprobs:
        return {}, {"reason": "empty_chat_top_logprobs"}

    id_distribution: Dict[int, float] = {}
    unknown_tokens: List[str] = []
    for item in top_logprobs:
        raw_token = item.get("token")
        logprob = item.get("logprob")
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is None:
            unknown_tokens.append(str(raw_token))
            continue
        id_distribution[tok_id] = float(math.exp(float(logprob)))

    token_ids = list(id_distribution.keys())
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_chat_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [
            tokenizer.decode(
                [tok_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for tok_id in token_ids
        ],
        "topk_true_probs": [round(float(id_distribution[tok_id]), 6) for tok_id in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode(
        [tok_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _disallowed_generation_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    forbidden_fragments = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
    ]
    if any(fragment in token_text for fragment in forbidden_fragments):
        return "special_token_text"
    # Hard constraint for committed translation tokens: avoid code-switching
    # back into English and block obvious garbled tokens.
    if re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    if "\ufffd" in token_text or "�" in token_text:
        return "replacement_char"
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
# 过滤掉那些 special token ids，或者那些看起来像特殊 token 的 ids（即使它们不在 tokenizer 的 special ids 里），因为它们不适合用来做"共识 decoding"
    filtered: Dict[int, float] = {}
    removed: List[Dict[str, Any]] = []
    for tok_id, prob in id_distribution.items():
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None:
            removed.append({
                "token_id": int(tok_id),
                "token_text": _single_token_text(tokenizer, tok_id),
                "prob": round(float(prob), 6),
                "reason": reason,
            })
            continue
        filtered[int(tok_id)] = float(prob)

    out_debug = dict(dist_debug)
    out_debug["filtered_disallowed_tokens"] = removed
    if filtered:
        out_debug["topk_token_ids"] = list(filtered.keys())
        out_debug["topk_token_texts"] = [
            _single_token_text(tokenizer, tok_id)
            for tok_id in filtered.keys()
        ]
        out_debug["topk_true_probs"] = [
            round(float(filtered[tok_id]), 6) for tok_id in filtered.keys()
        ]
    elif removed and out_debug.get("reason") == "ok":
        out_debug["reason"] = "all_top_tokens_filtered_as_disallowed"
    return filtered, out_debug


def get_model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


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


def sample_source_futures(
    base_tokenizer: Any,
    base_model: Any,
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    sample_temperature: float,
) -> List[str]:
    if not observed_source.strip():
        return []
    prompt = build_future_sampling_prompt(observed_source)
    encoded = base_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    model_device = get_model_input_device(base_model)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    with torch.inference_mode():
        generated_ids = base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=sample_temperature,
            top_p=0.95,
            top_k=50,
            max_new_tokens=future_tokens,
            num_return_sequences=num_futures,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
        )

    futures: List[str] = []
    seen = set()
    for seq in generated_ids:
        continuation_ids = seq[input_ids.shape[1]:].tolist()
        text = base_tokenizer.decode(
            continuation_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cleaned = clean_future_text(observed_source, text)
        #只保留看起来像正常英文 continuation 的 futures
        if cleaned and is_valid_future_text(cleaned) and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            futures.append(cleaned)
    return futures


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
    prompt = build_future_sampling_prompt(observed_source)
    payload = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": future_tokens,
        "temperature": sample_temperature,
        "top_p": 0.95,
        "n": num_futures,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(
        f"{normalize_api_base(api_base)}/completions",
        payload=payload,
        timeout=api_timeout,
    )
    choices = data.get("choices", [])
    futures: List[str] = []
    seen = set()
    for choice in choices:
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
    seen = set()
    for spec in base_specs:
        requested = int(spec.get("num_futures", 0) or 0)
        if requested <= 0:
            continue
        if spec["backend"] == "api":
            futures = sample_source_futures_api(
                observed_source=observed_source,
                num_futures=requested,
                future_tokens=future_tokens,
                sample_temperature=sample_temperature,
                api_base=spec["api_base"],
                api_model=spec["api_model"],
                api_timeout=spec["api_timeout"],
            )
        else:
            futures = sample_source_futures(
                base_tokenizer=spec["tokenizer"],
                base_model=spec["model"],
                observed_source=observed_source,
                num_futures=requested,
                future_tokens=future_tokens,
                sample_temperature=sample_temperature,
            )
        for future in futures:
            future_key = future.lower()
            if future_key in seen:
                continue
            seen.add(future_key)
            merged.append(future)
            merged_info.append(
                {
                    "source": spec["name"],
                    "path": spec["path"],
                    "future": future,
                }
            )
    return merged, merged_info


def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    return [tok_id for tok_id, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def minp_token_ids(dist: Dict[int, float], min_p: float) -> List[int]:
    """Return all token ids whose probability >= min_p, sorted by prob descending."""
    return [tok_id for tok_id, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True) if prob >= min_p]


def _distribution_to_debug(
    tokenizer: Any,
    id_distribution: Dict[int, float],
    policy: str,
) -> Dict[str, Any]:
    token_ids = list(id_distribution.keys())
    token_texts = [
        tokenizer.decode(
            [tok_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for tok_id in token_ids
    ]
    probs = [round(float(id_distribution[tok_id]), 6) for tok_id in token_ids]
    return {
        "reason": "ok" if id_distribution else "empty_distribution",
        "candidate_policy": policy,
        "topk_token_ids": token_ids,
        "topk_token_texts": token_texts,
        "topk_true_probs": probs,
        "candidate_token_ids": token_ids,
        "candidate_token_texts": token_texts,
        "candidate_probs": probs,
    }


def _is_vllm_engine_model(model: Any) -> bool:
    return isinstance(model, dict) and model.get("backend") == "vllm_engine"


def _vllm_logprob_dict_to_distribution(
    tokenizer: Any,
    logprob_dict: Any,
    policy: str,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not logprob_dict:
        return {}, {"reason": "empty_vllm_logprobs", "candidate_policy": policy}
    id_distribution = {
        int(tok_id): float(math.exp(float(obj.logprob)))
        for tok_id, obj in logprob_dict.items()
    }
    dist_debug = _distribution_to_debug(tokenizer, id_distribution, policy=policy)
    return id_distribution, dist_debug


def get_next_token_distribution(
    tokenizer: Any,
    model: Optional[Any],
    full_source: str,
    target_prefix: str,
    top_k: int = TOP_K,
    min_p: float = 0.0,
    api_base: Optional[str] = None,
    api_model: Optional[str] = None,
    api_timeout: float = 120.0,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
# 你必须从这个 prefix 继续往后翻， 现在我只关心"下一个 token 是什么"
    prompt = build_translation_probe_prompt(tokenizer, full_source, target_prefix)
    # 2种途径得到下一个 token 的分布：vllm本地模型推理，或者调用 API 获取 logits
    if api_base:
        logprobs_n = max(top_k, 20) if min_p > 0 else top_k
        payload = {
            "model": api_model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": logprobs_n,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        data = _http_json(
            f"{normalize_api_base(api_base)}/completions",
            payload=payload,
            timeout=api_timeout,
        )
        choices = data.get("choices", [])
        if not choices:
            return {}, {"reason": "missing_choice", "raw_response": data}
        choice = choices[0]
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
        dist, dist_debug = _parse_completion_top_logprobs(
            logprobs.get("top_logprobs"),
            tokenizer=tokenizer,
        )
        dist_debug["returned_token_ids"] = choice.get("token_ids")
        dist_debug["returned_tokens"] = logprobs.get("tokens", [])
        dist_debug["api_backend"] = "vllm_completion"
        return filter_distribution_token_ids(tokenizer, dist, dist_debug)

    if _is_vllm_engine_model(model):
        llm = model["llm"]
        logprobs_arg = -1 if min_p > 0 else max(1, int(top_k))
        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=logprobs_arg,
            detokenize=False,
        )
        outs = llm.generate([prompt], sampling_params=sp)
        output = outs[0].outputs[0]
        dist, dist_debug = _vllm_logprob_dict_to_distribution(
            tokenizer,
            output.logprobs[0] if output.logprobs else None,
            policy="min_p_full_distribution" if min_p > 0 else "top_k",
        )
        dist_debug["api_backend"] = "vllm_engine"
        if min_p > 0:
            dist_debug["min_p"] = min_p
        else:
            dist_debug["candidate_top_k"] = top_k
        return filter_distribution_token_ids(tokenizer, dist, dist_debug)

    if model is None:
        return {}, {"reason": "missing_local_model"}


    # 路线B： local本地模型推理得到下一个 token 的 logits 分布
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    model_device = get_model_input_device(model)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded)
        # Quantized / low-precision logits can make the displayed probabilities
        # numerically inconsistent (for example 1.0 plus other positive values).
        # Cast to float32 before softmax so debug probabilities remain trustworthy.
        next_token_logits = outputs.logits[0, -1, :].float()

    full_probs = torch.softmax(next_token_logits, dim=-1)
    if min_p > 0:
        candidate_indices = torch.nonzero(full_probs >= min_p, as_tuple=False).flatten()
        candidate_probs = full_probs[candidate_indices]
        if candidate_indices.numel() > 0:
            candidate_probs, order = torch.sort(candidate_probs, descending=True)
            candidate_indices = candidate_indices[order]
        token_ids = candidate_indices.tolist()
        probs = candidate_probs.tolist()
        id_distribution = {
            int(tok_id): float(prob)
            for tok_id, prob in zip(token_ids, probs)
        }
        dist_debug = _distribution_to_debug(tokenizer, id_distribution, policy="min_p_full_distribution")
        dist_debug["min_p"] = min_p
    else:
        topk_probs, topk_indices = torch.topk(full_probs, k=top_k, dim=-1)
        token_ids = topk_indices.tolist()
        probs = topk_probs.tolist()
        id_distribution = {
            int(tok_id): float(prob)
            for tok_id, prob in zip(token_ids, probs)
        }
        dist_debug = _distribution_to_debug(tokenizer, id_distribution, policy="top_k")
        dist_debug["candidate_top_k"] = top_k
    return filter_distribution_token_ids(tokenizer, id_distribution, dist_debug)


def batch_get_next_token_distributions(
    tokenizer: Any,
    model: Optional[Any],
    full_sources: List[str],
    target_prefix: str,
    top_k: int = TOP_K,
    min_p: float = 0.0,
    api_base: Optional[str] = None,
    api_model: Optional[str] = None,
    api_timeout: float = 120.0,
) -> List[Tuple[Dict[int, float], Dict[str, Any]]]:
    """Batched next-token distributions for either API or local inference."""
    prompts = [
        build_translation_probe_prompt(tokenizer, full_source, target_prefix)
        for full_source in full_sources
    ]
    if not prompts:
        return []

    if api_base:
        logprobs_n = max(top_k, 20) if min_p > 0 else top_k
        payload = {
            "model": api_model,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": logprobs_n,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        data = _http_json(
            f"{normalize_api_base(api_base)}/completions",
            payload=payload,
            timeout=api_timeout,
        )
        choices = data.get("choices", [])
        results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
        for i in range(len(prompts)):
            if i >= len(choices):
                results.append(({}, {"reason": "missing_choice", "raw_response": data}))
                continue
            choice = choices[i]
            logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
            dist, dist_debug = _parse_completion_top_logprobs(
                logprobs.get("top_logprobs"),
                tokenizer=tokenizer,
            )
            dist_debug["returned_token_ids"] = choice.get("token_ids")
            dist_debug["returned_tokens"] = logprobs.get("tokens", [])
            dist_debug["api_backend"] = "vllm_completion"
            if min_p > 0:
                dist_debug["candidate_policy"] = "min_p"
                dist_debug["min_p"] = min_p
            else:
                dist_debug["candidate_policy"] = "top_k"
                dist_debug["candidate_top_k"] = top_k
            results.append(filter_distribution_token_ids(tokenizer, dist, dist_debug))
        return results

    if model is None:
        return [({}, {"reason": "missing_local_model"}) for _ in prompts]

    if _is_vllm_engine_model(model):
        llm = model["llm"]
        logprobs_arg = -1 if min_p > 0 else max(1, int(top_k))
        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=logprobs_arg,
            detokenize=False,
        )
        outs = llm.generate(prompts, sampling_params=sp)
        results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
        for out in outs:
            output = out.outputs[0]
            dist, dist_debug = _vllm_logprob_dict_to_distribution(
                tokenizer,
                output.logprobs[0] if output.logprobs else None,
                policy="min_p_full_distribution" if min_p > 0 else "top_k",
            )
            dist_debug["api_backend"] = "vllm_engine"
            if min_p > 0:
                dist_debug["min_p"] = min_p
            else:
                dist_debug["candidate_top_k"] = top_k
            results.append(filter_distribution_token_ids(tokenizer, dist, dist_debug))
        return results

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    model_device = get_model_input_device(model)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    attention_mask = encoded["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1

    with torch.inference_mode():
        outputs = model(**encoded)
        batch_logits = outputs.logits.float()

    next_token_logits = batch_logits[torch.arange(batch_logits.shape[0], device=batch_logits.device), last_positions, :]
    full_probs_batch = torch.softmax(next_token_logits, dim=-1)

    results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
    vocab_size = full_probs_batch.shape[-1]
    for row_probs in full_probs_batch:
        if min_p > 0:
            candidate_indices = torch.nonzero(row_probs >= min_p, as_tuple=False).flatten()
            candidate_probs = row_probs[candidate_indices]
            if candidate_indices.numel() > 0:
                candidate_probs, order = torch.sort(candidate_probs, descending=True)
                candidate_indices = candidate_indices[order]
            token_ids = candidate_indices.tolist()
            probs = candidate_probs.tolist()
            id_distribution = {
                int(tok_id): float(prob)
                for tok_id, prob in zip(token_ids, probs)
            }
            dist_debug = _distribution_to_debug(tokenizer, id_distribution, policy="min_p_full_distribution")
            dist_debug["min_p"] = min_p
        else:
            k = min(top_k, vocab_size)
            topk_probs, topk_indices = torch.topk(row_probs, k=k, dim=-1)
            token_ids = topk_indices.tolist()
            probs = topk_probs.tolist()
            id_distribution = {
                int(tok_id): float(prob)
                for tok_id, prob in zip(token_ids, probs)
            }
            dist_debug = _distribution_to_debug(tokenizer, id_distribution, policy="top_k")
            dist_debug["candidate_top_k"] = top_k
        results.append(filter_distribution_token_ids(tokenizer, id_distribution, dist_debug))
    return results


def choose_consensus_token(
    distributions: List[Dict[int, float]],
    min_p: float = MIN_P,
    candidate_top_k: int = TOP_K,
) -> Tuple[Optional[int], Dict[str, Any]]:
    if not distributions:
        return None, {"reason": "no_distributions"}

    if min_p > 0:
        candidate_lists = [minp_token_ids(dist, min_p) for dist in distributions]
    else:
        candidate_lists = [topk_token_ids(dist, candidate_top_k) for dist in distributions]

    intersection = set(candidate_lists[0])
    for clist in candidate_lists[1:]:
        intersection &= set(clist)

    if not intersection:
        return None, {
            "reason": "empty_intersection",
            "topk_lists": candidate_lists,
        }

    best_token = max(
        intersection,
        key=lambda tok: sum(dist.get(tok, 0.0) for dist in distributions) / len(distributions),
    )
    return best_token, {
        "reason": "ok",
        "intersection": sorted(intersection),
        "avg_topk_score": sum(dist.get(best_token, 0.0) for dist in distributions) / len(distributions),
        "topk_lists": candidate_lists,
    }


def decode_token_ids_to_text(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def inspect_token_ids(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
# 给一串 token ids，解码成文本，并且看看最后一个 token 是什么，主要是为了判断"待定" token buffer 里是否有一些 token 导致文本结尾不正常（比如半个字被切断了），从而决定是否要把它剔除掉，直到剩下的部分可以安全提交了
    decoded_text = decode_token_ids_to_text(tokenizer, token_ids)
    last_token_id = token_ids[-1] if token_ids else None
    last_token_text = (
        _single_token_text(tokenizer, last_token_id)
        if last_token_id is not None
        else ""
    )
    return {
        "decoded_text": decoded_text,
        "last_token_id": last_token_id,
        "last_token_text": last_token_text,
    }


def has_suspicious_tail(text: str, last_token_text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in last_token_text or "�" in last_token_text:
        return True
    if text.endswith("\ufffd") or text.endswith("�"):
        return True
    if "\ufffd" in text[-4:] or "�" in text[-4:]:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_pending_token_ids(
    tokenizer: Any,
    pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]]]:
# 先做一轮"全串"清洗：只要 buffer 中任意位置出现明显英文 / 乱码 token，就直接剔除，
# 不再等它碰巧落在最后一个 token 再被 tail trim 看到。
    kept: List[int] = []
    removed: List[Dict[str, Any]] = []
    for idx, tok_id in enumerate(pending_token_ids):
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is None:
            kept.append(tok_id)
            continue
        removed.append({
            "position": idx,
            "token_id": int(tok_id),
            "token_text": _single_token_text(tokenizer, tok_id),
            "reason": reason,
        })
    return kept, removed


def trim_pending_tokens_to_complete_boundary(
    tokenizer: Any,
    committed_text: str,
    pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[int], Dict[str, Any]]:
# 从"待定" token ids 里剔除掉那些导致文本结尾不正常的 token ids，直到剩下的部分可以安全提交了
    work, removed_disallowed_tokens = sanitize_pending_token_ids(tokenizer, pending_token_ids)
    removed_tail_token_ids: List[int] = []
    while work:
        pending_view = inspect_token_ids(tokenizer, work)
        full_text = committed_text + pending_view["decoded_text"]
        if not has_suspicious_tail(full_text, pending_view["last_token_text"]):#如果文本结尾看起来正常了，就停下来，不要再剔除掉更多的 token ids 了
            pending_view["full_text"] = full_text
            return work, removed_disallowed_tokens, removed_tail_token_ids, pending_view
        removed_tail_token_ids.insert(0, work.pop())

    return [], removed_disallowed_tokens, removed_tail_token_ids, {
        "decoded_text": "",
        "last_token_id": None,
        "last_token_text": "",
        "full_text": committed_text,
    }


def finalize_external_commit( #把"待定" token ids 里真正可以提交的部分提交掉
    tokenizer: Any,
    committed_text: str,
    pending_token_ids: List[int],
) -> Tuple[str, str, Dict[str, Any]]:

    trimmed_pending, removed_disallowed_tokens, removed_tail_token_ids, pending_view = trim_pending_tokens_to_complete_boundary(
        tokenizer,
        committed_text,
        pending_token_ids,
    )#把 pending_token_ids 里那些导致文本结尾不正常的 token ids 都剔除掉，剩下的 trimmed_pending 就是可以安全提交的部分了
    commit_text = pending_view["decoded_text"]
    new_committed = committed_text + commit_text
    return new_committed, commit_text, {
        "pending_token_ids_before_trim": pending_token_ids,
        "pending_before_trim": decode_token_ids_to_text(tokenizer, pending_token_ids),
        "pending_token_ids_after_trim": trimmed_pending,
        "pending_after_trim": commit_text,
        "removed_disallowed_tokens": removed_disallowed_tokens,
        "removed_tail_token_ids": removed_tail_token_ids,
        "removed_tail_text": decode_token_ids_to_text(tokenizer, removed_tail_token_ids),
        "removed_token_ids": [item["token_id"] for item in removed_disallowed_tokens] + removed_tail_token_ids,
        "last_token_id_after_trim": pending_view["last_token_id"],
        "last_token_text_after_trim": pending_view["last_token_text"],
        "full_text_after_trim": pending_view["full_text"],
    }


def extend_pending_tokens(
    instruct_tokenizer: Any,
    instruct_model: Optional[Any],
    source_observed: str,
    futures: List[str],
    committed_text: str,
    max_consensus_steps: int,
    candidate_top_k: int = TOP_K,
    instruct_api_base: Optional[str] = None,
    instruct_api_model: Optional[str] = None,
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
) -> Tuple[List[int], List[Dict[str, Any]]]:
# 在 committed_text 的基础上，基于多个 futures 进行多轮"共识 decoding"，不断往后找下一个 consensus token，直到没有共识 token 或者达到 max_consensus_steps
    pending_token_ids: List[int] = [] # token buffer，存储从上次提交后的"待定" token ids
    grow_logs: List[Dict[str, Any]] = [] 

    for step_idx in range(max_consensus_steps):
        target_prefix = committed_text + decode_token_ids_to_text(instruct_tokenizer, pending_token_ids) #得到上一轮的 token id + committed target

        distributions: List[Dict[int, float]] = []
        per_future: List[Dict[str, Any]] = []
        # 对每个 future，构造一个"完整 source 假设"
        full_sources = [append_text_continuation(source_observed, f) for f in futures]

        batch_results = batch_get_next_token_distributions(
            tokenizer=instruct_tokenizer,
            model=instruct_model,
            full_sources=full_sources,
            target_prefix=target_prefix,
            top_k=candidate_top_k,
            min_p=min_p,
            api_base=instruct_api_base,
            api_model=instruct_api_model,
            api_timeout=instruct_api_timeout,
        )

        for i, (dist, dist_debug) in enumerate(batch_results):
            if not dist: # 如果某个 future 没有有效分布，立刻停
                grow_logs.append({
                    "step": step_idx,
                    "stop": "empty_distribution",
                    "future": futures[i],
                    "pending_token_ids": pending_token_ids.copy(),
                    "pending_text": decode_token_ids_to_text(instruct_tokenizer, pending_token_ids),
                    "dist_debug": dist_debug,
                })
                return pending_token_ids, grow_logs

            distributions.append(dist)
            topk_ids = topk_token_ids(dist, candidate_top_k)

            if min_p > 0:
                candidate_ids = minp_token_ids(dist, min_p)
            else:
                candidate_ids = topk_ids
            per_future.append({
                "future": futures[i],
                "topk_token_ids": topk_ids,
                "topk_token_texts": [
                    instruct_tokenizer.decode(
                        [tok_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for tok_id in topk_ids
                ],
                "topk_true_probs": dist_debug.get("topk_true_probs", []),
                "candidate_ids": candidate_ids,
                "candidate_texts": [
                    instruct_tokenizer.decode(
                        [tok_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for tok_id in candidate_ids
                ],
                "candidate_probs": [dist.get(tok_id, 0.0) for tok_id in candidate_ids],
                "num_candidates": len(candidate_ids),
            })
        # 对这些 top-k 列表做consensus，看看有没有交集，如果有交集就选一个平均概率最高的 token 作为 consensus token，加入 pending token buffer 里；如果没有交集了，就停
        consensus_token_id, meta = choose_consensus_token(
            distributions,
            min_p=min_p,
            candidate_top_k=candidate_top_k,
        )
        if consensus_token_id is None:
            grow_logs.append({
                "step": step_idx,
                "stop": "no_consensus_token",
                "pending_token_ids": pending_token_ids.copy(),
                "pending_text": decode_token_ids_to_text(instruct_tokenizer, pending_token_ids),
                "per_future": per_future,
                "meta": meta,
            })
            break

        pending_token_ids.append(consensus_token_id) #把 consensus token id 加入 pending token buffer 里，进入下一轮
        # inspect 当前 buffer，并记日志
        pending_view = inspect_token_ids(instruct_tokenizer, pending_token_ids)
        grow_logs.append({
            "step": step_idx,
            "accepted_token_id": consensus_token_id,
            "accepted_token_text": pending_view["last_token_text"],
            "pending_token_ids": pending_token_ids.copy(),
            "pending_text": pending_view["decoded_text"],
            "per_future": per_future,
            "meta": meta,
        })

    return pending_token_ids, grow_logs #返回最终的 pending token buffer，以及每一步的日志，日志里包含每个 future 的 top-k token ids 和文本，以及 consensus 的结果和一些 debug 信息


def force_complete_translation(
    tokenizer: Any,
    model: Optional[Any],
    full_source: str,
    committed_text: str,
    api_base: Optional[str] = None,
    api_model: Optional[str] = None,
    api_timeout: float = 120.0,
) -> str:
    prompt = build_final_completion_prompt(tokenizer, full_source, committed_text)
    if api_base:
        payload = {
            "model": api_model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        }
        data = _http_json(
            f"{normalize_api_base(api_base)}/completions",
            payload=payload,
            timeout=api_timeout,
        )
        choices = data.get("choices", [])
        if not choices:
            return ""
        raw = str(choices[0].get("text", ""))
        return clean_model_text(raw)

    if model is None:
        return ""

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    model_device = get_model_input_device(model)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    continuation_ids = generated_ids[0, input_ids.shape[1]:].tolist()
    raw = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_model_text(raw)


def _extract_reference_text_from_row(row: Dict[str, Any]) -> Optional[str]:
    candidate_keys = [
        "llm_reference_text",
        "tgt_text_full",
        "tgt_text",
        "target_text",
        "translation",
        "ref_text",
        "reference",
    ]
    for key in candidate_keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


def compute_laal(
    source_chunks: List[str],
    target_deltas: List[str],
    actions: List[str],
    reference: str,
) -> float:
    timeline: List[int] = []
    source_read = 0

    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        words_in_chunk = len(str(chunk).strip().split()) if str(chunk).strip() else 0
        source_read += words_in_chunk
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)

    y = "".join(d for d in target_deltas if d)
    y_len = len(y)
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())

    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")

    denom = max(y_len, yref_len)
    if denom <= 0 or len(timeline) == 0:
        return float("nan")

    total_lagging = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        d_star_i = (i - 1) * x_len / denom
        total_lagging += (d_i - d_star_i)
    return total_lagging / denom


def _char_tokens_zh(text: str) -> List[str]:
    return [c for c in str(text) if not c.isspace()]


def compute_bleu_char(hypothesis: str, reference: str, max_order: int = 4, smooth: bool = True) -> float:
    hyp = _char_tokens_zh(hypothesis)
    ref = _char_tokens_zh(reference)
    hyp_len = len(hyp)
    ref_len = len(ref)

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
        clipped = 0
        for ng, cnt in hyp_ngrams.items():
            clipped += min(cnt, ref_ngrams.get(ng, 0))
        if smooth:
            p_n = (clipped + 1.0) / (total + 1.0)
        else:
            if clipped == 0:
                return 0.0
            p_n = clipped / total
        precisions.append(p_n)

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / eff_order)
    return bleu * 100.0


def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    instruct_model: Optional[Any],
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = parse_trajectory(row["src_trajectory"]) #把 source trajectory 还原成 streaming 输入
    full_source_text = get_full_source_text(row)

    committed_text = ""
    target_deltas: List[str] = []
    actions: List[str] = []

    _vlog(verbose_log_file, "\n" + "#" * 60)
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source_text}")
    _vlog(verbose_log_file, f"# Chunks: {len(chunks)}")
    candidate_policy = (
        f"true_min_p_full_distribution(threshold={args.min_p})"
        if args.min_p > 0 else
        f"top_k(k={args.candidate_top_k})"
    )
    _vlog(verbose_log_file, f"# num_futures={args.num_futures}, candidate_policy={candidate_policy}")
    for spec in base_specs:
        backend_desc = (
            f"api model={spec['api_model']} base={normalize_api_base(spec['api_base'])}"
            if spec["backend"] == "api" else
            f"local path={spec['path']} device={spec['device']}"
        )
        _vlog(
            verbose_log_file,
            f"# base_model[{spec['name']}]: backend={spec['backend']} {backend_desc} num_futures={spec['num_futures']}",
        )
    _vlog(
        verbose_log_file,
        "# decoder_impl: strict token-id level; "
        "candidate sets are intersected by authoritative token ids rather than decoded token strings.",
    )
    _vlog(
        verbose_log_file,
        f"# instruct_backend: {'vllm_api' if args.instruct_api_base else ('vllm_engine' if args.instruct_use_vllm_engine else 'local_transformers')}",
    )
    _vlog(verbose_log_file, "#" * 60)

    for t in range(len(chunks)):
        source_observed = build_source_observed(chunks, t) #当前时刻系统已经看到的 source prefix
        _vlog(verbose_log_file, "\n" + "=" * 60)
        _vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}")
        _vlog(verbose_log_file, f"source_observed: {source_observed!r}")
        _vlog(verbose_log_file, f"committed_before: {committed_text!r}")

        if t == len(chunks) - 1:
            final_delta = force_complete_translation( # last chunk 
                tokenizer=instruct_tokenizer,
                model=instruct_model,
                full_source=full_source_text,
                committed_text=committed_text,
                api_base=args.instruct_api_base or None,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
            )
            if final_delta:
                committed_text += final_delta
                target_deltas.append(final_delta)
                actions.append("WRITE")
                _vlog(verbose_log_file, "[FinalChunk] force completion")
                _vlog(verbose_log_file, f"final_delta: {final_delta!r}")
                _vlog(verbose_log_file, f"committed_after: {committed_text!r}")
            else:
                target_deltas.append("")
                actions.append("READ")
                _vlog(verbose_log_file, "[FinalChunk] no new delta")
            continue
        # sampling future continuation from base Model
        futures, future_infos = sample_source_futures_multi(
            base_specs=base_specs,
            observed_source=source_observed,
            future_tokens=args.future_tokens,
            sample_temperature=args.sample_temperature,
        )
        for line in _summarize_future_infos(future_infos):
            _vlog(verbose_log_file, line)

        if len(futures) < 2:
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "-> READ (too few futures)")
            continue

        pending_token_ids, grow_logs = extend_pending_tokens( #在 committed_text 的基础上，基于多个 futures 进行多轮"共识 decoding"，不断往后找下一个 consensus token，直到没有共识 token 或者达到 max_consensus_steps
            instruct_tokenizer=instruct_tokenizer,
            instruct_model=instruct_model,
            source_observed=source_observed,
            futures=futures,
            committed_text=committed_text,
            max_consensus_steps=args.max_consensus_steps,
            candidate_top_k=args.candidate_top_k,
            instruct_api_base=args.instruct_api_base or None,
            instruct_api_model=args.instruct_api_model,
            instruct_api_timeout=args.instruct_api_timeout,
            min_p=args.min_p,
        )
        for line in _summarize_grow_logs(instruct_tokenizer, grow_logs):
            _vlog(verbose_log_file, line)


        new_committed, delta, finalize_meta = finalize_external_commit(#把 pending_token_ids 里真正可以提交的部分提交掉，剩下的部分继续保留在 pending 里
            tokenizer=instruct_tokenizer,
            committed_text=committed_text,
            pending_token_ids=pending_token_ids,
        )
        action = "WRITE" if delta else "READ"
        target_deltas.append(delta)
        actions.append(action)
        for line in _summarize_finalize_meta(finalize_meta):
            _vlog(verbose_log_file, line)
        _vlog(verbose_log_file, f"-> {action} delta={delta!r}")
        _vlog(verbose_log_file, f"committed_after: {new_committed!r}")
        committed_text = new_committed

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "src_trajectory": chunks,
        "source_full_text": full_source_text,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {
            "strict_token_id_level": True,
            "next_token_source": (
                "vllm_completion_logprobs_token_ids"
                if args.instruct_api_base else
                ("vllm_engine_generate_logprobs" if args.instruct_use_vllm_engine else "direct_local_transformers_logits")
            ),
            "token_id_source": (
                "authoritative token ids returned by vLLM OpenAI-compatible completion endpoint"
                if args.instruct_api_base else
                ("authoritative token ids returned by local vLLM engine" if args.instruct_use_vllm_engine else "authoritative local tokenizer/model token ids")
            ),
            "candidate_policy": candidate_policy,
        },
    }

    reference_text = _extract_reference_text_from_row(row)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    laal_error: Optional[str] = None
    bleu_char_error: Optional[str] = None

    try:
        if not reference_text:
            raise ValueError("reference_text_unavailable")
        laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
        bleu_char_value = compute_bleu_char(committed_text, reference_text)
    except Exception as e:
        laal_error = str(e)
        bleu_char_error = str(e)

    result["reference_text"] = reference_text or ""
    result["metrics"] = {
        "laal_text": laal_value,
        "bleu_char": bleu_char_value,
        "laal_error": laal_error,
        "bleu_char_error": bleu_char_error,
    }

    _vlog(verbose_log_file, "\n" + "-" * 60)
    _vlog(verbose_log_file, f"prediction: {committed_text!r}")
    _vlog(verbose_log_file, f"reference: {(reference_text or '')!r}")
    _vlog(verbose_log_file, f"bleu_char: {bleu_char_value}")
    _vlog(verbose_log_file, f"laal_text: {laal_value}")

    return result


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_column].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected

    if args.test_one:
        return df.iloc[[args.row_idx]]

    max_rows = max(1, int(args.max_rows))
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max_rows)
    return df.iloc[start:end]


def main() -> None:
    setup_env()
    args = parse_args()
    args.base_api_base = str(args.base_api_base or "").strip()
    args.base_api_model = str(args.base_api_model or "").strip()
    args.secondary_base_api_base = str(args.secondary_base_api_base or "").strip()
    args.secondary_base_api_model = str(args.secondary_base_api_model or "").strip()
    args.instruct_api_base = str(args.instruct_api_base or "").strip()
    args.secondary_base_model_path = str(args.secondary_base_model_path or "").strip()
    args.secondary_base_device = str(args.secondary_base_device or "").strip()

    if args.num_futures <= 0:
        raise ValueError("--num-futures must be > 0")
    if args.candidate_top_k <= 0:
        raise ValueError("--candidate-top-k must be > 0")
    if args.min_p < 0:
        raise ValueError("--min-p must be >= 0")
    if args.secondary_num_futures < 0:
        raise ValueError("--secondary-num-futures must be >= 0")
    if args.secondary_num_futures > args.num_futures:
        raise ValueError("--secondary-num-futures cannot exceed --num-futures")
    if args.base_api_base and not args.base_api_model:
        raise ValueError("--base-api-base requires --base-api-model")
    if args.secondary_base_api_base and not args.secondary_base_api_model:
        raise ValueError("--secondary-base-api-base requires --secondary-base-api-model")
    if args.secondary_num_futures > 0 and not (args.secondary_base_model_path or args.secondary_base_api_base):
        raise ValueError("--secondary-num-futures > 0 requires a secondary base model path or API")
    if (args.secondary_base_model_path or args.secondary_base_api_base) and args.secondary_num_futures <= 0:
        raise ValueError("secondary base configuration requires --secondary-num-futures > 0")
    if args.instruct_use_vllm_engine and args.instruct_api_base:
        raise ValueError("--instruct-use-vllm-engine cannot be combined with --instruct-api-base")

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)

    primary_num_futures = args.num_futures - args.secondary_num_futures
    if primary_num_futures <= 0:
        raise ValueError("primary base model must keep at least 1 future")

    base_specs: List[Dict[str, Any]] = []
    if args.base_api_base:
        models = verify_instruct_api(args.base_api_base, args.base_api_timeout)
        if args.base_api_model not in models:
            raise RuntimeError(
                f"base_api_model '{args.base_api_model}' not found at "
                f"{normalize_api_base(args.base_api_base)}; available={models}"
            )
        base_specs.append(
            {
                "name": "primary",
                "backend": "api",
                "path": args.base_model_path,
                "device": "api",
                "tokenizer": None,
                "model": None,
                "num_futures": primary_num_futures,
                "api_base": args.base_api_base,
                "api_model": args.base_api_model,
                "api_timeout": args.base_api_timeout,
            }
        )
    else:
        base_tokenizer, base_model = load_causal_lm(args.base_model_path, args.base_device)
        base_specs.append(
            {
                "name": "primary",
                "backend": "local",
                "path": args.base_model_path,
                "device": args.base_device,
                "tokenizer": base_tokenizer,
                "model": base_model,
                "num_futures": primary_num_futures,
                "api_base": "",
                "api_model": "",
                "api_timeout": 0.0,
            }
        )
    if args.secondary_base_api_base:
        models = verify_instruct_api(args.secondary_base_api_base, args.secondary_base_api_timeout)
        if args.secondary_base_api_model not in models:
            raise RuntimeError(
                f"secondary_base_api_model '{args.secondary_base_api_model}' not found at "
                f"{normalize_api_base(args.secondary_base_api_base)}; available={models}"
            )
        base_specs.append(
            {
                "name": "secondary",
                "backend": "api",
                "path": args.secondary_base_model_path,
                "device": "api",
                "tokenizer": None,
                "model": None,
                "num_futures": args.secondary_num_futures,
                "api_base": args.secondary_base_api_base,
                "api_model": args.secondary_base_api_model,
                "api_timeout": args.secondary_base_api_timeout,
            }
        )
    elif args.secondary_base_model_path:
        secondary_device = args.secondary_base_device or args.base_device
        secondary_tokenizer, secondary_model = load_causal_lm(
            args.secondary_base_model_path,
            secondary_device,
        )
        base_specs.append(
            {
                "name": "secondary",
                "backend": "local",
                "path": args.secondary_base_model_path,
                "device": secondary_device,
                "tokenizer": secondary_tokenizer,
                "model": secondary_model,
                "num_futures": args.secondary_num_futures,
                "api_base": "",
                "api_model": "",
                "api_timeout": 0.0,
            }
        )

    for spec in base_specs:
        if spec["backend"] == "api":
            print(
                f"[Base] name={spec['name']} backend=api model={spec['api_model']} "
                f"api={normalize_api_base(spec['api_base'])} num_futures={spec['num_futures']}"
            )
        else:
            print(
                f"[Base] name={spec['name']} backend=local path={spec['path']} "
                f"device={spec['device']} num_futures={spec['num_futures']}"
            )
    if args.instruct_api_base:
        models = verify_instruct_api(args.instruct_api_base, args.instruct_api_timeout)
        if args.instruct_api_model not in models:
            raise RuntimeError(
                f"instruct_api_model '{args.instruct_api_model}' not found at "
                f"{normalize_api_base(args.instruct_api_base)}; available={models}"
            )
        instruct_tokenizer = load_tokenizer(args.instruct_tokenizer_path)
        instruct_model = None
        print(
            f"[Instruct] Using vLLM API at {normalize_api_base(args.instruct_api_base)} "
            f"with model={args.instruct_api_model}"
        )
    elif args.instruct_use_vllm_engine:
        instruct_tokenizer, instruct_model = load_vllm_engine(
            args.instruct_model_path,
            args.instruct_tokenizer_path,
            args.gpu_memory_utilization,
            args.instruct_vllm_max_model_len,
        )
        print(
            f"[Instruct] Using local vLLM engine at {args.instruct_model_path} "
            f"(gpu_mem={args.gpu_memory_utilization}, max_model_len={args.instruct_vllm_max_model_len})"
        )
    else:
        instruct_tokenizer, instruct_model = load_causal_lm(
            args.instruct_model_path,
            args.instruct_device,
        )
        print(
            f"[Instruct] Using local transformers model at {args.instruct_model_path} "
            f"on {args.instruct_device}"
        )

    output_dir: Optional[str] = None
    if args.output_jsonl:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        os.makedirs(output_dir, exist_ok=True)
    if args.verbose and args.verbose_dir:
        os.makedirs(args.verbose_dir, exist_ok=True)

    def _process_one_row(row_idx: int, series_dict: Dict[str, Any]) -> Dict[str, Any]:
        row = series_dict
        utt_id = str(row.get(args.id_column, row.get("id", f"row_{row_idx}")))

        verbose_log_file: Optional[Any] = None
        if args.verbose:
            if args.verbose_dir:
                verbose_path = os.path.join(
                    args.verbose_dir,
                    f"verbose_{sanitize_filename(utt_id)}.log",
                )
                raw_file = open(verbose_path, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file) if args.test_one else raw_file
                _vlog(verbose_log_file, f"[VerboseLog] writing to {verbose_path}")
            elif args.test_one:
                verbose_log_file = sys.stdout

        try:
            result = run_one_utterance(
                row=row,
                args=args,
                base_specs=base_specs,
                instruct_tokenizer=instruct_tokenizer,
                instruct_model=instruct_model,
                verbose_log_file=verbose_log_file,
            )
        finally:
            if verbose_log_file is not None and args.verbose_dir:
                verbose_log_file.close()

        if output_dir is not None:
            out_path = os.path.join(output_dir, f"{sanitize_filename(result['utt_id'])}.json")
            write_pretty_json(out_path, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    num_concurrent = max(1, args.num_concurrent_cases)
    row_items = [(row_idx, series.to_dict()) for row_idx, (_, series) in enumerate(rows.iterrows())]

    if num_concurrent <= 1:
        # 串行模式，和之前一样
        for row_idx, row_dict in row_items:
            _process_one_row(row_idx, row_dict)
    else:
        # 并行模式：多个 case 同时跑，vLLM continuous batching 自动合并
        print(f"[Concurrent] Processing {len(row_items)} rows with {num_concurrent} concurrent cases")
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures_map = {
                executor.submit(_process_one_row, row_idx, row_dict): row_idx
                for row_idx, row_dict in row_items
            }
            for future in as_completed(futures_map):
                ridx = futures_map[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] Row {ridx} raised: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
