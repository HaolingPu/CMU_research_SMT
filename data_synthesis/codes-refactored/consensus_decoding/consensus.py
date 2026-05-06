"""Candidate selection, consensus token choice, growth loop, and tail flush."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .cli import MIN_P, TOP_K
from .distributions import batch_get_next_token_distributions
from .http_client import _http_json, normalize_api_base
from .prompts import build_final_completion_prompt
from .text_utils import append_text_continuation
from .tokens import _single_token_text, decode_token_ids_to_text, inspect_token_ids


# ---------------------------------------------------------------------------
# Candidate selection policies
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
# Growth loop and tail flush
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

    for step_idx in range(max_consensus_steps): #最多32 步
        # 2a) 每条 future 拼成完整 source，问 instruct 模型："给定这个完整英文 + 已有中文前缀，下一个中文 token 是啥？"
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
            # select 3 policies: topk, topp, minp
            candidate_ids = _select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
            per_future.append({
                "future": futures[i],
                "candidate_texts": [_single_token_text(instruct_tokenizer, t) for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        # 共识选 token
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


def force_complete_translation(
    tokenizer: Any,
    full_source: str,
    committed_text: str,
    api_base: str,
    api_model: str,
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
    max_tokens: int = 128,
) -> str:
    prompt = build_final_completion_prompt(tokenizer, full_source, committed_text, target_lang=target_lang)
    payload = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": max(1, int(max_tokens)),
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，带上已提交翻译前缀，让模型把剩余翻译补完
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("text", ""))
