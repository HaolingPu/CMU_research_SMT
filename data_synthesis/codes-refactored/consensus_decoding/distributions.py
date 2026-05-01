"""Batch next-token distribution probing via vLLM /completions logprobs."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .cli import MIN_P, TOP_K
from .http_client import _http_json, normalize_api_base
from .prompts import build_translation_probe_prompt_prefix_token_ids
from .tokens import _single_token_text, filter_distribution_token_ids


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


# Re-export MIN_P so consensus.py can import it from a single sibling module.
__all__ = ["MIN_P", "TOP_K", "batch_get_next_token_distributions",
           "_parse_completion_top_logprobs", "_parse_token_id_string"]
