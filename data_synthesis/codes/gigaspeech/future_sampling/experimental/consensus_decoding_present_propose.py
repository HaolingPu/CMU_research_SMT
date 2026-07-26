#!/usr/bin/env python3
"""Present-proposes / futures-verify gate (register fix #3, ref-free).

Verdict chain (wiki: 2026-07-present-rank-winner): with the flagship's strict
intersection-of-future-top-6 gate, re-ranking the survivors by the present
distribution moved nothing (-2.15 paired char-BLEU, register markers unmoved)
because the intersection typically leaves 1-2 tokens. The register bias enters
at the GATE: every candidate set is conditioned on an appended sampled future,
which shifts the distributions toward written register, and the gate is the
intersection of exactly those shifted sets.

This variant inverts the roles:
- The PRESENT distribution (probe on observed source only, no future) PROPOSES
  the candidate set: its top-K (same K=6 as the flagship gate).
- The futures only VERIFY: a proposal survives if it appears in the top-K of at
  least ceil(min_voters_ratio * num_futures) future-conditioned distributions
  (default 0.75 -> 15/20). This keeps the futures' real job - vetoing
  present-myopia and premature commits - without letting them author the
  wording.
- Winner = argmax present probability among survivors (canonical register).
- No survivor -> no consensus -> READ, same stopping semantics as flagship.

Differs from the failed soft-vote loosening (which relaxed the SAME
futures-proposed candidate universe): here the proposal universe itself moves
to the canonical present manifold. Fully ref-free. CLI identical to
consensus_decoding_token_id_level_instruct.py; --min-voters-ratio is honored,
other soft-vote args ignored.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import consensus_decoding_token_id_level_instruct as base
from typing import Any, Dict, List, Tuple

# Depth of every distribution (present + futures). Must not exceed the server
# --max-logprobs (100 in serve_instruct_gpu0.sh).
PRESENT_MIN_LOGPROBS = 50


def extend_pending_tokens_present_propose(
    instruct_tokenizer: Any,
    source_observed: str,
    futures: List[str],
    committed_text: str,
    committed_token_ids: List[int],
    max_consensus_steps: int,
    candidate_top_k: int = base.TOP_K,
    instruct_api_base: str = "",
    instruct_api_model: str = "",
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
    top_p: float = 0.0,
    target_lang: str = "Chinese",
    soft_vote_top_k: int = 20,
    soft_vote_min_p: float = 0.1,
    soft_vote_threshold: float = 0.8,
    min_voters_ratio: float = 0.75,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    del soft_vote_top_k, soft_vote_min_p, soft_vote_threshold
    pending_token_ids: List[int] = []
    grow_logs: List[Dict[str, Any]] = []

    for step_idx in range(max_consensus_steps):
        target_prefix_token_ids = list(committed_token_ids) + list(pending_token_ids)
        # Prompt 0 is the present (no-future) probe; the rest are the usual
        # future-conditioned probes.
        full_sources = [source_observed] + [
            base.append_text_continuation(source_observed, f) for f in futures
        ]

        batch_results = base.batch_get_next_token_distributions(
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
            min_logprobs=PRESENT_MIN_LOGPROBS,
        )
        if not batch_results:
            grow_logs.append({"step": step_idx, "stop": "empty_batch"})
            return pending_token_ids, grow_logs

        present_dist, present_debug = batch_results[0]
        future_results = batch_results[1:]
        if not present_dist:
            grow_logs.append({"step": step_idx, "stop": "empty_present_distribution",
                              "dist_debug": present_debug})
            return pending_token_ids, grow_logs

        future_candidate_sets: List[set] = []
        distributions: List[Dict[int, float]] = []
        per_future: List[Dict[str, Any]] = []
        for i, (dist, dist_debug) in enumerate(future_results):
            if not dist:
                grow_logs.append({"step": step_idx, "stop": "empty_distribution",
                                  "future": futures[i], "dist_debug": dist_debug})
                return pending_token_ids, grow_logs
            distributions.append(dist)
            candidate_ids = base._select_candidates(dist, top_p=top_p, min_p=min_p,
                                                    top_k=candidate_top_k)
            future_candidate_sets.append(set(candidate_ids))
            per_future.append({
                "future": futures[i],
                "candidate_texts": [base._single_token_text(instruct_tokenizer, t)
                                    for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        # ── present proposes ──
        proposals = base._select_candidates(present_dist, top_p=top_p, min_p=min_p,
                                            top_k=candidate_top_k)
        # ── futures verify by majority support ──
        n_fut = len(distributions)
        need_votes = max(1, math.ceil(min_voters_ratio * n_fut))
        support = {tok: sum(1 for s in future_candidate_sets if tok in s)
                   for tok in proposals}
        survivors = [tok for tok in proposals if support[tok] >= need_votes]
        if not survivors:
            grow_logs.append({"step": step_idx, "stop": "no_consensus_token",
                              "per_future": per_future,
                              "meta": {"reason": "no_majority_support",
                                       "proposals": proposals,
                                       "support": {str(t): support[t] for t in proposals},
                                       "need_votes": need_votes}})
            break

        consensus_token_id = max(survivors,
                                 key=lambda tok: present_dist.get(tok, 0.0))

        pending_token_ids.append(consensus_token_id)
        view = base.inspect_token_ids(instruct_tokenizer, pending_token_ids)
        meta = {
            "reason": "ok",
            "winner_rule": "present_propose",
            "proposals": proposals,
            "survivors": survivors,
            "support": support[consensus_token_id],
            "need_votes": need_votes,
            "present_prob": present_dist.get(consensus_token_id, 0.0),
            "avg_future_prob": sum(d.get(consensus_token_id, 0.0)
                                   for d in distributions) / n_fut,
        }
        grow_logs.append({
            "step": step_idx,
            "accepted_token_id": consensus_token_id,
            "accepted_token_text": view["last_token_text"],
            "pending_text": view["decoded_text"],
            "llm_prefix": base.decode_token_ids_to_text(instruct_tokenizer, target_prefix_token_ids),
            "llm_prefix_token_ids": target_prefix_token_ids,
            "per_future": per_future,
            "meta": meta,
        })

    return pending_token_ids, grow_logs


base.extend_pending_tokens = extend_pending_tokens_present_propose
print("patched: extend_pending_tokens_present_propose", file=sys.stderr)

if __name__ == "__main__":
    base.main()
