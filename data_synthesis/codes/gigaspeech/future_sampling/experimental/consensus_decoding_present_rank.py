#!/usr/bin/env python3
"""Present-ranked consensus winner (register fix #2 from the forensics wiki page).

The flagship (J_40k / top5-axis5) commits, per step, the intersection of every
future's top-K candidates, then picks the winner by the FUTURE-AVERAGED
probability. The register forensics showed that future-averaging is the
register amplifier: formal connectives (因此/而/将) survive under more
continuations, so they dominate the winner rule and the trained model inherits
a written-register surface distribution ~40 BLEU from the canonical greedy one.

This variant keeps the flagship's strict intersection gate untouched (the
timing / safety policy is identical) and changes ONLY the winner rule: among
gate-eligible tokens, pick the argmax of the PRESENT distribution — the probe
conditioned on the observed source alone, with no future appended. The present
prompt rides in the same batched /completions call, so cost is +1 prompt per
consensus step. Fully ref-free.

Ties to the failed alternatives: anchor-and-veto changed the timing (degenerate
over-generation at train time); best-of-N ref selection changed nothing
systematically (no transfer). This changes the token-level register
systematically while provably keeping the flagship gate.

CLI is identical to consensus_decoding_token_id_level_instruct.py; the
soft-vote arguments are accepted but ignored (the gate here is always the
strict top-K intersection the flagship shipped with).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import consensus_decoding_token_id_level_instruct as base
from typing import Any, Dict, List, Tuple

# Depth of the present distribution. Must not exceed the server --max-logprobs
# (100 in serve_instruct_gpu0.sh); deep enough that gate-eligible tokens are
# almost always scored.
PRESENT_MIN_LOGPROBS = 50


def extend_pending_tokens_present_rank(
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
    del soft_vote_top_k, soft_vote_min_p, soft_vote_threshold, min_voters_ratio
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

        present_dist, _present_debug = batch_results[0]
        future_results = batch_results[1:]

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
            per_future.append({
                "future": futures[i],
                "candidate_texts": [base._single_token_text(instruct_tokenizer, t)
                                    for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        # ── flagship gate: strict intersection of every future's candidates ──
        candidate_lists = [base._select_candidates(d, top_p=top_p, min_p=min_p,
                                                   top_k=candidate_top_k)
                           for d in distributions]
        intersection = set(candidate_lists[0])
        for clist in candidate_lists[1:]:
            intersection &= set(clist)
        if not intersection:
            grow_logs.append({"step": step_idx, "stop": "no_consensus_token",
                              "per_future": per_future,
                              "meta": {"reason": "empty_intersection"}})
            break

        n_fut = len(distributions)
        baseline_token = max(
            intersection,
            key=lambda tok: sum(d.get(tok, 0.0) for d in distributions) / n_fut,
        )
        # ── present-rank winner: argmax of the no-future distribution over the
        # gate-eligible set; futures-mean is only the tie-breaker for tokens
        # the present probe did not score at all.
        consensus_token_id = max(
            intersection,
            key=lambda tok: (present_dist.get(tok, 0.0),
                             sum(d.get(tok, 0.0) for d in distributions)),
        )
        present_prob = present_dist.get(consensus_token_id, 0.0)
        used_fallback = present_prob <= 0.0
        if used_fallback:
            consensus_token_id = baseline_token

        pending_token_ids.append(consensus_token_id)
        view = base.inspect_token_ids(instruct_tokenizer, pending_token_ids)
        meta = {
            "reason": "ok",
            "winner_rule": "present_rank",
            "intersection": sorted(intersection),
            "baseline_token": baseline_token,
            "baseline_token_text": base._single_token_text(instruct_tokenizer, baseline_token),
            "present_prob": present_prob,
            "changed_vs_baseline": consensus_token_id != baseline_token,
            "present_fallback": used_fallback,
            "avg_score": sum(d.get(consensus_token_id, 0.0) for d in distributions) / n_fut,
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


base.extend_pending_tokens = extend_pending_tokens_present_rank

if __name__ == "__main__":
    base.main()
