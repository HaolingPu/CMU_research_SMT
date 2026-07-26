#!/usr/bin/env python3
"""Anchor-and-veto consensus decoding (register fix, 2026-07).

Inverts the roles of the baseline consensus decoder: the WORDING comes from a
single greedy "anchor" continuation conditioned only on the observed source
(same model + same sub-sentence translator prompt that generated the hibiki
targets, so the register is canonical by construction), and the sampled
futures are demoted from authors to safety inspectors — they only VETO how far
the anchor may commit (the READ/WRITE timing).

Per chunk:
  1. anchor  = greedy continuation of (observed source, committed prefix)
               under the professional-translator prompt.
  2. score   = teacher-force the anchor tokens under each future
               (one batched /completions call with prompt_logprobs).
  3. commit  = longest anchor prefix whose every token passes the veto gate
               (p >= veto-min-p OR rank <= veto-top-k, under at least
               ceil(N * veto-min-voters-ratio) futures). First failure -> READ.

Motivation (see wiki: 2026-07-consensus-register-forensics): the baseline's
intersect/vote rule makes the futures CHOOSE the token, which systematically
elects future-proof written-register forms (所以→因此, 和→与, 把→将 ...) and
drifts the prefix off the model's greedy manifold; the anchor only has to be
*acceptable* under each future, not *top-ranked* under all of them.

Dual-track like consensus_decoding_retranslate.py: the vanilla consensus
baseline runs alongside on the SAME sampled futures, so every utterance is a
paired comparison. Output: `target_trajectory`/`prediction` = anchor track;
`consensus_*` = baseline track.

Reuses IO/futures/commit-trim/main from consensus_decoding_token_id_level_instruct.
New CLI args (all others identical to the baseline):
  --anchor-max-tokens 24      max anchor tokens generated per chunk
  --veto-min-p 0.05           per-future probability floor for an anchor token
  --veto-top-k 5              ...or the token is within the future's top-k
  --veto-min-voters-ratio 1.0 fraction of futures that must accept (1.0 = all)
"""
import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import consensus_decoding_token_id_level_instruct as base


# ---------------------------------------------------------------------------
# Anchor prompt — verbatim the sub-sentence translator setup that produced the
# hibiki targets (translate_subsentences.py), so the anchor draws from the
# same distribution those targets came from.
# ---------------------------------------------------------------------------

def build_anchor_prompt_prefix_token_ids(tokenizer: Any, source_text: str,
                                         target_lang: str = "Chinese") -> List[int]:
    messages = [
        {"role": "system", "content": (
            f"You are a professional translator. Translate the English source into "
            f"{target_lang}. Output only the {target_lang} translation, nothing else."
        )},
        {"role": "user", "content": f"Translate into {target_lang}:\n{source_text}"},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    return list(prompt_ids) + list(assistant_prefix_ids)


def _parse_generated_token_ids(choice: Dict[str, Any]) -> List[int]:
    """Token IDs of the generated text, robust to vllm response variants."""
    tids = choice.get("token_ids")
    if isinstance(tids, list) and tids:
        return [int(t) for t in tids]
    logprobs = choice.get("logprobs") or {}
    out: List[int] = []
    for tok in logprobs.get("tokens") or []:
        s = str(tok)
        if s.startswith("token_id:"):
            out.append(int(s.split(":", 1)[1]))
        else:
            return []  # fell back to text tokens; caller re-encodes
    return out


def generate_anchor_token_ids(
    args: argparse.Namespace,
    tokenizer: Any,
    source_observed_full: str,
    committed_token_ids: List[int],
) -> List[int]:
    prompt_ids = (build_anchor_prompt_prefix_token_ids(tokenizer, source_observed_full,
                                                       target_lang=args.target_lang)
                  + list(committed_token_ids))
    payload = {
        "model": args.instruct_api_model,
        "prompt": [prompt_ids],
        "max_tokens": max(1, int(args.anchor_max_tokens)),
        "temperature": 0.0,
        "logprobs": 0,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = base._http_json(f"{base.normalize_api_base(args.instruct_api_base)}/completions",
                           payload=payload, timeout=args.instruct_api_timeout)
    choices = data.get("choices", [])
    if not choices:
        return []
    ids = _parse_generated_token_ids(choices[0])
    if not ids:
        text = base.clean_model_text(str(choices[0].get("text", "")))
        if text:
            ids = tokenizer.encode(text, add_special_tokens=False)
    return ids


# ---------------------------------------------------------------------------
# Veto: teacher-force the anchor under every future
# ---------------------------------------------------------------------------

def score_anchor_under_futures(
    args: argparse.Namespace,
    tokenizer: Any,
    source_observed_full: str,
    futures: List[str],
    committed_token_ids: List[int],
    anchor_token_ids: List[int],
) -> List[List[Dict[str, float]]]:
    """Returns per_future[i] = list over anchor positions of {prob, rank}."""
    prompts = []
    for fut in futures:
        full_source = base.append_text_continuation(source_observed_full, fut)
        prompts.append(build_anchor_prompt_prefix_token_ids(tokenizer, full_source,
                                                            target_lang=args.target_lang)
                       + list(committed_token_ids) + list(anchor_token_ids))
    payload = {
        "model": args.instruct_api_model,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    }
    data = base._http_json(f"{base.normalize_api_base(args.instruct_api_base)}/completions",
                           payload=payload, timeout=args.instruct_api_timeout)
    choices = data.get("choices", [])
    n_anchor = len(anchor_token_ids)
    results: List[List[Dict[str, float]]] = []
    for i in range(len(prompts)):
        if i >= len(choices):
            results.append([{"prob": 0.0, "rank": float("inf")}] * n_anchor)
            continue
        plp = choices[i].get("prompt_logprobs")
        if not isinstance(plp, list) or len(plp) < n_anchor:
            raise RuntimeError(
                "vLLM response missing prompt_logprobs — server must support the "
                "'prompt_logprobs' completions extension (got keys: "
                f"{sorted(choices[i].keys())})")
        tail = plp[-n_anchor:]
        scores: List[Dict[str, float]] = []
        for pos, tok_id in enumerate(anchor_token_ids):
            entry = tail[pos] or {}
            info = entry.get(str(tok_id))
            if info is None and entry:
                # key may be int in some vllm versions
                info = entry.get(tok_id)
            if info is None:
                scores.append({"prob": 0.0, "rank": float("inf")})
            else:
                scores.append({"prob": math.exp(float(info.get("logprob", -100.0))),
                               "rank": float(info.get("rank", float("inf")))})
        results.append(scores)
    return results


def veto_commit_length(
    per_future_scores: List[List[Dict[str, float]]],
    veto_min_p: float,
    veto_top_k: int,
    veto_min_voters_ratio: float,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Longest anchor prefix where every position is accepted by enough futures."""
    if not per_future_scores:
        return 0, []
    num_futures = len(per_future_scores)
    n_pos = len(per_future_scores[0])
    min_voters = max(1, math.ceil(num_futures * veto_min_voters_ratio))
    pos_logs: List[Dict[str, Any]] = []
    commit_len = 0
    for pos in range(n_pos):
        accept = 0
        min_prob, worst_rank = 1.0, 0.0
        for f in range(num_futures):
            s = per_future_scores[f][pos]
            ok = (s["prob"] >= veto_min_p) or (s["rank"] <= veto_top_k)
            accept += int(ok)
            min_prob = min(min_prob, s["prob"])
            worst_rank = max(worst_rank, s["rank"])
        passed = accept >= min_voters
        pos_logs.append({"pos": pos, "accept": accept, "of": num_futures,
                         "min_prob": round(min_prob, 6),
                         "worst_rank": (None if worst_rank == float("inf") else int(worst_rank)),
                         "passed": passed})
        if not passed:
            break
        commit_len = pos + 1
    return commit_len, pos_logs


# ---------------------------------------------------------------------------
# Anchor-track token hygiene: unlike the baseline we ALLOW ascii letters
# (references legitimately contain "T.G.", "DNA", ...). We truncate at the
# first genuinely disallowed token instead of splicing it out mid-sequence.
# ---------------------------------------------------------------------------

def anchor_truncate_disallowed(tokenizer: Any, token_ids: List[int]) -> List[int]:
    out: List[int] = []
    for tok_id in token_ids:
        reason = base._disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None and reason != "ascii_letters":
            break
        out.append(tok_id)
    return out


def anchor_trim_to_boundary(tokenizer: Any, token_ids: List[int]) -> List[int]:
    """Longest prefix that decodes without broken byte-BPE (U+FFFD etc.)."""
    last_ok = -1
    for i in range(len(token_ids)):
        if not base.has_suspicious_content(
                base.decode_token_ids_to_text(tokenizer, token_ids[:i + 1])):
            last_ok = i
    return token_ids[:last_ok + 1]


def anchor_force_complete(args: argparse.Namespace, tokenizer: Any,
                          full_source: str, committed_text: str) -> str:
    messages = [
        {"role": "system", "content": (
            f"You are a professional translator. Translate the English source into "
            f"{args.target_lang}. Output only the {args.target_lang} translation, nothing else."
        )},
        {"role": "user", "content": f"Translate into {args.target_lang}:\n{full_source}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(committed_text or "").strip():
        prompt += committed_text
    payload = {
        "model": args.instruct_api_model,
        "prompt": prompt,
        "max_tokens": max(1, int(args.final_max_tokens)),
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = base._http_json(f"{base.normalize_api_base(args.instruct_api_base)}/completions",
                           payload=payload, timeout=args.instruct_api_timeout)
    choices = data.get("choices", [])
    return base.clean_model_text(str(choices[0].get("text", ""))) if choices else ""


# ---------------------------------------------------------------------------
# Per-utterance dual-track loop
# ---------------------------------------------------------------------------

def _sample_futures(args, sampler_tokenizer, sampler2_tokenizer,
                    source_observed: str, committed_text: str) -> List[str]:
    if getattr(args, "use_targeted_instruct_sampling", False):
        sampler_api_base = args.targeted_sampler_api_base or args.instruct_api_base
        sampler_api_model = args.targeted_sampler_api_model or args.instruct_api_model
        sampler_api_timeout = (args.targeted_sampler_api_timeout
                               if args.targeted_sampler_api_timeout and args.targeted_sampler_api_timeout > 0
                               else args.instruct_api_timeout)
        futures, _ = base.sample_source_futures_targeted_prefill(
            sampler_tokenizer=sampler_tokenizer,
            observed_source=source_observed,
            committed_text=committed_text,
            target_lang=args.target_lang,
            num_futures=args.targeted_num_futures,
            api_base=sampler_api_base,
            api_model=sampler_api_model,
            api_timeout=sampler_api_timeout,
            sample_temperature=args.targeted_sample_temperature,
            top_p=args.targeted_top_p,
            max_tokens=args.targeted_max_tokens,
            sampler2_tokenizer=sampler2_tokenizer,
            sampler2_api_base=args.targeted_sampler2_api_base,
            sampler2_api_model=args.targeted_sampler2_api_model,
            sampler2_api_timeout=args.targeted_sampler2_api_timeout,
        )
    else:
        futures, _ = base.sample_source_futures_multi(
            base_specs=_sample_futures.base_specs,
            observed_source=source_observed,
            future_tokens=args.future_tokens,
            sample_temperature=args.sample_temperature,
        )
    return futures


def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    sampler_tokenizer: Any = None,
    sampler2_tokenizer: Any = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    _sample_futures.base_specs = base_specs
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = base.parse_trajectory(row["src_trajectory"])
    source_units = base.parse_source_units(row.get("src_text_full"))
    full_source_text = base.get_full_source_text(row)

    # anchor track (the actual output)
    a_text = ""
    a_ids: List[int] = []
    a_deltas: List[str] = []
    a_actions: List[str] = []
    anchor_debug: List[Dict[str, Any]] = []
    # consensus track (vanilla baseline, for the paired comparison)
    c_text = ""
    c_ids: List[int] = []
    c_deltas: List[str] = []
    c_actions: List[str] = []

    base._vlog(verbose_log_file, "#" * 60)
    base._vlog(verbose_log_file, f"# [anchor_veto] utt_id: {utt_id}")
    base._vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    base._vlog(verbose_log_file, f"# Chunks: {len(chunks)}  veto_min_p={args.veto_min_p} "
                                 f"veto_top_k={args.veto_top_k} ratio={args.veto_min_voters_ratio} "
                                 f"anchor_max_tokens={args.anchor_max_tokens}")
    base._vlog(verbose_log_file, "#" * 60)

    for t in range(len(chunks)):
        source_observed_full = base.build_source_observed(chunks, t)
        source_observed = base.build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=args.future_source_window_chunks,
        )
        base._vlog(verbose_log_file, f"\n{'='*60}")
        base._vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}  chunk={chunks[t]!r}")
        base._vlog(verbose_log_file, f"anchor_before:    {a_text!r}")
        base._vlog(verbose_log_file, f"consensus_before: {c_text!r}")

        # ── last chunk: both tracks force-complete (full source visible) ──
        if t == len(chunks) - 1:
            a_delta = anchor_force_complete(args, instruct_tokenizer, source_observed_full, a_text)
            a_text += a_delta
            a_deltas.append(a_delta)
            a_actions.append("WRITE" if a_delta else "READ")
            c_delta = base.force_complete_translation(
                tokenizer=instruct_tokenizer, full_source=source_observed_full,
                committed_text=c_text, api_base=args.instruct_api_base,
                api_model=args.instruct_api_model, api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang, max_tokens=args.final_max_tokens)
            c_text += c_delta
            c_deltas.append(c_delta)
            c_actions.append("WRITE" if c_delta else "READ")
            base._vlog(verbose_log_file, f"  [Final] anchor={a_delta!r}")
            base._vlog(verbose_log_file, f"  [Final] consensus={c_delta!r}")
            continue

        futures = _sample_futures(args, sampler_tokenizer, sampler2_tokenizer,
                                  source_observed, c_text)
        base._vlog(verbose_log_file, f"[futures] total={len(futures)}")
        if len(futures) <= 3:
            a_deltas.append(""); a_actions.append("READ")
            c_deltas.append(""); c_actions.append("READ")
            base._vlog(verbose_log_file, "  -> READ (too few futures, both tracks)")
            continue

        # ── consensus track (exactly the baseline) ──
        pending_ids, _ = base.extend_pending_tokens(
            instruct_tokenizer=instruct_tokenizer,
            source_observed=source_observed_full,
            futures=futures,
            committed_text=c_text,
            committed_token_ids=c_ids,
            max_consensus_steps=args.max_consensus_steps,
            candidate_top_k=args.candidate_top_k,
            instruct_api_base=args.instruct_api_base,
            instruct_api_model=args.instruct_api_model,
            instruct_api_timeout=args.instruct_api_timeout,
            min_p=args.min_p,
            top_p=args.top_p,
            target_lang=args.target_lang,
            soft_vote_top_k=args.soft_vote_top_k,
            soft_vote_min_p=args.soft_vote_min_p,
            soft_vote_threshold=args.soft_vote_threshold,
            min_voters_ratio=args.min_voters_ratio,
        )
        if args.min_consensus_horizon > 1 and 0 < len(pending_ids) < args.min_consensus_horizon:
            pending_ids = []
        c_text, c_delta, c_delta_ids, _ = base.finalize_external_commit(
            tokenizer=instruct_tokenizer, committed_text=c_text, pending_token_ids=pending_ids)
        c_ids.extend(c_delta_ids)
        c_deltas.append(c_delta)
        c_actions.append("WRITE" if c_delta else "READ")

        # ── anchor track ──
        raw_anchor = generate_anchor_token_ids(args, instruct_tokenizer,
                                               source_observed_full, a_ids)
        anchor_ids = anchor_truncate_disallowed(instruct_tokenizer, raw_anchor)
        chunk_dbg: Dict[str, Any] = {
            "chunk": t,
            "anchor_text": base.decode_token_ids_to_text(instruct_tokenizer, anchor_ids),
            "anchor_len": len(anchor_ids),
        }
        if not anchor_ids:
            a_delta = ""
            chunk_dbg["commit_len"] = 0
            chunk_dbg["stop"] = "empty_anchor"
        else:
            scores = score_anchor_under_futures(args, instruct_tokenizer,
                                                source_observed_full, futures,
                                                a_ids, anchor_ids)
            commit_len, pos_logs = veto_commit_length(
                scores, veto_min_p=args.veto_min_p, veto_top_k=args.veto_top_k,
                veto_min_voters_ratio=args.veto_min_voters_ratio)
            chunk_dbg["commit_len"] = commit_len
            if commit_len < len(pos_logs):
                blocked = pos_logs[commit_len]
                blocked_tok = anchor_ids[commit_len]
                chunk_dbg["veto"] = {
                    "token": base._single_token_text(instruct_tokenizer, blocked_tok),
                    **{k: blocked[k] for k in ("accept", "of", "min_prob", "worst_rank")},
                }
            commit_ids = anchor_ids[:commit_len]
            if args.min_consensus_horizon > 1 and 0 < len(commit_ids) < args.min_consensus_horizon:
                chunk_dbg["stop"] = "below_min_horizon"
                commit_ids = []
            commit_ids = anchor_trim_to_boundary(instruct_tokenizer, commit_ids)
            a_delta = base.decode_token_ids_to_text(instruct_tokenizer, commit_ids)
            if a_delta:
                a_ids.extend(commit_ids)
                a_text += a_delta
        a_deltas.append(a_delta)
        a_actions.append("WRITE" if a_delta else "READ")
        anchor_debug.append(chunk_dbg)

        base._vlog(verbose_log_file, f"  anchor_gen:  {chunk_dbg['anchor_text']!r}")
        base._vlog(verbose_log_file, f"  [CMP] anchor    {a_actions[-1]:5s} delta={a_delta!r} "
                                     f"(commit {chunk_dbg.get('commit_len', 0)}/{chunk_dbg['anchor_len']}"
                                     f"{' veto@' + repr(chunk_dbg['veto']['token']) if 'veto' in chunk_dbg else ''})")
        base._vlog(verbose_log_file, f"  [CMP] consensus {c_actions[-1]:5s} delta={c_delta!r}")

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        # actual training target = anchor track
        "target_trajectory": a_deltas,
        "actions": a_actions,
        "prediction": a_text,
        # vanilla consensus baseline on the same futures (paired comparison)
        "consensus_target_trajectory": c_deltas,
        "consensus_actions": c_actions,
        "consensus_prediction": c_text,
        "anchor_debug": anchor_debug,
        "decoder_impl": {"method": "anchor_veto",
                         "veto": {"min_p": args.veto_min_p, "top_k": args.veto_top_k,
                                  "min_voters_ratio": args.veto_min_voters_ratio},
                         "backend": "vllm_completion"},
    }

    reference_text = base._extract_reference_text_from_row(row, target_lang=args.target_lang)
    result["reference_text"] = reference_text or ""
    result["metrics"] = _metrics(a_text, a_deltas, a_actions, chunks, reference_text, full_source_text)
    result["consensus_metrics"] = _metrics(c_text, c_deltas, c_actions, chunks, reference_text, full_source_text)

    m, cm = result["metrics"], result["consensus_metrics"]
    base._vlog(verbose_log_file, f"\n[RESULT] anchor    bleu={m['bleu_char']:.2f} laal={m['laal_text']:.2f} "
               f"len_ratio_ref={m['length_ratio_ref']:.2f}  pred={a_text!r}")
    base._vlog(verbose_log_file, f"[RESULT] consensus bleu={cm['bleu_char']:.2f} laal={cm['laal_text']:.2f} "
               f"len_ratio_ref={cm['length_ratio_ref']:.2f}  pred={c_text!r}")
    return result


def _metrics(pred, deltas, actions, chunks, reference_text, full_source_text) -> Dict[str, Any]:
    laal = bleu = lr_ref = float("nan")
    try:
        if reference_text:
            laal = base.compute_laal(chunks, deltas, actions, reference_text)
            bleu = base.compute_bleu_char(pred, reference_text)
            lr_ref = base.compute_length_ratio_ref(pred, reference_text)
    except Exception:
        pass
    return {
        "laal_text": laal,
        "bleu_char": bleu,
        "length_ratio_ref": lr_ref,
        "length_ratio_src": base.compute_length_ratio_src(pred, full_source_text),
        "pred_chars": base._nonspace_char_count(pred),
        "ref_chars": base._nonspace_char_count(reference_text or ""),
        "src_words": len(str(full_source_text or "").split()),
    }


# ---------------------------------------------------------------------------
# CLI plumbing: intercept our args, defer the rest to the baseline parser
# ---------------------------------------------------------------------------

_base_parse_args = base.parse_args


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--anchor-max-tokens", type=int, default=24)
    pre.add_argument("--veto-min-p", type=float, default=0.05)
    pre.add_argument("--veto-top-k", type=int, default=5)
    pre.add_argument("--veto-min-voters-ratio", type=float, default=1.0)
    mine, remaining = pre.parse_known_args()
    argv_backup = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        args = _base_parse_args()
    finally:
        sys.argv = argv_backup
    for k, v in vars(mine).items():
        setattr(args, k, v)
    return args


def main() -> None:
    base.parse_args = parse_args
    base.run_one_utterance = run_one_utterance
    base.main()


if __name__ == "__main__":
    main()
