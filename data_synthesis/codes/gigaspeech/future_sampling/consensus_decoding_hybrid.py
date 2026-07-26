#!/usr/bin/env python3
"""Hybrid consensus decoding: consensus timing + anchor wording (2026-07).

Follow-up to consensus_decoding_anchor.py after the anchor-and-veto trained
checkpoint failed (wiki: 2026-07-anchor-smoke500-sweep, trained-model verdict):
removing the vote as the commit gate destroyed the trained model's silence
behavior (onomatopoeia loops on non-speech audio), while the anchor's WORDING
win (+7 paired char-BLEU, register normalized) was validated.

This decoder splits the roles cleanly:
  - TIMING: the vanilla consensus track runs exactly as the flagship
    (top5-axis5) decoder, on its own committed prefix. Whatever number of
    tokens it commits in a chunk — including zero — is the commit budget k.
  - WORDING: when k > 0, greedy-generate the anchor continuation (observed
    source + hybrid committed prefix, hibiki translator prompt) and commit its
    first k tokens. Futures never choose wording; the vote never sees the
    anchor text (the two prefixes evolve independently), so the timing track
    is bit-identical to the flagship baseline.

No veto scoring is needed — the hybrid track costs one greedy anchor call per
WRITE chunk on top of the baseline decode.

Dual-track output like the anchor decoder: `target_trajectory`/`prediction` =
hybrid track; `consensus_*` = vanilla baseline (paired comparison).

New CLI args (all others identical to the baseline):
  --anchor-max-tokens 24   max anchor tokens generated per chunk
"""
import argparse
import sys
from typing import Any, Dict, List, Optional

import consensus_decoding_token_id_level_instruct as base
import consensus_decoding_anchor as anchor


def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    sampler_tokenizer: Any = None,
    sampler2_tokenizer: Any = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    anchor._sample_futures.base_specs = base_specs
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = base.parse_trajectory(row["src_trajectory"])
    source_units = base.parse_source_units(row.get("src_text_full"))
    full_source_text = base.get_full_source_text(row)

    # hybrid track (the actual output): anchor words at consensus commit times
    h_text = ""
    h_ids: List[int] = []
    h_deltas: List[str] = []
    h_actions: List[str] = []
    hybrid_debug: List[Dict[str, Any]] = []
    # consensus track (vanilla baseline = the timing authority)
    c_text = ""
    c_ids: List[int] = []
    c_deltas: List[str] = []
    c_actions: List[str] = []

    base._vlog(verbose_log_file, "#" * 60)
    base._vlog(verbose_log_file, f"# [hybrid] utt_id: {utt_id}")
    base._vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    base._vlog(verbose_log_file, f"# Chunks: {len(chunks)}  anchor_max_tokens={args.anchor_max_tokens}")
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
        base._vlog(verbose_log_file, f"hybrid_before:    {h_text!r}")
        base._vlog(verbose_log_file, f"consensus_before: {c_text!r}")

        # ── last chunk: both tracks force-complete (full source visible) ──
        if t == len(chunks) - 1:
            h_delta = anchor.anchor_force_complete(args, instruct_tokenizer,
                                                   source_observed_full, h_text)
            h_text += h_delta
            h_deltas.append(h_delta)
            h_actions.append("WRITE" if h_delta else "READ")
            c_delta = base.force_complete_translation(
                tokenizer=instruct_tokenizer, full_source=source_observed_full,
                committed_text=c_text, api_base=args.instruct_api_base,
                api_model=args.instruct_api_model, api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang, max_tokens=args.final_max_tokens)
            c_text += c_delta
            c_deltas.append(c_delta)
            c_actions.append("WRITE" if c_delta else "READ")
            base._vlog(verbose_log_file, f"  [Final] hybrid={h_delta!r}")
            base._vlog(verbose_log_file, f"  [Final] consensus={c_delta!r}")
            continue

        futures = anchor._sample_futures(args, sampler_tokenizer, sampler2_tokenizer,
                                         source_observed, c_text)
        base._vlog(verbose_log_file, f"[futures] total={len(futures)}")
        if len(futures) <= 3:
            h_deltas.append(""); h_actions.append("READ")
            c_deltas.append(""); c_actions.append("READ")
            base._vlog(verbose_log_file, "  -> READ (too few futures, both tracks)")
            continue

        # ── consensus track (exactly the baseline; sets the commit budget) ──
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

        # ── hybrid track: commit k anchor tokens, k = consensus commit size ──
        k = len(c_delta_ids)
        chunk_dbg: Dict[str, Any] = {"chunk": t, "budget": k}
        if k == 0:
            h_delta = ""
        else:
            raw_anchor = anchor.generate_anchor_token_ids(args, instruct_tokenizer,
                                                          source_observed_full, h_ids)
            anchor_ids = anchor.anchor_truncate_disallowed(instruct_tokenizer, raw_anchor)
            chunk_dbg["anchor_len"] = len(anchor_ids)
            commit_ids = anchor.anchor_trim_to_boundary(instruct_tokenizer, anchor_ids[:k])
            chunk_dbg["commit_len"] = len(commit_ids)
            h_delta = base.decode_token_ids_to_text(instruct_tokenizer, commit_ids)
            if h_delta:
                h_ids.extend(commit_ids)
                h_text += h_delta
        h_deltas.append(h_delta)
        h_actions.append("WRITE" if h_delta else "READ")
        hybrid_debug.append(chunk_dbg)

        base._vlog(verbose_log_file, f"  [CMP] hybrid    {h_actions[-1]:5s} delta={h_delta!r} "
                                     f"(budget {k}, committed {chunk_dbg.get('commit_len', 0)})")
        base._vlog(verbose_log_file, f"  [CMP] consensus {c_actions[-1]:5s} delta={c_delta!r}")

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        # actual training target = hybrid track
        "target_trajectory": h_deltas,
        "actions": h_actions,
        "prediction": h_text,
        # vanilla consensus baseline on the same futures (paired comparison)
        "consensus_target_trajectory": c_deltas,
        "consensus_actions": c_actions,
        "consensus_prediction": c_text,
        "anchor_debug": hybrid_debug,
        "decoder_impl": {"method": "hybrid_consensus_timing_anchor_wording",
                         "backend": "vllm_completion"},
    }

    reference_text = base._extract_reference_text_from_row(row, target_lang=args.target_lang)
    result["reference_text"] = reference_text or ""
    result["metrics"] = anchor._metrics(h_text, h_deltas, h_actions, chunks,
                                        reference_text, full_source_text)
    result["consensus_metrics"] = anchor._metrics(c_text, c_deltas, c_actions, chunks,
                                                  reference_text, full_source_text)

    m, cm = result["metrics"], result["consensus_metrics"]
    base._vlog(verbose_log_file, f"\n[RESULT] hybrid    bleu={m['bleu_char']:.2f} laal={m['laal_text']:.2f} "
               f"len_ratio_ref={m['length_ratio_ref']:.2f}  pred={h_text!r}")
    base._vlog(verbose_log_file, f"[RESULT] consensus bleu={cm['bleu_char']:.2f} laal={cm['laal_text']:.2f} "
               f"len_ratio_ref={cm['length_ratio_ref']:.2f}  pred={c_text!r}")
    return result


_base_parse_args = base.parse_args


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--anchor-max-tokens", type=int, default=24)
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
