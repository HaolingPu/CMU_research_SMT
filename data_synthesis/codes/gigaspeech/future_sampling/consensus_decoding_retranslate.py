#!/usr/bin/env python3
"""Consensus-gated fluency polish (direction 3, variant c).

Consensus runs *exactly* as the baseline and is the sole source of content +
timing. A second "polish" track then rewrites consensus's already-committed
text into fluent {target_lang}, append-only. Crucially the polisher sees ONLY
consensus's committed draft (never the source), so it physically cannot run
ahead of consensus's safe boundary or anticipate un-revealed content — it can
only smooth the surface form of what consensus already agreed on.

This is the conservative answer to the earlier free-re-translation experiment,
which anticipated (e.g. committed "双重性。" after seeing only "...both") and
lost BLEU. Here the polisher is bounded by the draft.

Two tracks are kept in the verbose log / output:
  - `consensus`: vanilla token-consensus buffer (unchanged baseline, the DRAFT).
  - `polish`   : fluent rewrite of that draft (the actual training target).

Reuses everything from `consensus_decoding_token_id_level_instruct` (5-axis
future sampling, distributions, consensus, IO, main). Only `run_one_utterance`
is overridden; `main()` monkeypatches it and defers to the base module so the
CLI is identical to the baseline.
"""
import consensus_decoding_token_id_level_instruct as base
from typing import Any, Dict, List, Optional

# Cap per-step polish length. The polished suffix only needs to cover the new
# consensus content, so this stays small.
POLISH_MAX_TOKENS = 64


def build_polish_prompt(tokenizer: Any, draft_text: str, polished_prefix: str,
                        target_lang: str = "Chinese") -> str:
    """Append-only fluency rewrite of `draft_text`, bounded by it (no source)."""
    rules = (
        "[RULES]\n"
        "This is a MINIMAL surface edit, NOT a rewrite or translation.\n"
        "You may ONLY: reorder words that are already present, insert/fix grammatical "
        "particles (的/了/地/得/了/着 etc.), and delete duplicated tokens.\n"
        "You MUST NOT: add any new content word (noun/verb/adjective/adverb), add any "
        "clause or idea, add literary flourish, paraphrase, or insert sentence-final "
        "punctuation (。！？) that closes a thought.\n"
        "Preserve every content word of the [DRAFT]. The output must be no longer than "
        "the [DRAFT]. If the [DRAFT] is already fine, copy it verbatim."
    )
    if not str(polished_prefix or "").strip():
        content = (
            f"[TASK]\nLightly clean up the word order and particles of the [DRAFT] "
            f"({target_lang}), changing as little as possible.\n\n"
            f"[DRAFT]\n{draft_text}\n\n"
            f"{rules}\n"
            f"Output only the cleaned {target_lang} text."
        )
    else:
        content = (
            f"[TASK]\nLightly clean up the word order and particles of the [DRAFT] "
            f"({target_lang}), changing as little as possible.\n\n"
            f"[DRAFT]\n{draft_text}\n\n"
            f"[IMPORTANT]\nA cleaned prefix is already committed at the start of the "
            "assistant reply. Continue from that exact prefix and output ONLY the cleaned "
            "version of the [DRAFT] content not yet covered by that prefix.\n"
            "Do not repeat the committed prefix. If nothing remains, output nothing.\n\n"
            f"{rules}"
        )
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(polished_prefix or "").strip():
        prompt += polished_prefix
    return prompt


def _polish(args, tokenizer, draft_text: str, polished_prefix: str) -> str:
    if not str(draft_text or "").strip():
        return ""
    prompt = build_polish_prompt(tokenizer, draft_text, polished_prefix, target_lang=args.target_lang)
    payload = {
        "model": args.instruct_api_model,
        "prompt": prompt,
        "max_tokens": POLISH_MAX_TOKENS,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = base._http_json(f"{base.normalize_api_base(args.instruct_api_base)}/completions",
                           payload=payload, timeout=args.instruct_api_timeout)
    choices = data.get("choices", [])
    return base.clean_model_text(str(choices[0].get("text", ""))) if choices else ""


def _force_complete(args, tokenizer, revealed_source: str, committed_prefix: str) -> str:
    """Consensus-track last-chunk completion (sees full source; baseline behaviour)."""
    return base.force_complete_translation(
        tokenizer=tokenizer,
        full_source=revealed_source,
        committed_text=committed_prefix,
        api_base=args.instruct_api_base,
        api_model=args.instruct_api_model,
        api_timeout=args.instruct_api_timeout,
        target_lang=args.target_lang,
        max_tokens=args.final_max_tokens,
    )


def run_one_utterance(
    row: Dict[str, Any],
    args,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    sampler_tokenizer: Any = None,
    sampler2_tokenizer: Any = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = base.parse_trajectory(row["src_trajectory"])
    source_units = base.parse_source_units(row.get("src_text_full"))
    full_source_text = base.get_full_source_text(row)

    # consensus track (baseline, == the DRAFT) -------------------------------
    committed_text = ""
    committed_token_ids: List[int] = []
    cons_deltas: List[str] = []
    cons_actions: List[str] = []
    # polish track (actual output) -------------------------------------------
    polished_text = ""
    pol_deltas: List[str] = []
    pol_actions: List[str] = []

    base._vlog(verbose_log_file, "#" * 60)
    base._vlog(verbose_log_file, f"# [polish] utt_id: {utt_id}")
    base._vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    base._vlog_pretty_value(verbose_log_file, "src_trajectory", chunks)
    base._vlog(verbose_log_file, f"# Chunks: {len(chunks)}  num_futures={args.num_futures} "
                                 f"top_k={args.candidate_top_k} min_p={args.min_p}")
    base._vlog(verbose_log_file, "#" * 60)

    for t in range(len(chunks)):
        current_source_chunk = str(chunks[t] or "")
        source_observed_full = base.build_source_observed(chunks, t)
        source_observed = base.build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=args.future_source_window_chunks,
        )
        base._vlog(verbose_log_file, f"\n{'='*60}")
        base._vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}  source_chunk={current_source_chunk!r}")
        base._vlog(verbose_log_file, f"revealed_source: {source_observed_full!r}")
        base._vlog(verbose_log_file, f"cons_draft_before:    {committed_text!r}")
        base._vlog(verbose_log_file, f"polished_before:      {polished_text!r}")

        # ── consensus track: exactly baseline ──
        if t == len(chunks) - 1:
            cons_delta = _force_complete(args, instruct_tokenizer, source_observed_full, committed_text)
            committed_text += cons_delta
            cons_action = "WRITE" if cons_delta else "READ"
        else:
            if getattr(args, "use_targeted_instruct_sampling", False):
                sampler_api_base = args.targeted_sampler_api_base or args.instruct_api_base
                sampler_api_model = args.targeted_sampler_api_model or args.instruct_api_model
                sampler_api_timeout = (
                    args.targeted_sampler_api_timeout
                    if args.targeted_sampler_api_timeout and args.targeted_sampler_api_timeout > 0
                    else args.instruct_api_timeout
                )
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
                    base_specs=base_specs,
                    observed_source=source_observed,
                    future_tokens=args.future_tokens,
                    sample_temperature=args.sample_temperature,
                )
            base._vlog(verbose_log_file, f"[futures] total={len(futures)}")

            if len(futures) <= 3:
                cons_delta = ""
                cons_action = "READ"
                base._vlog(verbose_log_file, "  consensus -> READ (too few futures)")
            else:
                pending_token_ids, _ = base.extend_pending_tokens(
                    instruct_tokenizer=instruct_tokenizer,
                    source_observed=source_observed_full,
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
                    soft_vote_top_k=args.soft_vote_top_k,
                    soft_vote_min_p=args.soft_vote_min_p,
                    soft_vote_threshold=args.soft_vote_threshold,
                    min_voters_ratio=args.min_voters_ratio,
                )
                if (args.min_consensus_horizon > 1
                        and 0 < len(pending_token_ids) < args.min_consensus_horizon):
                    pending_token_ids = []
                committed_text, cons_delta, cons_delta_ids, _ = base.finalize_external_commit(
                    tokenizer=instruct_tokenizer,
                    committed_text=committed_text,
                    pending_token_ids=pending_token_ids,
                )
                committed_token_ids.extend(cons_delta_ids)
                cons_action = "WRITE" if cons_delta else "READ"

        cons_deltas.append(cons_delta)
        cons_actions.append(cons_action)

        # ── polish track: rewrite the new consensus content, bounded by the draft ──
        if cons_action == "WRITE":
            pol_delta = _polish(args, instruct_tokenizer, committed_text, polished_text)
        else:
            pol_delta = ""
        polished_text += pol_delta
        pol_deltas.append(pol_delta)
        pol_actions.append("WRITE" if pol_delta else "READ")

        base._vlog(verbose_log_file, f"  [CMP] consensus {cons_action:5s} delta={cons_delta!r}")
        base._vlog(verbose_log_file, f"  [CMP] polish    {pol_actions[-1]:5s} delta={pol_delta!r}")
        base._vlog(verbose_log_file, f"  cons_draft_after:  {committed_text!r}")
        base._vlog(verbose_log_file, f"  polished_after:    {polished_text!r}")

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        # actual training target = polish track
        "target_trajectory": pol_deltas,
        "actions": pol_actions,
        "prediction": polished_text,
        # baseline consensus draft kept for comparison (do NOT discard)
        "consensus_target_trajectory": cons_deltas,
        "consensus_actions": cons_actions,
        "consensus_prediction": committed_text,
        "decoder_impl": {"method": "consensus_gated_polish", "backend": "vllm_completion"},
    }

    reference_text = base._extract_reference_text_from_row(row, target_lang=args.target_lang)
    result["reference_text"] = reference_text or ""
    result["metrics"] = _metrics(polished_text, pol_deltas, pol_actions, chunks,
                                 reference_text, full_source_text)
    result["consensus_metrics"] = _metrics(committed_text, cons_deltas, cons_actions, chunks,
                                            reference_text, full_source_text)

    m, cm = result["metrics"], result["consensus_metrics"]
    base._vlog(verbose_log_file, f"\n[RESULT] polish    bleu={m['bleu_char']:.2f} laal={m['laal_text']:.2f} "
               f"len_ratio_ref={m['length_ratio_ref']:.2f}  pred={polished_text!r}")
    base._vlog(verbose_log_file, f"[RESULT] consensus bleu={cm['bleu_char']:.2f} laal={cm['laal_text']:.2f} "
               f"len_ratio_ref={cm['length_ratio_ref']:.2f}  pred={committed_text!r}")
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


def main() -> None:
    base.run_one_utterance = run_one_utterance
    base.main()


if __name__ == "__main__":
    main()
