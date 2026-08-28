#!/usr/bin/env python3
"""Verbose step-by-step replay of LA-N synthesis for a few utterances.

Runs the *real* MT model (qwen3-instruct via vLLM OpenAI API) and logs, for
each step:
  - source_observed prefix passed to MT
  - full hypothesis returned by MT
  - sliding window (last la_n hypotheses) before/after appending
  - LCP of the window
  - whether LCP extends committed_text and by how many chars
  - delta committed
  - new committed_text

Three cases run with fixed segment_size = 1, 2, 3 (one each), so the user
can directly inspect what "segment_size random_uniform_1_3" produces step by
step.

Outputs land alongside this file:
  trace_seg1_<utt>.txt
  trace_seg2_<utt>.txt
  trace_seg3_<utt>.txt
  result_seg1_<utt>.json   # same schema as production runs
  result_seg2_<utt>.json
  result_seg3_<utt>.json
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Reuse production helpers — keep the loop identical to the synth code path.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from local_agreement import (  # noqa: E402
    _extract_reference_text,
    build_source_observed,
    compute_bleu_char,
    compute_laal,
    compute_lcp,
    force_complete_translation,
    get_full_source_text,
    get_source_subsentences,
    load_tokenizer,
    parse_trajectory,
    setup_env,
    translate_source_prefix,
    verify_api,
)


DEFAULT_TSV = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/"
    "train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", default=DEFAULT_TSV)
    p.add_argument("--mt-tokenizer-path", required=True)
    p.add_argument("--mt-api-base", required=True)
    p.add_argument("--mt-api-model", default="qwen3-instruct")
    p.add_argument("--mt-api-timeout", type=float, default=120.0)
    p.add_argument("--la-n", type=int, default=2)
    p.add_argument("--lcp-mode", choices=["char", "word"], default="char")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--target-lang", default="Chinese")
    p.add_argument("--out-dir", default=HERE,
                   help="Where to drop trace_*.txt and result_*.json")
    # Three cases: utt_id + fixed segment_size
    p.add_argument("--utt-seg", action="append", required=True,
                   help="utt_id:segment_size (repeatable, run once per pair)")
    return p.parse_args()


def fmt_window(window: List[str], width: int = 90) -> str:
    if not window:
        return "    (empty)"
    out = []
    for i, h in enumerate(window):
        h_one = h.replace("\n", " ").strip()
        if len(h_one) > width:
            h_one = h_one[:width] + "…"
        out.append(f"    [h{i}] {h_one}")
    return "\n".join(out)


def run_one_verbose(
    row: Dict[str, Any],
    segment_size: int,
    args: argparse.Namespace,
    mt_tokenizer: Any,
    log,
) -> Dict[str, Any]:
    utt_id = str(row.get("id", "row"))
    chunks = parse_trajectory(row["src_trajectory"])
    full_source_text = get_full_source_text(row)
    source_subs = get_source_subsentences(row, full_source_text)
    n_chunks = len(chunks)
    n_steps = (n_chunks + segment_size - 1) // segment_size

    log(f"#" * 100)
    log(f"# utt_id={utt_id}   segment_size={segment_size}   la_n={args.la_n}")
    log(f"# n_chunks={n_chunks}   n_steps={n_steps}   target_lang={args.target_lang}")
    log(f"# full_source: {full_source_text}")
    log(f"# reference  : {_extract_reference_text(row, args.target_lang) or '<none>'}")
    log(f"#" * 100)

    committed_text = ""
    target_deltas = [""] * n_chunks
    actions = ["READ"] * n_chunks
    recent_hypotheses: List[str] = []

    step_start = 0
    step_idx = 0
    while step_start < n_chunks:
        step_end = min(step_start + segment_size, n_chunks)
        is_last_step = (step_end == n_chunks)
        last_chunk_idx = step_end - 1
        source_observed = build_source_observed(chunks, last_chunk_idx)

        log("")
        role = "FORCE_COMPLETE (last step)" if is_last_step else "LCP-decision step"
        log(f"───── step {step_idx}  chunks=[{step_start}..{step_end - 1}]  {role} ─────")
        if step_end - step_start > 1:
            for ci in range(step_start, step_end - 1):
                log(f"  pre-read chunk[{ci}]: {chunks[ci]!r}  (READ, no MT call)")
        log(f"  decision chunk[{last_chunk_idx}]: {chunks[last_chunk_idx]!r}")
        log(f"  source_observed (sent to MT): {source_observed!r}")
        log(f"  committed_text BEFORE step  : {committed_text!r}")

        if is_last_step:
            log(f"  >> FORCE_COMPLETE: pass full_source to MT, prefix-constrain on committed.")
            delta = force_complete_translation(
                mt_tokenizer, full_source_text, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            log(f"  MT continuation delta       : {delta!r}")
            if delta:
                committed_text += delta
            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = "WRITE" if delta else "READ"
            log(f"  committed_text AFTER step   : {committed_text!r}")
        elif not source_observed.strip():
            log(f"  >> empty source prefix; skip MT, keep step as READ.")
            step_start = step_end
            step_idx += 1
            continue
        else:
            log(f"  >> calling MT for hypothesis…")
            hyp = translate_source_prefix(
                mt_tokenizer, source_observed, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            log(f"  MT full hypothesis          : {hyp!r}")
            log(f"  sliding window (la_n={args.la_n}) BEFORE append:")
            log(fmt_window(recent_hypotheses))
            recent_hypotheses.append(hyp)
            if len(recent_hypotheses) > args.la_n:
                dropped = recent_hypotheses.pop(0)
                log(f"  dropped oldest hyp from window: {dropped[:60]!r}…")
            log(f"  sliding window AFTER append:")
            log(fmt_window(recent_hypotheses))

            if len(recent_hypotheses) < args.la_n:
                delta = ""
                action = "READ"
                log(f"  LCP skipped: window has {len(recent_hypotheses)} hyp(s) < la_n={args.la_n}")
                log(f"  >> LA-{args.la_n} requires {args.la_n} hyps to agree → no commit (READ)")
            else:
                lcp = compute_lcp(recent_hypotheses, mode=args.lcp_mode)
                log(f"  LCP({args.lcp_mode}, |window|={len(recent_hypotheses)})       : {lcp!r}")
                if lcp.startswith(committed_text) and len(lcp) > len(committed_text):
                    delta = lcp[len(committed_text):]
                    committed_text = lcp
                    log(f"  >> LCP extends committed by {len(delta)} chars  → COMMIT delta: {delta!r}")
                    action = "WRITE"
                else:
                    delta = ""
                    if not lcp.startswith(committed_text):
                        log(f"  >> LCP does not start with committed_text  → no commit (READ)")
                    else:
                        log(f"  >> LCP == committed_text  → no extension, no commit (READ)")
                    action = "READ"
            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = action
            log(f"  committed_text AFTER step   : {committed_text!r}")

        step_start = step_end
        step_idx += 1

    # Metrics
    reference_text = _extract_reference_text(row, args.target_lang)
    pred_chars = len(str(committed_text).replace(" ", ""))
    ref_chars = len(str(reference_text or "").replace(" ", ""))
    src_words = len(str(full_source_text).strip().split())
    laal_value = float("nan")
    bleu_char_value = float("nan")
    try:
        if reference_text:
            laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
            bleu_char_value = compute_bleu_char(committed_text, reference_text)
    except Exception:
        pass
    length_ratio_ref = pred_chars / ref_chars if ref_chars > 0 else float("nan")
    length_ratio_src = pred_chars / src_words if src_words > 0 else float("nan")

    log("")
    log("=" * 100)
    log(f"FINAL prediction : {committed_text}")
    log(f"FINAL reference  : {reference_text or ''}")
    log(f"metrics: bleu_char={bleu_char_value:.2f}  laal_text={laal_value:.2f}  "
        f"pred_chars={pred_chars}  ref_chars={ref_chars}  src_words={src_words}")
    log("=" * 100)

    return {
        "utt_id": utt_id,
        "src_trajectory": chunks,
        "src_text_full": source_subs,
        "source_full_text": full_source_text,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "reference_text": reference_text or "",
        "decoder_impl": {
            "method": "local_agreement",
            "la_n": args.la_n,
            "segment_size": segment_size,
            "segment_size_sampling": "fixed_for_verbose_replay",
            "lcp_mode": args.lcp_mode,
        },
        "metrics": {
            "laal_text": laal_value,
            "bleu_char": bleu_char_value,
            "pred_chars": pred_chars,
            "ref_chars": ref_chars,
            "src_words": src_words,
            "length_ratio_ref": length_ratio_ref,
            "length_ratio_src": length_ratio_src,
        },
    }


def main() -> None:
    setup_env()
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = verify_api(args.mt_api_base, args.mt_api_timeout)
    if args.mt_api_model not in models:
        raise RuntimeError(f"model '{args.mt_api_model}' not in {models}")
    print(f"[MT] model={args.mt_api_model}  api={args.mt_api_base}", flush=True)
    mt_tokenizer = load_tokenizer(args.mt_tokenizer_path)

    df = pd.read_csv(args.input_tsv, sep="\t")

    for spec in args.utt_seg:
        utt_id, seg_str = spec.rsplit(":", 1)
        seg = int(seg_str)
        rows = df[df["id"].astype(str) == utt_id]
        if rows.empty:
            print(f"[SKIP] utt_id not found: {utt_id}")
            continue
        row = rows.iloc[0].to_dict()

        trace_path = os.path.join(args.out_dir, f"trace_seg{seg}_{utt_id}.txt")
        json_path = os.path.join(args.out_dir, f"result_seg{seg}_{utt_id}.json")

        with open(trace_path, "w", encoding="utf-8") as fh:
            def log(msg=""):
                print(msg, flush=True)
                fh.write(msg + "\n")
                fh.flush()
            result = run_one_verbose(row, seg, args, mt_tokenizer, log)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"[WROTE] {trace_path}")
        print(f"[WROTE] {json_path}")


if __name__ == "__main__":
    main()
