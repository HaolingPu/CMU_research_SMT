"""Per-utterance orchestration and the main entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

from .cli import parse_args, setup_env
from .consensus import extend_pending_tokens, force_complete_translation
from .future_sampling import sample_source_futures_multi
from .http_client import normalize_api_base, verify_api
from .metrics import (
    _extract_reference_text_from_row,
    _nonspace_char_count,
    compute_bleu_char,
    compute_laal,
    compute_length_ratio_ref,
    compute_length_ratio_src,
)
from .text_utils import (
    build_source_observed,
    build_source_observed_recent_units,
    get_full_source_text,
    parse_source_units,
    parse_trajectory,
    sanitize_filename,
)
from .tokens import _single_token_text, decode_token_ids_to_text, finalize_external_commit
from .verbose import _TeeWriter, _vlog, _vlog_pretty_value, write_pretty_json


def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    source_units = parse_source_units(row.get("src_text_full"))
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
    _vlog(verbose_log_file, f"# utt_id: {utt_id}")
    _vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    _vlog_pretty_value(verbose_log_file, "src_text_full", source_units)
    _vlog_pretty_value(verbose_log_file, "src_trajectory", chunks)
    _vlog(verbose_log_file, f"# Chunks: {len(chunks)}")
    _vlog(verbose_log_file, f"# num_futures={args.num_futures}, top_k={args.candidate_top_k}, min_p={args.min_p}")
    for si, bs in enumerate(base_specs):
        label = bs.get("name", f"spec_{si}")
        _vlog(verbose_log_file, f"# base_model[{label}]: api model={bs.get('api_model','')} base={bs.get('api_base','')} num_futures={bs.get('num_futures','')}")
    _vlog(verbose_log_file, f"# instruct_backend: vllm_completion")
    _vlog(verbose_log_file, "############################################################")

    for t in range(len(chunks)):
        current_source_chunk = str(chunks[t] or "")
        source_observed_full = build_source_observed(chunks, t)
        source_observed = build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=args.future_source_window_chunks,
        )
        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}")
        _vlog(verbose_log_file, f"source_observed: {current_source_chunk!r}")
        _vlog(verbose_log_file, f"future_source_prefix: {source_observed!r}")
        if source_observed != source_observed_full:
            _vlog(verbose_log_file, f"source_observed_full: {source_observed_full!r}")
        _vlog(verbose_log_file, f"committed_before: {committed_text!r}")

        if t == len(chunks) - 1: #最后一个chunk，不再做共识，直接让instruct model把翻译补完
            final_delta = force_complete_translation(
                tokenizer=instruct_tokenizer,
                full_source=source_observed_full,
                committed_text=committed_text,
                api_base=args.instruct_api_base,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang,
                max_tokens=args.final_max_tokens,
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

        if len(futures) <= 3: #future太少无法做共识，等待更多源语言输入
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        pending_token_ids, grow_logs = extend_pending_tokens(
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
        )

        # ── min-consensus-horizon filter ──
        # If the consensus only carried us a few tokens before breaking, the path is fragile.
        # Discard the whole pending buffer and READ instead, to avoid early-commit lock-in.
        if (
            args.min_consensus_horizon > 1
            and 0 < len(pending_token_ids) < args.min_consensus_horizon
        ):
            pending_text_dropped = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
            _vlog(verbose_log_file,
                  f"[Step 5.5] horizon_filter: dropped pending={pending_text_dropped!r} "
                  f"(len={len(pending_token_ids)} < min_horizon={args.min_consensus_horizon}) -> READ")
            grow_logs.append({"step": "filter", "stop": "below_min_horizon",
                              "horizon": len(pending_token_ids),
                              "min_horizon": args.min_consensus_horizon})
            pending_token_ids = []

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
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {"candidate_policy": candidate_policy, "backend": "vllm_completion"},
    }

    reference_text = _extract_reference_text_from_row(row, target_lang=args.target_lang)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    length_ratio_ref = float("nan")
    try:
        if not reference_text:
            raise ValueError("reference_text_unavailable")
        laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
        bleu_char_value = compute_bleu_char(committed_text, reference_text)
        length_ratio_ref = compute_length_ratio_ref(committed_text, reference_text)
    except Exception:
        pass
    length_ratio_src = compute_length_ratio_src(committed_text, full_source_text)

    result["reference_text"] = reference_text or ""
    result["metrics"] = {
        "laal_text": laal_value,
        "bleu_char": bleu_char_value,
        "length_ratio_ref": length_ratio_ref,
        "length_ratio_src": length_ratio_src,
        "pred_chars": _nonspace_char_count(committed_text),
        "ref_chars": _nonspace_char_count(reference_text or ""),
        "src_words": len(str(full_source_text or "").split()),
    }
    _vlog(
        verbose_log_file,
        f"  prediction={committed_text!r} bleu={bleu_char_value:.2f} laal={laal_value:.2f} "
        f"len_ratio_ref={length_ratio_ref:.2f} len_ratio_src={length_ratio_src:.2f}",
    )
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

    primary_num_futures = args.num_futures - args.secondary_num_futures # 两个base 各10个 future sampling
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
    failures: List[str] = []

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
                    row_id = futs[fut]
                    print(f"[ERROR] Row {row_id} raised: {exc}", file=sys.stderr)
                    failures.append(f"Row {row_id}: {exc}")

    if failures:
        print(f"[FATAL] {len(failures)} row(s) failed", file=sys.stderr)
        raise SystemExit(1)
