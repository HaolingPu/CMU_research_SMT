#!/usr/bin/env python3
"""Repackage EAST llm_output_merged as consensus-format JSONs for SegAlign reuse.

For each EAST utt that has a non-error streaming JSON, emit one fake-consensus
JSON per latency:
    consensus_root/job_east_<lang>/task_0/<utt_id>__<lat>.json
with fields {utt_id (=doc_id), src_text_full (input_sentences), prediction
(joined Target chunks at this latency), reference_text}, plus EAST metadata.

This lets prepare_segale_shards.py consume the data unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPACE_JOIN_LANGS = {"de", "en", "fr", "es"}


def join_target(chunks, target_lang):
    sep = " " if target_lang in SPACE_JOIN_LANGS else ""
    return sep.join(c.strip() for c in chunks if c is not None and str(c).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-merged-dir", required=True)
    ap.add_argument("--stream-dir", required=True)
    ap.add_argument("--out-root", required=True,
                    help="Will write <out-root>/job_east_<lang>/task_0/<doc>.json")
    ap.add_argument("--target-lang", required=True)
    ap.add_argument("--latencies", nargs="+", default=["low", "medium", "high"])
    args = ap.parse_args()

    llm_dir = Path(args.llm_merged_dir)
    stream_dir = Path(args.stream_dir)
    out_root = Path(args.out_root)
    job_dir = out_root / f"job_east_{args.target_lang}" / "task_0"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Build set of OK utts (streaming JSON without 'error').
    ok_utts = set()
    for f in stream_dir.glob("*/*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "error" not in d and d.get("utt_id"):
            ok_utts.add(d["utt_id"])
    print(f"[info] OK streaming utts: {len(ok_utts)}")

    n_in = 0
    n_skip_err = 0
    n_skip_not_ok = 0
    n_skip_no_sents = 0
    n_out = 0
    for f in llm_dir.glob("*.json"):
        n_in += 1
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error"):
            n_skip_err += 1
            continue
        utt = d.get("utt_id")
        if not utt or utt not in ok_utts:
            n_skip_not_ok += 1
            continue
        src_sents = d.get("input_sentences") or []
        src_sents = [str(s).strip() for s in src_sents if str(s).strip()]
        if not src_sents:
            n_skip_no_sents += 1
            continue

        # streaming json path (real)
        stream_subdir = utt.rsplit("_", 1)[0]
        stream_path = stream_dir / stream_subdir / f"{utt}.json"

        for lat in args.latencies:
            block = d.get(f"{lat}_latency") or {}
            tgts = block.get("Target") or []
            if not tgts:
                continue
            joined = join_target(tgts, args.target_lang)
            if not joined:
                continue
            doc_id = f"{utt}__{lat}"
            out_obj = {
                "utt_id": doc_id,
                "src_text_full": src_sents,
                "prediction": joined,
                "reference_text": "",
                "east_utt_id": utt,
                "east_latency": lat,
                "east_stream_json": str(stream_path),
            }
            (job_dir / f"{doc_id}.json").write_text(
                json.dumps(out_obj, ensure_ascii=False), encoding="utf-8")
            n_out += 1

    print(f"[done] llm_merged seen={n_in}  skipped(err={n_skip_err}, not_ok={n_skip_not_ok}, no_sents={n_skip_no_sents})")
    print(f"[done] consensus-format JSONs written: {n_out}  -> {job_dir}")


if __name__ == "__main__":
    main()
