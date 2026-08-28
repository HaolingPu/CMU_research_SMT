#!/usr/bin/env python3
"""
Adapt SALAMI raw LLM output → format consumable by multi_trajectory_gigaspeech.py.

SALAMI raw JSON shape (per utt):
  {
    "segmented_pairs": [[eng_chunk, tgt_chunk], ...],
    "input":  "<full English sentence(s)>",
    "utt_id": "...",
    "target_lang": "Japanese" | "German" | "Chinese",
    "output": "<raw LLM text>"
  }

Output JSON shape (multi_trajectory-ready):
  {
    "utt_id": "...",
    "input":  "...",
    "target_lang": "ja" | "de" | "zh",
    "src_trajectory": [..]              # joined from manifest TSV
    "offline": {"Source": [...], "Target": [...]}
  }

Rows whose segmented_pairs is empty / utt has 'error' field / not found in
manifest → written out with an "error" field so downstream just skips them.
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from tqdm import tqdm


TGT_LANG_NORMALIZE = {
    "japanese": "ja",
    "ja": "ja",
    "german": "de",
    "de": "de",
    "chinese": "zh",
    "zh": "zh",
}


def source_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def parse_trajectory(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
    except Exception:
        return None
    if not isinstance(v, list):
        return None
    return [str(x) for x in v]


def load_manifest_passthrough(tsv_path: str, id_col: str = "id") -> Dict[str, Dict[str, str]]:
    """utt_id -> dict of passthrough fields needed downstream."""
    keep = {"audio", "n_frames", "speaker", "src_text_full", "src_lang",
            "tgt_lang", "src_trajectory", "asr", "src_text"}
    out: Dict[str, Dict[str, str]] = {}
    csv.field_size_limit(sys.maxsize)
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="load manifest"):
            utt = row.get(id_col)
            if not utt:
                continue
            out[utt] = {k: row.get(k, "") for k in keep}
    return out


def adapt_one(salami_obj: Dict[str, Any], manifest_row: Optional[Dict[str, str]]) -> Dict[str, Any]:
    utt_id = salami_obj.get("utt_id", "")
    if "error" in salami_obj:
        return {"utt_id": utt_id, "error": f"salami_raw_error: {salami_obj.get('error')}"}
    if manifest_row is None:
        return {"utt_id": utt_id, "error": "utt_not_in_manifest"}

    pairs = salami_obj.get("segmented_pairs")
    if not isinstance(pairs, list) or len(pairs) == 0:
        return {"utt_id": utt_id, "error": "empty_segmented_pairs"}

    src: List[str] = []
    tgt: List[str] = []
    for p in pairs:
        if not (isinstance(p, list) and len(p) == 2):
            return {"utt_id": utt_id, "error": "bad_pair_shape"}
        s = "" if p[0] is None else str(p[0]).strip()
        t = "" if p[1] is None else str(p[1]).strip()
        if not s or not t:
            # skip empty-on-either-side chunks instead of failing
            continue
        src.append(s)
        tgt.append(t)
    if not src:
        return {"utt_id": utt_id, "error": "all_pairs_empty_after_strip"}

    input_text = str(salami_obj.get("input", ""))
    if input_text and source_tokens(" ".join(src)) != source_tokens(input_text):
        return {"utt_id": utt_id, "error": "source_input_token_mismatch"}

    tgt_lang_raw = salami_obj.get("target_lang") or manifest_row.get("tgt_lang", "")
    tgt_lang = TGT_LANG_NORMALIZE.get(str(tgt_lang_raw).lower(), str(tgt_lang_raw).lower())

    out: Dict[str, Any] = {
        "utt_id": utt_id,
        "input": input_text,
        "target_lang": tgt_lang,
        "tgt_lang": tgt_lang,
        "src_lang": manifest_row.get("src_lang", "en"),
        "audio": manifest_row.get("audio", ""),
        "n_frames": manifest_row.get("n_frames", ""),
        "speaker": manifest_row.get("speaker", ""),
        "src_text": manifest_row.get("src_text", ""),
        "src_text_full": manifest_row.get("src_text_full", ""),
        "asr": manifest_row.get("asr", ""),
        "src_trajectory": manifest_row.get("src_trajectory", ""),
        "offline": {"Source": src, "Target": tgt},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="dir of SALAMI raw per-utt JSON")
    ap.add_argument("--manifest-tsv", required=True,
                    help="gigaspeech manifest TSV with src_trajectory column")
    ap.add_argument("--output-dir", required=True,
                    help="where to write adapted per-utt JSON")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[adapt] loading manifest: {args.manifest_tsv}")
    manifest = load_manifest_passthrough(args.manifest_tsv)
    print(f"[adapt] manifest rows: {len(manifest):,}")

    files = sorted(os.listdir(args.raw_dir))
    files = [f for f in files if f.endswith(".json")]
    print(f"[adapt] raw files: {len(files):,}")

    ok = err = skipped = miss_manifest = 0
    for fn in tqdm(files, desc="adapt"):
        out_path = os.path.join(args.output_dir, fn)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            with open(os.path.join(args.raw_dir, fn), "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": os.path.splitext(fn)[0],
                           "error": f"json_load: {e}"}, f, ensure_ascii=False)
            err += 1
            continue
        utt_id = obj.get("utt_id") or os.path.splitext(fn)[0]
        row = manifest.get(utt_id)
        if row is None:
            miss_manifest += 1
        adapted = adapt_one({**obj, "utt_id": utt_id}, row)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(adapted, f, ensure_ascii=False)
        if "error" in adapted:
            err += 1
        else:
            ok += 1

    print(f"\n[adapt] done. ok={ok:,}  err={err:,}  skipped_existing={skipped:,}  "
          f"miss_manifest={miss_manifest:,}")


if __name__ == "__main__":
    main()
