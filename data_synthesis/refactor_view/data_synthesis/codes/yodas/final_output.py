import os
import json
from tqdm import tqdm
from collections import defaultdict


def parse_utt_id(utt: str):
    """utt_en000_00000000_0017 → (en000, 00000000)"""
    parts = utt.split("_")
    return parts[1], parts[2]


def build_stream_path_index(stream_root: str, only_lang=None, only_pq=None):
    """
    Build utt_id -> file_path index (store paths only, not full JSON).
    Optional filters:
      - only_lang: e.g. "en000"
      - only_pq:   e.g. "00000000"
    """
    print(f"\n📥 Indexing streaming dataset paths from: {stream_root}")
    if only_lang:
        print(f"   • only_lang = {only_lang}")
    if only_pq:
        print(f"   • only_pq   = {only_pq}")

    index = {}
    bad = 0

    for dirpath, _, files in os.walk(stream_root):
        # Fast path pruning by directory structure if possible: .../<lang>/<pq>/
        # If user uses that structure, we can skip unrelated dirs.
        if only_lang and (os.sep + only_lang + os.sep) not in (dirpath + os.sep):
            continue
        if only_pq and (os.sep + only_pq + os.sep) not in (dirpath + os.sep):
            continue

        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                utt_id = data.get("utt_id")
                if not utt_id:
                    bad += 1
                    continue

                lang, pqid = parse_utt_id(utt_id)
                if only_lang and lang != only_lang:
                    continue
                if only_pq and pqid != only_pq:
                    continue

                index[utt_id] = p
            except Exception:
                bad += 1
                continue

    print(f"✓ Indexed {len(index)} utt paths. Bad/failed: {bad}\n")
    return index


def open_writers(output_root: str):
    """
    Cache file handles: (lang, pqid, latency) -> open file
    """
    writers = {}
    os.makedirs(output_root, exist_ok=True)

    def get_writer(lang: str, pqid: str, latency: str):
        key = (lang, pqid, latency)
        if key in writers:
            return writers[key]

        base = os.path.join(output_root, lang, pqid)
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, f"{latency}_latency.jsonl")
        f = open(out_path, "a", encoding="utf-8")  # append mode
        writers[key] = f
        return f

    return writers, get_writer


def build_jsonl(
    metricx_file: str,
    stream_root: str,
    output_root: str,
    only_lang=None,
    only_pq=None,
    max_lines=None,
):
    print(f"\n{'='*60}")
    print("Building trajectory-style dataset (streaming segments)")
    print(f"{'='*60}\n")

    # 1) utt_id -> path (filtered)
    stream_path_index = build_stream_path_index(stream_root, only_lang=only_lang, only_pq=only_pq)

    # 2) open writers cache
    writers, get_writer = open_writers(output_root)

    processed = 0
    skipped = 0
    read_lines = 0

    if not os.path.exists(metricx_file):
        print(f"❌ Error: MetricX file not found: {metricx_file}")
        return

    print(f"📥 Reading MetricX filtered file (streaming): {metricx_file}")
    if only_lang:
        print(f"   • only_lang = {only_lang}")
    if only_pq:
        print(f"   • only_pq   = {only_pq}")
    if max_lines is not None:
        print(f"   • max_lines = {max_lines}")

    try:
        with open(metricx_file, "r", encoding="utf-8") as f:
            # tqdm wants a total; if max_lines is set, we can set total=max_lines
            it = tqdm(f, desc="Processing entries", total=max_lines if max_lines is not None else None)

            for line in it:
                if max_lines is not None and read_lines >= max_lines:
                    break
                read_lines += 1

                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    item = json.loads(line)
                    meta = item.get("metadata", {})
                    utt = meta.get("utt_id")
                    latency = meta.get("latency")

                    if not utt or not latency:
                        skipped += 1
                        continue

                    lang, pqid = parse_utt_id(utt)

                    # Apply filters early
                    if only_lang and lang != only_lang:
                        skipped += 1
                        continue
                    if only_pq and pqid != only_pq:
                        skipped += 1
                        continue

                    # normalize latency values
                    if latency not in {"low", "medium", "high", "offline"}:
                        skipped += 1
                        continue

                    # fetch corresponding streaming json path
                    p = stream_path_index.get(utt)
                    if not p:
                        skipped += 1
                        continue

                    with open(p, "r", encoding="utf-8") as sf:
                        seg = json.load(sf)

                    # keys in your streaming json
                    if latency == "offline":
                        src_key = "source_offline"
                        tgt_key = "target_offline"
                    else:
                        src_key = f"source_{latency}_latency"   # source_low_latency
                        tgt_key = f"target_{latency}_latency"   # target_low_latency

                    if src_key not in seg or tgt_key not in seg:
                        skipped += 1
                        continue

                    traj_entry = {
                        "utt_id": utt,
                        "latency": latency,
                        "source": seg[src_key],   # list of segments
                        "target": seg[tgt_key],   # list of segments
                    }

                    out_f = get_writer(lang, pqid, latency)
                    out_f.write(json.dumps(traj_entry, ensure_ascii=False) + "\n")
                    processed += 1

                except Exception:
                    skipped += 1
                    continue

    finally:
        # close all writers
        for wf in writers.values():
            try:
                wf.close()
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("🎉 DONE!")
    print(f"{'='*60}")
    print(f"Read lines      : {read_lines}")
    print(f"Processed entries: {processed}")
    print(f"Skipped entries  : {skipped}")
    print(f"Output directory : {output_root}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metricx_jsonl", required=True,
                        help="Sentence-level MetricX QE output (filtered)")
    parser.add_argument("--stream_dir", required=True,
                        help="Streaming dataset directory")
    parser.add_argument("--output_dir", required=True,
                        help="Where to store structured dataset")

    # ✅ Debug options
    parser.add_argument("--only-lang", type=str, default=None,
                        help="Only process this language, e.g. en000")
    parser.add_argument("--only-pq", type=str, default=None,
                        help="Only process this parquet id, e.g. 00000000")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Only read first N lines of metricx_jsonl (debug)")

    args = parser.parse_args()

    build_jsonl(
        metricx_file=args.metricx_jsonl,
        stream_root=args.stream_dir,
        output_root=args.output_dir,
        only_lang=args.only_lang,
        only_pq=args.only_pq,
        max_lines=args.max_lines,
    )
