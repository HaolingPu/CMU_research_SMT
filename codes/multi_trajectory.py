import os
import json
import re
import tgt
from tqdm import tqdm


############################################################
# Utility
############################################################
def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_word_alignment(textgrid_path):
    tg = tgt.read_textgrid(textgrid_path)
    words = tg.get_tier_by_name("words").intervals
    out = []
    for w in words:
        if w.text.strip():
            out.append({
                "word": w.text.lower(),
                "start": w.start_time,
                "end": w.end_time
            })
    return out


def match_llm_chunks_to_mfa(llm_chunks, mfa_words):
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    results = []

    for chunk in llm_chunks:
        tokens = normalize_text(chunk).split()
        matched = []

        for t in tokens:
            for i, w in enumerate(mfa_tokens):
                if w == t:
                    matched.append(i)
                    break

        if not matched:
            results.append({"chunk": chunk, "start": None, "end": None})
            continue

        start = min(mfa_words[i]["start"] for i in matched)
        end = max(mfa_words[i]["end"] for i in matched)
        results.append({"chunk": chunk, "start": start, "end": end})

    return results


def assign_chunks_by_second(aligned_chunks):
    valid = [x for x in aligned_chunks if x["end"] is not None]
    if not valid:
        return [{"second": 0, "emit": []}]

    max_time = max(x["end"] for x in valid)
    max_sec = int(max_time) + 1

    emitted = []
    timeline = []

    for sec in range(max_sec):
        sec_end = sec + 1
        to_emit = []

        for i, item in enumerate(aligned_chunks):
            if i in emitted:
                continue
            if item["end"] is not None and item["end"] <= sec_end:
                to_emit.append(item["chunk"])
                emitted.append(i)

        timeline.append({"second": sec, "emit": to_emit})

    return timeline


def build_final_segments(timeline, eng_chunks, zh_chunks):
    eng2zh = {e: z for e, z in zip(eng_chunks, zh_chunks)}
    sources, targets = [], []

    for entry in timeline:
        eng_list = entry["emit"]
        src = " ".join(eng_list).strip()
        tgt = "".join([eng2zh.get(e, "") for e in eng_list]).strip()

        sources.append(src)
        targets.append(tgt)

    return sources, targets


############################################################
# MAIN FOR ONE LANGUAGE
############################################################
def process_language(lang, LLM_ROOT, MFA_ROOT, GOOD_JSONL, OUTPUT_ROOT):

    print(f"\n==============================")
    print(f"🌍 Processing language {lang}")
    print(f"==============================\n")

    # ---- load good IDs (只存utt_base，不包含路径) ----
    good_ids = set()
    with open(GOOD_JSONL, "r") as f:
        for line in f:
            obj = json.loads(line)
            # 只添加file字段，例如 "utt_en000_00000000_0000"
            good_ids.add(obj["file"])

    print(f"Loaded {len(good_ids)} good IDs for {lang}\n")

    lang_llm_dir = os.path.join(LLM_ROOT, lang)
    lang_mfa_dir = os.path.join(MFA_ROOT, lang)

    if not os.path.isdir(lang_llm_dir) or not os.path.isdir(lang_mfa_dir):
        print(f"⚠️ Missing LLM or MFA directory for {lang}, skip.")
        return

    # 获取所有parquet目录（8位数字）
    parquet_dirs = sorted([
        d for d in os.listdir(lang_llm_dir)
        if os.path.isdir(os.path.join(lang_llm_dir, d)) and d.isdigit() and len(d) == 8
    ])

    total_processed = 0

    for pq in parquet_dirs:
        llm_pq_dir = os.path.join(lang_llm_dir, pq)
        mfa_pq_dir = os.path.join(lang_mfa_dir, pq)

        if not os.path.isdir(llm_pq_dir):
            continue
        if not os.path.isdir(mfa_pq_dir):
            print(f"⚠️ Missing MFA for parquet {pq}, skip.")
            continue

        print(f"  📁 Parquet: {pq}")

        out_dir = os.path.join(OUTPUT_ROOT, lang, pq)
        os.makedirs(out_dir, exist_ok=True)

        llm_files = sorted(f for f in os.listdir(llm_pq_dir) if f.endswith(".json"))

        parquet_count = 0

        for fname in tqdm(llm_files, desc=f"    {pq}"):

            utt_base = fname.replace(".json", "")  # "utt_en000_00000000_0000"
            
            # ✅ 修复：直接用utt_base检查，不加路径
            if utt_base not in good_ids:
                continue

            llm_path = os.path.join(llm_pq_dir, fname)
            textgrid_path = os.path.join(mfa_pq_dir, utt_base + ".TextGrid")
            out_path = os.path.join(out_dir, utt_base + ".json")

            # 检查TextGrid是否存在
            if not os.path.exists(textgrid_path):
                print(f"    ⚠️ TextGrid not found: {utt_base}")
                continue

            try:
                seg = json.load(open(llm_path))
                mfa_words = load_word_alignment(textgrid_path)

                out_json = {
                    "utt_id": utt_base,
                    "original_text": seg.get("input", "")
                }

                for level in ["low_latency", "medium_latency", "high_latency"]:
                    eng = seg[level]["English"]
                    zh = seg[level]["Chinese"]

                    aligned = match_llm_chunks_to_mfa(eng, mfa_words)
                    timeline = assign_chunks_by_second(aligned)
                    src, tgt = build_final_segments(timeline, eng, zh)

                    out_json[f"source_{level}"] = src
                    out_json[f"target_{level}"] = tgt

                json.dump(out_json, open(out_path, "w"), ensure_ascii=False, indent=2)
                parquet_count += 1
                total_processed += 1

            except Exception as e:
                print(f"    ❌ Error processing {utt_base}: {e}")
                continue
        
        print(f"    ✓ Processed {parquet_count} files from {pq}")

    print(f"\n✨ Done {lang}! Total processed: {total_processed}\n")


############################################################
# RUN FOR ALL LANGS
############################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate streaming trajectories from good LLM + MFA results."
    )

    parser.add_argument(
        "--llm-root",
        type=str,
        required=True,
        help="Root directory of LLM outputs (contains lang subdirs).",
    )
    parser.add_argument(
        "--mfa-root",
        type=str,
        required=True,
        help="Root directory of MFA TextGrid outputs (contains lang subdirs).",
    )
    parser.add_argument(
        "--good-root",
        type=str,
        required=True,
        help="Directory containing good_{lang}_all.jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output directory for streaming_dataset.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en000"],
        help="Language splits to process (default: en000).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for lang in args.langs:
        good_jsonl = os.path.join(args.good_root, f"good_EAST_{lang}_all.jsonl")

        if not os.path.exists(good_jsonl):
            print(f"❌ Good list not found: {good_jsonl}")
            continue

        print(f"\n===== Processing {lang} =====")
        process_language(
            lang,
            args.llm_root,
            args.mfa_root,
            good_jsonl,
            args.output_root,
        )

    print("\n==============================")
    print("🎉 ALL LANGUAGES COMPLETED!")
    print("==============================\n")
