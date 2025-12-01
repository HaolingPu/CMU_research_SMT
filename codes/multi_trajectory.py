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


############################################################
# Load MFA word alignment
############################################################
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


############################################################
# Match LLM chunks to MFA timestamps
############################################################
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


############################################################
# Conservative second-based streaming
############################################################
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


############################################################
# Build final source/target segments
############################################################
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
# MAIN: process all languages / all parquets / all utt
############################################################
def generate_all(LLM_ROOT, MFA_ROOT, GOOD_JSONL, OUTPUT_ROOT):

    # ---- load good IDs ----
    good_ids = set()
    with open(GOOD_JSONL, "r") as f:
        for line in f:
            obj = json.loads(line)
            good_ids.add(obj["file"])

    print(f"Loaded {len(good_ids)} good IDs\n")

    # ---- loop languages ----
    langs = sorted(os.listdir(LLM_ROOT))
    for lang in langs:
        lang_llm_dir = os.path.join(LLM_ROOT, lang)
        if not os.path.isdir(lang_llm_dir):
            continue

        print(f"🌍 Processing language: {lang}")

        # match MFA directory
        lang_mfa_dir = os.path.join(MFA_ROOT, lang)
        if not os.path.isdir(lang_mfa_dir):
            print(f"❌ Missing MFA dir for {lang}, skipping.")
            continue

        # loop parquet dirs (00000000, 00000001...)
        parquet_dirs = sorted(os.listdir(lang_llm_dir))
        for pq in parquet_dirs:
            llm_pq_dir = os.path.join(lang_llm_dir, pq)
            mfa_pq_dir = os.path.join(lang_mfa_dir, pq)

            if not os.path.isdir(llm_pq_dir):
                continue
            if not os.path.isdir(mfa_pq_dir):
                print(f"⚠️ Missing MFA for {pq}, skip.")
                continue

            print(f"  📁 Parquet: {pq}")

            # output directory
            out_dir = os.path.join(OUTPUT_ROOT, lang, pq)
            os.makedirs(out_dir, exist_ok=True)

            llm_files = sorted(f for f in os.listdir(llm_pq_dir) if f.endswith(".json"))

            for fname in tqdm(llm_files):

                utt_base = fname[:-5]
                if utt_base not in good_ids:
                    continue

                llm_path = os.path.join(llm_pq_dir, fname)
                textgrid_path = os.path.join(mfa_pq_dir, utt_base + ".TextGrid")
                out_path = os.path.join(out_dir, utt_base + ".json")

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

    print("✨ DONE! All trajectories generated.")


############################################################
# Run
############################################################
if __name__ == "__main__":
    generate_all(
        LLM_ROOT="/data/user_data/haolingp/outputs/llm_segmentation_json",
        MFA_ROOT="/data/user_data/haolingp/outputs/mfa_textgrid_output",
        GOOD_JSONL="/data/user_data/haolingp/outputs/good_en000_all.jsonl",
        OUTPUT_ROOT="/data/user_data/haolingp/outputs/streaming_dataset"
    )
