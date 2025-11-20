import os
import json
import tgt

# =============================
# Load MFA word alignment
# =============================
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


# =============================
# Match LLM chunk → MFA timestamps
# =============================
def normalize_text(s):
    import re
    s = s.lower()
    # 保留字母、空格、撇号
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def match_llm_chunks_to_mfa(llm_chunks, mfa_words):
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    

    results = []
    for chunk in llm_chunks:
        tokens = normalize_text(chunk).split()

        matched_indices = []
        for t in tokens: # 遍历chunk里的每一个词
            for i, w in enumerate(mfa_tokens):  # loop MFA words
                if w == t:
                    matched_indices.append(i)
                    break

        if not matched_indices:
            results.append({"chunk": chunk, "start": None, "end": None})
            continue

        start = min(mfa_words[i]["start"] for i in matched_indices)
        end   = max(mfa_words[i]["end"]   for i in matched_indices)

        results.append({"chunk": chunk, "start": start, "end": end})

    return results


# =============================
# Conservative second-based streaming
# =============================
def assign_chunks_by_second(aligned_chunks):
    max_time = max(x["end"] for x in aligned_chunks)
    max_second = int(max_time) + 1

    emitted = []
    timeline = []

    for sec in range(max_second):
        sec_end = sec + 1.0
        to_emit = []

        for i, item in enumerate(aligned_chunks):
            if i in emitted:
                continue
            if item["end"] <= sec_end:
                to_emit.append(item["chunk"])
                emitted.append(i)

        timeline.append({"second": sec, "emit": to_emit})

    return timeline


# =============================
# Build final source/target segments
# =============================
def build_final_segments(timeline, llm_eng, llm_zh):
    eng2zh = {e: z for e, z in zip(llm_eng, llm_zh)}

    sources = []
    targets = []

    for entry in timeline:
        eng_list = entry["emit"]
        eng_concat = " ".join(eng_list).strip()

        zh_list = [eng2zh.get(e, "") for e in eng_list]
        zh_concat = "".join(zh_list).strip()

        sources.append(eng_concat)
        targets.append(zh_concat)

    return sources, targets



# =======================================================
#                  MAIN BATCH PROCESSOR
# =======================================================
def process_all(
    llm_dir,
    mfa_dir,
    output_dir,
    allowed_ids=None,
    limit=None
):
    os.makedirs(output_dir, exist_ok=True)


    llm_files = sorted(os.listdir(llm_dir))
    if limit is not None:
        llm_files = llm_files[:limit]

    print(f"Processing {len(llm_files)} files...")

    for idx, fname in enumerate(llm_files):
        if not fname.endswith(".json"):
            continue

        utt_id = fname[:-5]   # "utt_000001"

        # ✅ 这里直接用 padded 形式
        textgrid_name = utt_id
        textgrid_path = os.path.join(mfa_dir, f"{utt_id}.TextGrid")


        # === FILTERING: skip if not in allowed list ===
        if allowed_ids is not None and textgrid_name not in allowed_ids:
            print(f"❌Skipping {utt_id} (not in good alignment list)")
            continue

        # Load LLM JSON
        llm_path = os.path.join(llm_dir, fname)
        with open(llm_path, "r", encoding="utf-8") as f:
            seg_json = json.load(f)

        # Load MFA words
        mfa_words = load_word_alignment(textgrid_path)

        out_json = {"utt_id": utt_id, "original_text": seg_json.get("input", "")}

        # Process low/medium/high
        for level in ["low_latency", "medium_latency", "high_latency"]:
            eng_chunks = seg_json[level]["English"]
            zh_chunks  = seg_json[level]["Chinese"]

            aligned = match_llm_chunks_to_mfa(eng_chunks, mfa_words)
            # print(aligned)
            timeline = assign_chunks_by_second(aligned)
            src, tgt = build_final_segments(timeline, eng_chunks, zh_chunks)

            out_json[f"source_{level}"] = src
            out_json[f"target_{level}"] = tgt

        # Write output
        save_path = os.path.join(output_dir, f"{utt_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        print(f"✅Saved: {save_path}")

    print("👌All done!")


# =======================================================
# Example call
# =======================================================

good_ids = [json.loads(line)["file"].replace(".TextGrid", "") for line in open("/data/user_data/haolingp/outputs/good.jsonl")]


if __name__ == "__main__":
    process_all(
        llm_dir="/data/user_data/haolingp/outputs/llm_segmentation_json",
        mfa_dir="/data/user_data/haolingp/outputs/mfa_output",
        output_dir="/data/user_data/haolingp/outputs/streaming_dataset_v2",
        allowed_ids=good_ids,
        limit=20     # ← 你可以调整这里
    )
