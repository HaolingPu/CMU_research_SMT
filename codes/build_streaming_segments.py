import os
import io
import re
import json
import pandas as pd
import soundfile as sf
import tgt

############################################################
# 1. configuration
############################################################

PARQUET_PATH = "/data/hf_cache/yodas-granary/data/en000/asr_only/00000000.parquet"
MFA_CORPUS_DIR = "/data/user_data/haolingp/outputs/mfa_corpus_test"
MFA_OUTPUT_DIR = "/data/user_data/haolingp/outputs/mfa_output"
LLM_JSON_PATH = "/data/user_data/haolingp/outputs/llm_segmentation_json/utt_000000.json"

os.makedirs(MFA_CORPUS_DIR, exist_ok=True)

MISALIGNMENT_LOG = "/data/user_data/haolingp/outputs/misalignment_log.json"



############################################################
# 2. From parquet, read N  audio/text → write into corpus
############################################################

def export_corpus(parquet_path, corpus_dir, num_samples=10):
    df = pd.read_parquet(parquet_path)
    df = df.iloc[:num_samples]

    for i, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]  # 二进制数据（wav）
        text = row["text"]
        uttid = f"utt_{i}"

        wav_path = os.path.join(corpus_dir, f"{uttid}.wav")
        lab_path = os.path.join(corpus_dir, f"{uttid}.lab")

        # create wav file from bytes
        audio_file = io.BytesIO(audio_bytes) # 原始音频字节包装成一个“文件对象”
        audio, sr = sf.read(audio_file)  #numpy array， sample rate
        sf.write(wav_path, audio, sr) # 保存成真正的 .wav 文件

        with open(lab_path, "w") as f:
            f.write(text)

        # print(f"Wrote {wav_path} and {lab_path}")

    print("Done exporting corpus →", corpus_dir)
    return df


############################################################
# 3. Analyze and fetch each TextGrid → get word-level timeframe
############################################################

def load_word_alignment(textgrid_path):
    try:
        tg = tgt.read_textgrid(textgrid_path) 
        words = tg.get_tier_by_name("words").intervals # 从 "words" tier 取得所有 word intervals

        out = []
        for w in words:
            # 跳过空白 label
            if w.text.strip():
                out.append({
                    # "word": w.text.lower(), # 单词 (统一转成小写便于匹配)
                    "word": w.text,
                    "start": w.start_time,
                    "end": w.end_time
                })
        
        return out
    
    except Exception as e:
        print(f"❌ Error loading TextGrid {textgrid_path}: {e}")
        return None


############################################################
# 4. 构造 1 秒 segments
############################################################

def build_1s_segments(words):
    max_time = words[-1]["end"]
    segments = []


    i = 0          # 当前扫描到第几个 word
    n = len(words)

    for sec in range(max_time):
        sec_start = sec
        sec_end   = sec + 1
        seg_words = []

        # 将指针推进到第一次可能 overlap 的 word
        while i < n and words[i]["end"] <= sec_start:
            i += 1

        # 从 i 开始收集本秒的所有 overlap 单词
        j = i
        while j < n and words[j]["start"] < sec_end:
            # overlap 条件：start < sec_end 且 end > sec_start
            if words[j]["end"] > sec_start:
                seg_words.append(words[j]["word"])
            j += 1

        segments.append({"second": sec, "words": seg_words})
    return segments


############################################################
# 5. 将 LLM chunks 匹配到 MFA word-level 时间
############################################################

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    return " ".join(s.split())

def match_llm_chunks_to_mfa(llm_chunks, mfa_words):
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    matched = []

    for chunk in llm_chunks:
        tokens = normalize_text(chunk).split()
        matched_indices = []

        for t in tokens:
            for i, w in enumerate(mfa_tokens):
                if w == t:
                    matched_indices.append(i)
                    break

        if not matched_indices:
            matched.append({"chunk": chunk, "start": None, "end": None})
            continue

        start = min(mfa_words[i]["start"] for i in matched_indices)
        end   = max(mfa_words[i]["end"] for i in matched_indices)

        matched.append({"chunk": chunk, "start": start, "end": end})

    return matched


############################################################
# 6. 按 1 秒区间 emit（保守策略）
############################################################

def assign_chunks_by_second(aligned_chunks):
    max_time = max(x["end"] for x in aligned_chunks if x["end"] is not None)
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


############################################################
# 7. 构造最终 streaming source/target 输出
############################################################

def build_final_segments(timeline, segmentation_json):
    eng_chunks = segmentation_json["low_latency"]["English"]
    zh_chunks  = segmentation_json["low_latency"]["Chinese"]

    eng2zh = {e: z for e, z in zip(eng_chunks, zh_chunks)}

    source_segments = []
    target_segments = []

    for entry in timeline:
        eng_list = entry["emit"]

        # 英文加空格拼接
        eng_concat = " ".join(eng_list).strip()

        # 中文不加空格拼接
        zh_concat = "".join(eng2zh.get(e, "") for e in eng_list).strip()

        source_segments.append(eng_concat)
        target_segments.append(zh_concat)

    return {"source": source_segments, "target": target_segments}


############################################################
# 8. main
############################################################

def main():
    print("\n=== Step 1: Export corpus ===")
    export_corpus(PARQUET_PATH, MFA_CORPUS_DIR, num_samples=10)

    print("\n=== Step 2: Load MFA alignment ===")
    tg_path = os.path.join(MFA_OUTPUT_DIR, "utt_0.TextGrid")
    words = load_word_alignment(tg_path)
    print("Loaded words:", words[:10])

    print("\n=== Step 3: LLM segmentation JSON ===")
    with open(LLM_JSON_PATH, "r", encoding="utf-8") as f:
        segmentation_json = json.load(f)

    llm_chunks = segmentation_json["low_latency"]["English"]

    print("\n=== Step 4: Match LLM chunks → MFA timestamps ===")
    aligned_chunks = match_llm_chunks_to_mfa(llm_chunks, words)
    for a in aligned_chunks:
        print(a)

    print("\n=== Step 5: Build timeline ===")
    timeline = assign_chunks_by_second(aligned_chunks)
    print(timeline)

    print("\n=== Step 6: Build final source/target ===")
    final_output = build_final_segments(timeline, segmentation_json)
    print("\nFinal streaming output:\n", final_output)

    print("\n🎉 DONE.")


if __name__ == "__main__":
    main()
