import os
import json

def check_llm_segmentation_json(dir_path, save_bad="bad.jsonl", save_good="good.jsonl"):
    """
    dir_path: folder containing utt_XXXXXX.json (LLM segmentation outputs)

    Each JSON is expected to contain:
    {
        "low_latency": {
            "English": [...],
            "Chinese": [...]
        },
        "medium_latency": {...},
        "high_latency": {...}
    }
    """
    bad_cases = []
    good_cases = []

    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])

    for fn in files:
        full_path = os.path.join(dir_path, fn)

        try:
            data = json.load(open(full_path))
        except Exception as e:
            bad_cases.append({
                "file": fn,
                "reason": f"JSON load error: {str(e)}"
            })
            continue

        status = "good"
        reasons = []

        # 三个 level：low, medium, high 都检查
        for level in ["low_latency", "medium_latency", "high_latency"]:
            if level not in data:
                status = "bad"
                reasons.append(f"missing {level}")
                continue

            en = data[level].get("English", [])
            zh = data[level].get("Chinese", [])

            if not isinstance(en, list) or not isinstance(zh, list):
                status = "bad"
                reasons.append(f"{level}: English/Chinese not list")
                continue

            if len(en) != len(zh):
                status = "bad"
                reasons.append(
                    f"{level}: English len {len(en)} != Chinese len {len(zh)}"
                )

        if status == "bad":
            bad_cases.append({"file": fn, "reasons": reasons})
        else:
            good_cases.append(fn)

    # 保存结果
    with open(save_bad, "w") as f:
        for item in bad_cases:
            f.write(json.dumps(item) + "\n")

    with open(save_good, "w") as f:
        for item in good_cases:
            f.write(json.dumps({"file": item}) + "\n")

    print("===== SUMMARY =====")
    print("Total:", len(files))
    print("Good :", len(good_cases))
    print("Bad  :", len(bad_cases))
    print("===================")

    return good_cases, bad_cases







import tgt

def filter_unk_textgrids(mfa_dir, save_good="good_list.jsonl", save_bad="bad_list.jsonl"):
    good = []
    bad = []

    tg_files = sorted([f for f in os.listdir(mfa_dir) if f.endswith(".TextGrid")])

    for tg_file in tg_files:
        path = os.path.join(mfa_dir, tg_file)

        try:
            tg = tgt.read_textgrid(path)
        except:
            bad.append({"file": tg_file, "reason": "cannot open"})
            continue

        words = tg.get_tier_by_name("words").intervals

        has_unk = any([w.text.strip() == "<unk>" for w in words])

        if has_unk:
            bad.append({"file": tg_file, "reason": "<unk> present"})
        else:
            good.append({"file": tg_file})

    # 保存结果
    with open(save_good, "w") as f:
        for item in good:
            f.write(json.dumps(item) + "\n")

    with open(save_bad, "w") as f:
        for item in bad:
            f.write(json.dumps(item) + "\n")

    print("===== SUMMARY =====")
    print("Total:", len(tg_files))
    print("Good :", len(good))
    print("Bad  :", len(bad))
    print("===================")

    return good, bad





bad_path = "/data/user_data/haolingp/outputs/bad.jsonl"
good_path = "/data/user_data/haolingp/outputs/good.jsonl"
filter_unk_textgrids("/data/user_data/haolingp/outputs/mfa_output",good_path, bad_path)
