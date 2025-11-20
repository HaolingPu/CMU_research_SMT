import os
import json
import tgt


############################################################
# 1. Check LLM segmentation JSON
############################################################

def check_llm_segmentation_json(dir_path):
    """
    Return dict: { filename → { status, reasons, English_lengths, Chinese_lengths } }
    """
    results = {}

    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])

    for fn in files:
        full_path = os.path.join(dir_path, fn)

        try:
            data = json.load(open(full_path))
        except Exception as e:
            results[fn] = {
                "status": "bad",
                "reasons": [f"JSON load error: {str(e)}"]
            }
            continue

        status = "good"
        reasons = []
        lengths = {}

        for level in ["low_latency", "medium_latency", "high_latency"]:
            if level not in data:
                status = "bad"
                reasons.append(f"missing {level}")
                continue

            en = data[level].get("English", [])
            zh = data[level].get("Chinese", [])

            lengths[level] = (len(en), len(zh))

            # type check
            if not isinstance(en, list) or not isinstance(zh, list):
                status = "bad"
                reasons.append(f"{level}: English/Chinese not list")
                continue

            # length mismatch
            if len(en) != len(zh):
                status = "bad"
                reasons.append(f"{level}: English len {len(en)} != Chinese len {len(zh)}")

        results[fn] = {
            "status": status,
            "reasons": reasons,
            "lengths": lengths
        }

    return results


############################################################
# 2. Check MFA TextGrid (<unk> filter)
############################################################

def check_mfa_textgrids(mfa_dir):
    """
    Return dict: { filename → { status, reason } }
    """
    results = {}

    tg_files = sorted([f for f in os.listdir(mfa_dir) if f.endswith(".TextGrid")])

    for tg_file in tg_files:
        path = os.path.join(mfa_dir, tg_file)

        try:
            tg = tgt.read_textgrid(path)
        except:
            results[tg_file] = {"status": "bad", "reason": "cannot open"}
            continue

        words = tg.get_tier_by_name("words").intervals

        has_unk = any([w.text.strip() == "<unk>" for w in words])

        if has_unk:
            results[tg_file] = {"status": "bad", "reason": "<unk> present"}
        else:
            results[tg_file] = {"status": "good"}

    return results


############################################################
# 3. Merge both filters & save final good/bad jsonl
############################################################

def merge_filters(llm_dir, mfa_dir, save_good="good.jsonl", save_bad="bad.jsonl"):
    # Always clear old logs before writing
    open(save_good, "w").close()
    open(save_bad, "w").close()
    
    # Step 1: get LLM check results
    llm_results = check_llm_segmentation_json(llm_dir)

    # Step 2: get MFA check results
    mfa_results = check_mfa_textgrids(mfa_dir)

    good = []
    bad = []

    # ⚠️ Assume filenames are like: utt_000123.json and utt_000123.TextGrid
    # Extract base name
    all_keys = set([fn.replace(".json", "") for fn in llm_results]) | \
               set([fn.replace(".TextGrid", "") for fn in mfa_results])

    for base in sorted(all_keys):
        llm_info = llm_results.get(base + ".json", None)
        mfa_info = mfa_results.get(base + ".TextGrid", None)

        is_good = True
        reasons = []

        # LLM check
        if llm_info is None:
            is_good = False
            reasons.append("missing LLM JSON")
        else:
            if llm_info["status"] == "bad":
                is_good = False
                reasons += llm_info["reasons"]

        # MFA check
        if mfa_info is None:
            is_good = False
            reasons.append("missing TextGrid")
        else:
            if mfa_info["status"] == "bad":
                is_good = False
                reasons.append(mfa_info["reason"])

        # Save to list
        if is_good:
            good.append({"file": base})
        else:
            bad.append({"file": base, "reasons": reasons})

    # Write output
    with open(save_good, "w") as f:
        for item in good:
            f.write(json.dumps(item) + "\n")

    with open(save_bad, "w") as f:
        for item in bad:
            f.write(json.dumps(item) + "\n")

    print("===== FINAL SUMMARY =====")
    print("Total:", len(all_keys))
    print("Good :", len(good))
    print("Bad  :", len(bad))
    print("=========================")

    return good, bad



merge_filters(
    llm_dir="/data/user_data/haolingp/outputs/llm_segmentation_json",
    mfa_dir="/data/user_data/haolingp/outputs/mfa_output",
    save_good="/data/user_data/haolingp/outputs/good.jsonl",
    save_bad="/data/user_data/haolingp/outputs/bad.jsonl"
)
