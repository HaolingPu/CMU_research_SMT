import os
import json
import tgt


############################################################
# 小工具：递归收集所有某后缀文件
############################################################
def collect_files_with_ext(root_dir, ext):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(ext):
                full = os.path.join(dirpath, fn)
                paths.append(full)
    return sorted(paths)


############################################################
# 1. Check LLM segmentation JSON  （递归版）
############################################################
def check_llm_segmentation_json(root_dir):
    """
    Return dict: { basename → { status, reasons, English_lengths, Chinese_lengths } }

    basename 例如: utt_en000_00000000_0000
    """
    results = {}

    json_paths = collect_files_with_ext(root_dir, ".json")

    for full_path in json_paths:
        fn = os.path.basename(full_path)

        try:
            data = json.load(open(full_path, "r", encoding="utf-8"))
        except Exception as e:
            results[fn] = {
                "status": "bad",
                "reasons": [f"JSON load error: {str(e)}"],
                "lengths": {}
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
                reasons.append(
                    f"{level}: English len {len(en)} != Chinese len {len(zh)}"
                )

        results[fn] = {
            "status": status,
            "reasons": reasons,
            "lengths": lengths
        }

    return results


############################################################
# 2. Check MFA TextGrid (<unk> filter)  （递归版）
############################################################
def check_mfa_textgrids(root_dir):
    """
    Return dict: { basename → { status, reason } }

    basename 例如: utt_en000_00000000_0000
    """
    results = {}

    tg_paths = collect_files_with_ext(root_dir, ".TextGrid")

    for full_path in tg_paths:
        fn = os.path.basename(full_path)

        try:
            tg = tgt.read_textgrid(full_path)
        except Exception:
            results[fn] = {"status": "bad", "reason": "cannot open"}
            continue

        words = tg.get_tier_by_name("words").intervals
        has_unk = any((w.text.strip() == "<unk>" for w in words))

        if has_unk:
            results[fn] = {"status": "bad", "reason": "<unk> present"}
        else:
            results[fn] = {"status": "good"}

    return results


############################################################
# 3. Merge both filters & save final good/bad jsonl
############################################################
def merge_filters(llm_root, mfa_root, save_good="good.jsonl", save_bad="bad.jsonl"):
    # 清空旧文件
    open(save_good, "w").close()
    open(save_bad, "w").close()

    # Step 1: get LLM check results
    llm_results = check_llm_segmentation_json(llm_root)

    # Step 2: get MFA check results
    mfa_results = check_mfa_textgrids(mfa_root)

    good = []
    bad = []

    # 文件名统一成不带后缀的 basename，例如 utt_en000_00000000_0000
    all_keys = set([fn.replace(".json", "") for fn in llm_results.keys()]) | \
               set([fn.replace(".TextGrid", "") for fn in mfa_results.keys()])

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

        # Save to list（保持原来结构：只存 base；你后面会做 .replace(".TextGrid","")）
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
    llm_root="/data/user_data/haolingp/outputs/llm_segmentation_json/en000",
    mfa_root="/data/user_data/haolingp/outputs/mfa_textgrid_output/en000",
    save_good="/data/user_data/haolingp/outputs/good_en000_all.jsonl",
    save_bad="/data/user_data/haolingp/outputs/bad_en000_all.jsonl"
)