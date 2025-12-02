import os
import json
import tgt


############################################################
# Collect all files recursively, return map: key → full_path
############################################################
def collect_files(root, ext, lang):
    mapping = {}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(ext):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)   # e.g. "00000000/utt_..."
                base = rel[: -len(ext)]            # no suffix
                key = f"{lang}/{base}"             # add language prefix
                mapping[key] = full

    return mapping


############################################################
# Check JSON file content
############################################################
def check_llm_json(full_path):
    try:
        data = json.load(open(full_path, "r", encoding="utf-8"))
    except Exception as e:
        return False, [f"JSON load error: {str(e)}"]

    reasons = []
    ok = True

    for level in ["low_latency", "medium_latency", "high_latency"]:
        if level not in data:
            ok = False
            reasons.append(f"missing {level}")
            continue

        en = data[level].get("English", [])
        zh = data[level].get("Chinese", [])

        if not isinstance(en, list) or not isinstance(zh, list):
            ok = False
            reasons.append(f"{level}: English/Chinese not list")
            continue

        if len(en) != len(zh):
            ok = False
            reasons.append(
                f"{level}: English len {len(en)} != Chinese len {len(zh)}"
            )

    return ok, reasons


############################################################
# Check MFATextGrid (<unk>)
############################################################
def check_textgrid(full_path):
    try:
        tg = tgt.read_textgrid(full_path)
    except Exception:
        return False, ["cannot open"]

    words = tg.get_tier_by_name("words").intervals
    if any((w.text.strip() == "<unk>" for w in words)):
        return False, ["<unk> present"]

    return True, []


############################################################
# Final merge
############################################################
def merge_filters(llm_root, mfa_root, save_good, save_bad, lang):
    # Collect recursive file maps
    llm_map = collect_files(llm_root, ".json", lang)
    mfa_map = collect_files(mfa_root, ".TextGrid", lang)

    all_keys = sorted(set(llm_map.keys()) | set(mfa_map.keys()))

    good = []
    bad  = []

    # Clear old outputs
    open(save_good, "w").close()
    open(save_bad, "w").close()

    for key in all_keys:
        reasons = []
        ok = True

        # LLM JSON check
        if key not in llm_map:
            ok = False
            reasons.append("missing LLM JSON")
        else:
            llm_ok, llm_reasons = check_llm_json(llm_map[key])
            if not llm_ok:
                ok = False
                reasons.extend(llm_reasons)

        # TextGrid check
        if key not in mfa_map:
            ok = False
            reasons.append("missing TextGrid")
        else:
            tg_ok, tg_reasons = check_textgrid(mfa_map[key])
            if not tg_ok:
                ok = False
                reasons.extend(tg_reasons)

        # Save
        item = {"file": key}
        if ok:
            with open(save_good, "a") as f:
                f.write(json.dumps(item) + "\n")
        else:
            item["reasons"] = reasons
            with open(save_bad, "a") as f:
                f.write(json.dumps(item) + "\n")

    print("===== SUMMARY =====")
    print("Total:", len(all_keys))
    print("Good :", sum(1 for _ in open(save_good)))
    print("Bad  :", sum(1 for _ in open(save_bad)))
    print("====================")


############################################################
# Run
############################################################

for en in ["en000"]:
    merge_filters(
        llm_root=f"/data/user_data/haolingp/outputs/llm_segmentation_json/{en}",
        mfa_root=f"/data/user_data/haolingp/outputs/mfa_textgrid_output/{en}",
        save_good=f"/data/user_data/haolingp/outputs/good_{en}_all.jsonl",
        save_bad=f"/data/user_data/haolingp/outputs/bad_{en}_all.jsonl",
        lang=en
    )
