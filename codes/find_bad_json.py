import os
import json
import tgt


############################################################
# Collect files in a single parquet directory
############################################################
def collect_files_single_parquet(parquet_dir, ext):
    """
    Collect files from a single parquet directory.
    Returns: {filename_base: full_path}
    """
    mapping = {}
    
    if not os.path.exists(parquet_dir):
        return mapping
    
    for fn in os.listdir(parquet_dir):
        if fn.endswith(ext):
            full = os.path.join(parquet_dir, fn)
            base = fn[:-len(ext)]  # remove extension
            mapping[base] = full
    
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
# Check MFA TextGrid (<unk>)
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
# Process all parquets for a language
############################################################
def process_language(llm_root, mfa_root, save_good, save_bad, lang):
    """
    Process all parquet directories for a language
    Output to 2 files (all good in one, all bad in one)
    """
    print(f"\n===== Processing {lang} =====")
    
    # Get all parquet directories
    lang_llm_dir = os.path.join(llm_root, lang)
    lang_mfa_dir = os.path.join(mfa_root, lang)
    
    if not os.path.exists(lang_llm_dir):
        print(f"ERROR: LLM directory not found: {lang_llm_dir}")
        return
    
    # Get list of parquet directories (8-digit folders)
    parquet_dirs = sorted([
        d for d in os.listdir(lang_llm_dir)
        if os.path.isdir(os.path.join(lang_llm_dir, d)) and d.isdigit() and len(d) == 8
    ])
    
    print(f"Found {len(parquet_dirs)} parquet directories")
    
    # Clear output files
    open(save_good, "w").close()
    open(save_bad, "w").close()
    
    total_good = 0
    total_bad = 0
    
    # Process each parquet
    for parquet_name in parquet_dirs:
        llm_parquet_dir = os.path.join(lang_llm_dir, parquet_name)
        mfa_parquet_dir = os.path.join(lang_mfa_dir, parquet_name)
        
        # Collect files in this parquet
        llm_map = collect_files_single_parquet(llm_parquet_dir, ".json")
        mfa_map = collect_files_single_parquet(mfa_parquet_dir, ".TextGrid")
        
        all_keys = sorted(set(llm_map.keys()) | set(mfa_map.keys()))
        
        parquet_good = 0
        parquet_bad = 0
        
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

            # Save to file
            item = {"file": key}
            if ok:
                with open(save_good, "a") as f:
                    f.write(json.dumps(item) + "\n")
                parquet_good += 1
                total_good += 1
            else:
                item["reasons"] = reasons
                with open(save_bad, "a") as f:
                    f.write(json.dumps(item) + "\n")
                parquet_bad += 1
                total_bad += 1
        
        print(f"  [{parquet_name}] Good: {parquet_good}, Bad: {parquet_bad}")
    
    print(f"\n===== {lang} SUMMARY =====")
    print(f"Total Good: {total_good}")
    print(f"Total Bad : {total_bad}")
    print(f"Total     : {total_good + total_bad}")
    print("=" * 30)


############################################################
# Main
############################################################
if __name__ == "__main__":
    # Output files (just 2 files)
    output_dir = "/data/user_data/haolingp/outputs"
    
    # Process en000
    process_language(
        llm_root="/data/user_data/haolingp/outputs/llm_output",
        mfa_root="/data/user_data/haolingp/outputs/mfa_textgrid_output",
        save_good=f"{output_dir}/good_en000_all.jsonl",
        save_bad=f"{output_dir}/bad_en000_all.jsonl",
        lang="en000"
    )
    
    print("\n✅ All quality checks completed!")
    print(f"Results saved:")
    print(f"  - {output_dir}/good_en000_all.jsonl")
    print(f"  - {output_dir}/bad_en000_all.jsonl")