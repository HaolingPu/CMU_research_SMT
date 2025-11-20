import os

def rename_mfa_files(mfa_dir):
    files = sorted([
        f for f in os.listdir(mfa_dir)
        if f.endswith(".TextGrid") and f.startswith("utt_")
    ])

    for f in files:
        # 当前格式如：utt_0.TextGrid 或 utt_13.TextGrid
        base = f.replace(".TextGrid", "")   # utt_0
        parts = base.split("_")            # ["utt", "0"]

        if len(parts) != 2:
            continue

        idx = int(parts[1])                # 0 → int

        # 统一格式：6-digit padding
        new_name = f"utt_{idx:06d}.TextGrid"
        old_path = os.path.join(mfa_dir, f)
        new_path = os.path.join(mfa_dir, new_name)

        print(f"{f}  →  {new_name}")
        os.rename(old_path, new_path)

    print("Rename complete.")


rename_mfa_files("/data/user_data/haolingp/outputs/mfa_output")