from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

configs = ["en_librilight", "en_voxpopuli", "en_yodas"]
target_hours = 1000
target_seconds = target_hours * 3600

rows = []
total = 0

for cfg in configs:
    print(f"🔹 Loading {cfg}")
    ds = load_dataset("nvidia/Granary", cfg, split="asr", streaming=True)
    for ex in tqdm(ds, desc=f"Streaming {cfg}"):
        dur = ex.get("duration", 0)
        if dur and dur > 0:
            rows.append({
                "audio": ex["audio_filepath"],
                "text": ex["text"],
                "duration": dur,
                "source": cfg
            })
            total += dur
            if total >= target_seconds:
                break
    if total >= target_seconds:
        break

print(f"✅ Collected {total/3600:.1f} hours")
df = pd.DataFrame(rows)
df.to_csv("/data/user_data/haolingp/datasets/granary_en_1000h_manifest.tsv", sep="\t", index=False)
print("💾 Saved to /data/user_data/haolingp/datasets/granary_en_1000h_manifest.tsv")
