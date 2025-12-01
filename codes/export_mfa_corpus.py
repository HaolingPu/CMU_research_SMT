import os
import re
import io
import argparse
import pandas as pd
import soundfile as sf


# ============================================================
# Text cleaner
# ============================================================
def clean_text(t: str):
    t = t.lower()
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# Process SINGLE parquet
# ============================================================
def process_parquet(parquet_path, out_base_dir, lang, num_samples=None):
    parquet_name = os.path.splitext(os.path.basename(parquet_path))[0]

    print(f"\n=== Processing parquet: {parquet_name} ===")

    # Output directory example: /mfa_corpus/en000/00000000/
    out_dir = os.path.join(out_base_dir, lang, parquet_name)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    if num_samples == "all":
        num_samples = None
    else:
        num_samples = int(num_samples)
        df = df.iloc[:num_samples]

    for idx, row in df.iterrows():
        # Audio bytes
        audio_bytes = row["audio"]["bytes"]
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file)

        # Text
        text = clean_text(row["text"])

        # uttid
        uttid = f"utt_{lang}_{parquet_name}_{idx:04d}"

        wav_path = os.path.join(out_dir, f"{uttid}.wav")
        lab_path = os.path.join(out_dir, f"{uttid}.lab")

        # Write wav
        sf.write(wav_path, audio, sr)

        # Write lab
        with open(lab_path, "w") as f:
            f.write(text)

        print(f"  → wrote {wav_path}")

    print(f"Done parquet {parquet_name}. Files saved in {out_dir}")


# ============================================================
# Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-root", type=str, required=True,
                        help="Root path of dataset: .../yodas-granary/data/")

    parser.add_argument("--lang", type=str, required=True,
                        help="Language folder, e.g. en000")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output base directory for MFA corpus")

    parser.add_argument("--num-parquets", type=str, default="all",
                        help="How many parquets to process: 1,2,3 or 'all'")

    parser.add_argument("--num-samples", type=str, default="all",
                        help="Number of samples per parquet (default = all)")

    args = parser.parse_args()

    # Input parquet dir:
    parquet_dir = os.path.join(args.input_root, args.lang, "asr_only")

    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    if args.num_parquets != "all":
        num = int(args.num_parquets)
        parquet_files = parquet_files[:num]

    print(f"Found {len(parquet_files)} parquets to process.")

    for pq in parquet_files:
        pq_path = os.path.join(parquet_dir, pq)
        process_parquet(
            pq_path,
            args.output_dir,
            args.lang,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
