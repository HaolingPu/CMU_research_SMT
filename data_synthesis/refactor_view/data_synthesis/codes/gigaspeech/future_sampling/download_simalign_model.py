#!/usr/bin/env python3
"""
把 simalign 用的 LaBSE 模型下载到指定目录（默认 /data/user_data/haolingp/models）。

需要下载的模型: pvl/labse_bert（Hugging Face 上的 LaBSE，用于中英词对齐）。

用法:
  # 下载到默认目录
  python download_simalign_model.py

  # 指定目录
  python download_simalign_model.py --cache-dir /data/user_data/haolingp/models
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Download simalign model (pvl/labse_bert) to local dir.")
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/data/user_data/haolingp/models",
        help="Directory to save the model (will create hub cache underneath).",
    )
    args = p.parse_args()
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Hugging Face / Transformers 会读 HF_HOME 或 TRANSFORMERS_CACHE
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    print(f"[Download] Using cache dir: {cache_dir}")
    print("[Download] Loading SentenceAligner (pvl/labse_bert), first run will download ...")

    from simalign import SentenceAligner

    aligner = SentenceAligner(
        model="pvl/labse_bert",
        token_type="bpe",
        matching_methods="a",
        device="cpu",
    )
    print("[Download] Model loaded successfully.")
    print(f"[Download] Files are under: {cache_dir}")
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
