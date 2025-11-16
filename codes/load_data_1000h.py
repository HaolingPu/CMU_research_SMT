#!/usr/bin/env python3
"""
重建 Granary English 1000h manifest（可选下载音频到本地）
数据源：nvidia/Granary（configs: en_librilight, en_voxpopuli, en_yodas）
用法：
  仅生成清单：
    python build_manifest.py --hours 1000 --out-dir /data/user_data/haolingp/datasets --manifest-only
  生成清单并把音频落地为 wav：
    python build_manifest.py --hours 1000 --out-dir /data/user_data/haolingp/datasets --dump-audio --audio-dir granary_audio
"""

from datasets import load_dataset, Audio
from tqdm import tqdm
import pandas as pd
import os, sys, argparse, time
import numpy as np
import soundfile as sf

CONFIGS = ["en_librilight", "en_voxpopuli", "en_yodas"]

def safe_write_wav(wav_path, array, sr):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    # 单声道 & float32
    if array.ndim > 1:
        array = np.mean(array, axis=1)
    sf.write(wav_path, array.astype(np.float32), sr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1000, help="目标小时数")
    ap.add_argument("--out-dir", type=str, required=True, help="manifest 输出目录")
    ap.add_argument("--manifest-name", type=str, default="granary_en_1000h_manifest.tsv")
    ap.add_argument("--manifest-only", action="store_true", help="仅生成清单，不落地音频")
    ap.add_argument("--dump-audio", action="store_true", help="按清单把音频落地为本地 wav")
    ap.add_argument("--audio-dir", type=str, default="granary_audio", help="本地音频目录（dump 模式）")
    ap.add_argument("--target-sr", type=int, default=16000, help="目标采样率（MFA 友好）")
    args = ap.parse_args()

    target_seconds = int(args.hours * 3600)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, args.manifest_name)
    rows = []
    total = 0.0
    n_kept = 0

    print(f"🎯 目标 {args.hours} h  = {target_seconds} s")
    print(f"📁 输出 manifest: {manifest_path}")
    if args.dump_audio:
        print(f"💾 本地音频目录: {args.audio_dir} (会断点续下/跳过已存在)")

    # 逐个 config 叠加
    for cfg in CONFIGS:
        if total >= target_seconds:
            break
        print(f"\n🔹 Loading nvidia/Granary :: {cfg} (split=asr, streaming=True)")
        ds = load_dataset("nvidia/Granary", cfg, split="asr", streaming=True)

        # 如果要落地音频，先声明音频解码器（datasets 的 lazy decode）
        audio_decoder = Audio(sampling_rate=args.target_sr) if args.dump_audio else None

        for ex in tqdm(ds, desc=f"Streaming {cfg}"):
            dur = ex.get("duration", 0.0) or 0.0
            if dur <= 0:
                continue

            utt_id = ex.get("utt_id") or os.path.splitext(os.path.basename(ex.get("audio_filepath","")))[0] or f"{cfg}_{n_kept:08d}"
            text = (ex.get("text") or "").strip()
            remote_audio = ex.get("audio_filepath")  # 远端 URL/路径（不一定是 http）
            lang = ex.get("lang", "en")

            item = {
                "utt_id": utt_id,
                "text": text,
                "duration": float(dur),
                "lang": lang,
                "source": cfg,
            }

            if args.dump_audio:
                # 用 datasets 的 Audio 解码器把远端音频读进来（兼容多源）
                try:
                    # 注意：部分样本可能没有内嵌二进制，需要 datasets 能处理 remote file；失败就跳过
                    ex_audio = ex["audio"] if "audio" in ex else None
                    if ex_audio is None:
                        # 回退：有些条目只有路径，无法直接解码，跳过
                        continue
                    # 统一到目标采样率
                    ex_cast = {"audio": ex_audio}
                    ex_cast = Audio(sampling_rate=args.target_sr).decode_example(ex_cast)
                    array = ex_cast["audio"]["array"]
                    sr = ex_cast["audio"]["sampling_rate"]

                    # 写本地 wav
                    wav_path = os.path.join(args.audio_dir, f"{utt_id}.wav")
                    if not os.path.exists(wav_path):
                        safe_write_wav(wav_path, array, sr)

                    item["audio_path"] = os.path.abspath(wav_path)

                except Exception as e:
                    # 有些链接/格式读不到，直接跳
                    # 你也可以在这里接 ffmpeg 简化解码，但为了稳妥先跳过问题样本
                    continue
            else:
                # manifest-only：保存远端路径，后续再批量下载/转码
                item["audio_path"] = remote_audio

            rows.append(item)
            n_kept += 1
            total += dur

            if n_kept % 200 == 0:
                print(f"\r已收集 {n_kept} 条，累计 {total/3600:.2f} h", end="")

            if total >= target_seconds:
                break

    # 写 manifest
    if not rows:
        print("❌ 没有收集到样本，检查配置或网络。"); sys.exit(1)

    df = pd.DataFrame(rows)
    # 列顺序更友好
    cols = ["utt_id","audio_path","text","duration","lang","source"]
    df = df[cols]
    df.to_csv(manifest_path, sep="\t", index=False)

    print("\n" + "="*70)
    print(f"✅ 完成：{len(df)} 条，{total/3600:.2f} 小时")
    print(f"📄 Manifest: {manifest_path}")
    if args.dump_audio:
        print(f"🎧 本地音频目录: {os.path.abspath(args.audio_dir)}")
    print("="*70)

if __name__ == "__main__":
    main()
