import os
import json
from typing import List, Dict

import torch
from transformers import AutoTokenizer
import sys
sys.path.append("/data/user_data/haolingp/codes/metricx")

from metricx24 import models


# ==== 路径配置：我直接填了你现在用的那些 ====
TOKENIZER_PATH = "/data/user_data/haolingp/models/mt5-xl"
MODEL_PATH = "/data/user_data/haolingp/models/metricx-24-hybrid-xl-v2p6"

INPUT_JSONL = "/data/user_data/haolingp/outputs/metricx_input.jsonl"
OUTPUT_JSONL = "/data/user_data/haolingp/outputs/metricx_output.jsonl"

MAX_INPUT_LENGTH = 1536
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_metricx_qe_model():
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print(f"Loading MetricX-24 QE model from: {MODEL_PATH}")
    model = models.MT5ForRegression.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    print("Model loaded and ready.\n")
    return tokenizer, model


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {path}")
    return data


def metricx_qe_score_single(model, tokenizer, source: str, hyp: str) -> float:
    """
    按 MetricX-24 QE 说明书构造输入：
    input = "source: {source} candidate: {hypothesis}"
    """
    text = f"source: {source} candidate: {hyp}"

    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)  # MT5ForRegressionOutput(predictions=...)
        # predictions 已经是 [batch] 的实数分数，范围 [0, 25]
        score = outputs.predictions.squeeze().item()

    return float(score)


def main():
    print("=== MetricX-24 QE (no datasets/pyarrow) ===")

    tokenizer, model = load_metricx_qe_model()
    data = read_jsonl(INPUT_JSONL)

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    out_f = open(OUTPUT_JSONL, "w")

    num_ok = 0
    num_err = 0

    for i, ex in enumerate(data):
        try:
            src = ex["source"]
            hyp = ex["hypothesis"]
        except KeyError as e:
            print(f"[WARN] Example {i} missing key {e}, skipping.")
            num_err += 1
            continue

        try:
            score = metricx_qe_score_single(model, tokenizer, src, hyp)
        except Exception as e:
            print(f"[ERROR] Scoring example {i} failed: {e}")
            num_err += 1
            continue

        # 把分数写回 JSON
        ex_out = dict(ex)
        ex_out["prediction"] = score
        out_f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
        num_ok += 1

        if num_ok % 10 == 0:
            print(f"  Scored {num_ok}/{len(data)} examples...")

    out_f.close()
    print(f"\nDone. Success: {num_ok}, Failed: {num_err}")
    print(f"Output written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
