#!/usr/bin/env python3
"""Memory-frugal MetricX-24 QE predict wrapper.

Same I/O as `python -m metricx24.predict`, but loads with low_cpu_mem_usage
+ fp16 so the cgroup-limited debug node doesn't OOM. Output schema matches.
"""
import argparse
import json
import os

import datasets
import torch
import transformers
from metricx24 import models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--max_input_length", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--qe", action="store_true")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16")
    return p.parse_args()


def get_dataset(input_file, tokenizer, max_input_length, device, is_qe):
    def _make_input(ex):
        if is_qe:
            ex["input"] = (
                "source: " + ex["source"]
                + " candidate: " + ex["hypothesis"]
            )
        else:
            ex["input"] = (
                "source: " + ex["source"]
                + " candidate: " + ex["hypothesis"]
                + " reference: " + ex["reference"]
            )
        return ex

    def _tokenize(ex):
        return tokenizer(
            ex["input"], max_length=max_input_length,
            truncation=True, padding=False,
        )

    def _remove_eos(ex):
        ex["input_ids"] = ex["input_ids"][:-1]
        ex["attention_mask"] = ex["attention_mask"][:-1]
        return ex

    ds = datasets.load_dataset("json", data_files={"test": input_file})
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)
    ds = ds.with_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
    return ds


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_dev_bs = args.batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_dev_bs = args.batch_size

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[load] tokenizer={args.tokenizer}", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[load] model={args.model_name_or_path} dtype={args.dtype} "
          f"low_cpu_mem_usage=True", flush=True)
    model = models.MT5ForRegression.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    ds = get_dataset(args.input_file, tokenizer, args.max_input_length,
                     device, args.qe)
    print(f"[data] {len(ds['test'])} examples", flush=True)

    training_args = transformers.TrainingArguments(
        output_dir=os.path.dirname(args.output_file) or ".",
        per_device_eval_batch_size=per_dev_bs,
        dataloader_pin_memory=False,
        report_to=[],
    )
    trainer = transformers.Trainer(model=model, args=training_args)
    predictions, _, _ = trainer.predict(test_dataset=ds["test"])

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
        for pred, ex in zip(predictions, ds["test"]):
            ex["prediction"] = float(pred)
            for k in ("input", "input_ids", "attention_mask"):
                ex.pop(k, None)
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[done] wrote {args.output_file}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
