#!/usr/bin/env python3
"""Build request JSONL from a source script and submit an OpenAI batch job.

Reuses create_request_data from the repo's create_batch.py verbatim so the
requests are byte-identical to the authors' (model snapshot, prompt, sampling
params, custom_id = md5 of sentence). Dedups like the original but in stable
first-occurrence order.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo", "scripts"))
from create_batch import create_request_data  # noqa: E402

from openai import OpenAI  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Only submit the first N unique sentences (pilot)")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--description", default="Simul-tst-COMMON en-zh rebuild")
    ap.add_argument("--dry-run", action="store_true", help="Write JSONL but do not submit")
    args = ap.parse_args()

    seen = set()
    sentences = []
    with open(args.source, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and text not in seen:
                seen.add(text)
                sentences.append(text)
    if args.limit:
        sentences = sentences[: args.limit]

    with open(args.jsonl_out, "w", encoding="utf-8") as f:
        for text in sentences:
            f.write(json.dumps(create_request_data(text, "Chinese"), ensure_ascii=False) + "\n")
    print(f"wrote {len(sentences)} requests to {args.jsonl_out}")

    if args.dry_run:
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(args.jsonl_out, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.description},
    )
    print(f"batch_id={batch.id} status={batch.status}")
    with open(args.jsonl_out + ".batch_id", "w") as f:
        f.write(batch.id + "\n")


if __name__ == "__main__":
    main()
