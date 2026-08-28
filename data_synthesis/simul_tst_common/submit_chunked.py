#!/usr/bin/env python3
"""Submit the full request set as sequential Batch-API chunks under a Tier-1
90k enqueued-token limit.

Enqueued tokens per request = prompt tokens + max_tokens, so we (a) lower
max_tokens per request adaptively (truncation-only; top_p=0 greedy output is
unaffected), and (b) pack chunks to <= BUDGET. Chunks run one at a time.
Completed chunk outputs are saved as chunk_NNN.output.jsonl; already-saved
chunks are skipped, so the script is resumable.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo", "scripts"))
from create_batch import create_request_data  # noqa: E402

import tiktoken  # noqa: E402
from openai import OpenAI  # noqa: E402

SOURCE = "Simul-tst-COMMON.en"
OUTDIR = "batch/chunks"
BUDGET = 85_000
ENC = tiktoken.get_encoding("o200k_base")


def build_requests():
    seen, reqs = set(), []
    for line in open(SOURCE, encoding="utf-8"):
        text = line.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        req = create_request_data(text, "Chinese")
        n_words = len(text.split())
        req["body"]["max_tokens"] = min(1200, max(200, 8 * n_words + 80))
        prompt_tokens = sum(len(ENC.encode(m["content"])) for m in req["body"]["messages"]) + 20
        reqs.append((req, prompt_tokens + req["body"]["max_tokens"]))
    return reqs


def chunk(reqs):
    chunks, cur, cur_cost = [], [], 0
    for req, cost in reqs:
        if cur and cur_cost + cost > BUDGET:
            chunks.append(cur)
            cur, cur_cost = [], 0
        cur.append(req)
        cur_cost += cost
    if cur:
        chunks.append(cur)
    return chunks


def run_chunk(client, i, requests):
    in_path = f"{OUTDIR}/chunk_{i:03d}.jsonl"
    out_path = f"{OUTDIR}/chunk_{i:03d}.output.jsonl"
    if os.path.exists(out_path):
        print(f"[chunk {i}] already done, skipping", flush=True)
        return
    with open(in_path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for attempt in range(10):
        with open(in_path, "rb") as f:
            bf = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=bf.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Simul-tst-COMMON en-zh chunk {i}"},
        )
        print(f"[chunk {i}] submitted {batch.id} ({len(requests)} reqs, attempt {attempt})", flush=True)
        while True:
            time.sleep(60)
            b = client.batches.retrieve(batch.id)
            if b.status in ("completed", "failed", "expired", "cancelled"):
                break
        if b.status == "completed" and b.output_file_id:
            content = client.files.content(b.output_file_id)
            with open(out_path, "wb") as f:
                f.write(content.read())
            rc = b.request_counts
            print(f"[chunk {i}] completed {rc.completed}/{rc.total} failed={rc.failed}", flush=True)
            if b.error_file_id:
                err = client.files.content(b.error_file_id)
                with open(out_path + ".errors", "wb") as f:
                    f.write(err.read())
            return
        errs = b.errors.data if b.errors else []
        if any(e.code == "token_limit_exceeded" for e in errs):
            print(f"[chunk {i}] queue full, waiting 5 min", flush=True)
            time.sleep(300)
            continue
        raise RuntimeError(f"chunk {i} terminal status {b.status}: {errs}")
    raise RuntimeError(f"chunk {i} exhausted retries")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    reqs = build_requests()
    chunks = chunk(reqs)
    total = sum(len(c) for c in chunks)
    print(f"{total} requests in {len(chunks)} chunks", flush=True)
    for i, c in enumerate(chunks):
        run_chunk(client, i, c)
    print("ALL CHUNKS DONE", flush=True)


if __name__ == "__main__":
    main()
