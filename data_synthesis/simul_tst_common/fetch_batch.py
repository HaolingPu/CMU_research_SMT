#!/usr/bin/env python3
"""Check an OpenAI batch job; download output when complete."""
import argparse
import os

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(args.batch_id)
    counts = batch.request_counts
    print(f"status={batch.status} completed={counts.completed}/{counts.total} failed={counts.failed}")

    if batch.status == "completed" and batch.output_file_id:
        content = client.files.content(batch.output_file_id)
        with open(args.output, "wb") as f:
            f.write(content.read())
        print(f"saved {args.output}")
        if batch.error_file_id:
            err = client.files.content(batch.error_file_id)
            with open(args.output + ".errors", "wb") as f:
                f.write(err.read())
            print(f"saved {args.output}.errors")


if __name__ == "__main__":
    main()
