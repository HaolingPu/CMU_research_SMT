#!/usr/bin/env python3
"""Map OpenAI batch output back onto source-line order.

create_batch.py dedups sentences into a set (nondeterministic order) and keys
each request by custom_id = md5(sentence). Here we walk the source script in
its canonical line order and emit one translation line per source line, so the
result is 1:1 aligned with the source for eval. Duplicate source lines share
one translation.
"""
import argparse
import hashlib
import json


def parse_content(content):
    return json.loads(content.replace("\n", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Corrected source script (one sentence per line)")
    ap.add_argument("--batch-output", required=True, nargs="+", help="OpenAI batch output JSONL file(s)")
    ap.add_argument("--out-prefix", required=True, help="Prefix for .zh / sep.en / sep.zh outputs")
    args = ap.parse_args()

    by_id = {}
    failures = 0
    for path in args.batch_output:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec["custom_id"]
                try:
                    content = rec["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = parse_content(content)
                    pairs = parsed["segmented_pairs"]
                    by_id[cid] = {
                        "output": parsed["output"],
                        "sep_en": " / ".join(str(p[0]) for p in pairs),
                        "sep_zh": " / ".join(str(p[1]) for p in pairs),
                    }
                except Exception as e:
                    failures += 1
                    print(f"unparseable response for {cid}: {e}")

    out, sep_en, sep_zh, missing = [], [], [], []
    with open(args.source, encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            cid = hashlib.md5(text.encode("utf-8")).hexdigest()
            rec = by_id.get(cid)
            if rec is None:
                missing.append((i, text))
                out.append("")
                sep_en.append("")
                sep_zh.append("")
            else:
                out.append(rec["output"])
                sep_en.append(rec["sep_en"])
                sep_zh.append(rec["sep_zh"])

    for suffix, lines in [(".zh", out), (".sep.en", sep_en), (".sep.zh", sep_zh)]:
        with open(args.out_prefix + suffix, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"source lines: {len(out)}, unique translations: {len(by_id)}, "
          f"unparseable: {failures}, missing: {len(missing)}")
    for i, text in missing[:10]:
        print(f"  MISSING line {i}: {text[:80]}")


if __name__ == "__main__":
    main()
