#!/usr/bin/env python3
"""Estimate GPT-4o reproduction fidelity from a (pilot) batch output.

For each translation line, check whether its sha256 hash appears among the
tgt edits.json line_hashes. The authors edited ~1892 of ~2850 lines (~66%),
so if our GPT outputs byte-match theirs, ~66% of lines should hash-match an
edit op. A much lower rate means the model outputs have drifted.
"""
import argparse
import hashlib
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", required=True, help="one translation per line (restore_order output)")
    ap.add_argument("--edits", required=True)
    args = ap.parse_args()

    pkg = json.load(open(args.edits, encoding="utf-8"))
    edit_hashes = {op["line_hash"] for op in pkg["operations"] if "line_hash" in op}

    lines = [l.rstrip("\r\n") for l in open(args.translations, encoding="utf-8")]
    lines = [l for l in lines if l]
    hits = sum(1 for l in lines if hashlib.sha256(l.encode("utf-8")).hexdigest() in edit_hashes)

    n = len(lines)
    print(f"{hits}/{n} translations hash-match a tgt edit op ({100*hits/n:.1f}%)")
    print("expected ~66% if byte-identical to the authors' GPT run")
    print(f"implied reproduction fidelity ≈ {100*hits/n/0.664:.0f}% (rough)")


if __name__ == "__main__":
    main()
