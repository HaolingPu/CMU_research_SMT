#!/usr/bin/env python3
"""Collect raw v3 (future_set_v3_suffix_icl) sampler responses and test the strict parser on them."""
import argparse, csv, json, re, sys, urllib.request, zlib, ast, traceback
csv.field_size_limit(10**9)
ap = argparse.ArgumentParser()
ap.add_argument("--runtime", required=True); ap.add_argument("--input-tsv", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--cases", type=int, default=12); ap.add_argument("--per-case", type=int, default=4)
ap.add_argument("--models", required=True, help="name=api_base=tokenizer_path;...")
ap.add_argument("--seed", type=int, default=1015); ap.add_argument("--prefix-file", default=""); ap.add_argument("--short-prefixes", type=int, default=0); ap.add_argument("--num-futures", type=int, default=20)
args = ap.parse_args()
sys.path.insert(0, args.runtime)
from ambiguity_sampler_prompt import build_coordinated_future_messages, parse_grouped_future_output, SUFFIX_ICL_PROMPT_VERSION
from transformers import AutoTokenizer
END = re.compile(r"[.!?][\"”’')\]]*\s*$")
def anchored(text):
    m = list(re.finditer(r"[.!?][\"”’')\]]*\s+", text))
    return text[m[-1].end():] if m else text
rows = list(csv.DictReader(open(args.input_tsv, newline="", encoding="utf-8"), delimiter="\t"))
prefixes = []
for row in rows[: args.cases]:
    chunks = ast.literal_eval(row["src_trajectory"]); cum = ""
    cands = []
    for t, c in enumerate(chunks[:-1]):
        cum = (cum + " " + c).strip() if cum else c.strip()
        if END.search(cum): continue
        p = anchored(cum).strip()
        if len(p.split()) >= 1: cands.append((t, p))
    step = max(1, len(cands) // args.per_case)
    for t, p in cands[::step][: args.per_case]: prefixes.append((row["id"], t, p))
if args.prefix_file:
    prefixes = [(f"file{i}", i, p) for i, p in enumerate(json.load(open(args.prefix_file)))]
if args.short_prefixes:
    pool = []
    for row in rows:
        chunks = ast.literal_eval(row["src_trajectory"]); cum = ""
        for t, c in enumerate(chunks[:-1]):
            cum = (cum + " " + c).strip() if cum else c.strip()
            if END.search(cum): continue
            p = anchored(cum).strip()
            if 1 <= len(p.split()) <= 2: pool.append((row["id"], t, p))
    seen = set()
    for item in pool:
        if item[2].lower() in seen: continue
        seen.add(item[2].lower()); prefixes.append(item)
        if len(prefixes) >= args.short_prefixes + (len(json.load(open(args.prefix_file))) if args.prefix_file else 0): break
print(f"{len(prefixes)} prefixes", flush=True)
models = []
for spec in args.models.split(";"):
    name, base, tok = spec.split("=")
    models.append((name, base.rstrip("/"), AutoTokenizer.from_pretrained(tok, trust_remote_code=True)))
def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
n_ok = n_bad = 0
with open(args.out, "w", encoding="utf-8") as out:
    for utt, t, prefix in prefixes:
        for name, base, tok in models:
            msgs = build_coordinated_future_messages(observed_source=prefix, target_lang="Chinese", committed_text="", num_candidates=args.num_futures, prompt_version=SUFFIX_ICL_PROMPT_VERSION)
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            payload = {"model": name, "prompt": prompt, "max_tokens": max(40, 32 * args.num_futures), "temperature": 1.0, "top_p": 0.98, "n": 1, "presence_penalty": 0.15,
                       "stop": ["<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>"],
                       "seed": (args.seed + zlib.crc32(f"{prefix}||{name}".encode())) % (2**31)}
            try:
                data = post(f"{base}/completions", payload); raw = data["choices"][0]["text"]; fin = data["choices"][0].get("finish_reason")
            except Exception as e:
                out.write(json.dumps({"utt": utt, "step": t, "model": name, "prefix": prefix, "http_error": str(e)}) + "\n"); continue
            try:
                parsed = parse_grouped_future_output(raw, args.num_futures); err = None; n_ok += 1
            except ValueError as e:
                parsed = None; err = str(e); n_bad += 1
            out.write(json.dumps({"utt": utt, "step": t, "model": name, "prefix": prefix, "finish_reason": fin, "raw": raw, "parse_error": err, "n_parsed": len(parsed) if parsed else None}, ensure_ascii=False) + "\n"); out.flush()
            print(f"{name:8s} {utt} step{t:2d} ok={err is None} {('' if err is None else err)[:60]}", flush=True)
print(f"DONE ok={n_ok} bad={n_bad}")
