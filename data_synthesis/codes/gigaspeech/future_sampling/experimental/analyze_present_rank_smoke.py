#!/usr/bin/env python
"""Paired analysis: present-rank smoke vs the flagship J_40k decode of the SAME
utt_ids (single-track smoke; the baseline lives on disk).

Usage: analyze_present_rank_smoke.py <smoke_root> [baseline_root]

Leading indicators (wiki: 2026-07-consensus-register-forensics fix plan #4):
char-BLEU vs frozen ref 59 -> 65+, 4-gram recall 0.48 -> 0.6+, marker profile
closer to ref, len ratio -> 1.0, LAAL distribution matched to baseline.
Hygiene gates (must NOT regress): length_ratio_ref tail, rep-4gram
(within-prediction repetition), non-speech loop markers (呼/嘘/谢谢 runs).
Run with the evaluation env python (needs jieba).
"""
import json, glob, os, collections, statistics, sys, warnings
warnings.filterwarnings("ignore")
import jieba
jieba.setLogLevel(60)

SMOKE = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else \
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/J_40k"

smoke = {}
for fp in glob.glob(f"{SMOKE}/task_*/per_utt/*.json"):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    if d.get("prediction") and d.get("reference_text"):
        smoke[d["utt_id"]] = d

base = {}
for fp in glob.glob(f"{BASE}/task_*/per_utt/*.json"):
    uid = os.path.basename(fp)[:-5]
    if uid in smoke and uid not in base:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if d.get("prediction"):
            base[uid] = d

ids = sorted(set(smoke) & set(base))
print(f"paired utterances: {len(ids)} (smoke {len(smoke)}, baseline matched {len(base)})\n")
if not ids:
    sys.exit(0)

def m(d, k):
    v = (d.get("metrics") or {}).get(k)
    return v if v == v else None

pairs = [(smoke[i], base[i]) for i in ids]

print(f"{'':24}{'present':>10}{'J_40k':>10}{'delta':>9}")
for k in ["bleu_char", "laal_text", "length_ratio_ref"]:
    sv = [m(s, k) for s, b in pairs if m(s, k) is not None and m(b, k) is not None]
    bv = [m(b, k) for s, b in pairs if m(s, k) is not None and m(b, k) is not None]
    print(f"{k:24}{statistics.mean(sv):10.3f}{statistics.mean(bv):10.3f}"
          f"{statistics.mean(sv)-statistics.mean(bv):+9.3f}")

deltas = [m(s, "bleu_char") - m(b, "bleu_char") for s, b in pairs
          if m(s, "bleu_char") is not None and m(b, "bleu_char") is not None]
w = sum(1 for d in deltas if d > 0.5); l = sum(1 for d in deltas if d < -0.5)
print(f"\npaired char-BLEU: mean delta {statistics.mean(deltas):+.2f}, "
      f"median {statistics.median(deltas):+.2f}, W/L/T {w}/{l}/{len(deltas)-w-l}")

# LAAL distribution match (timing must be in-family)
for q in (0.5, 0.9, 0.99):
    sv = sorted(m(s, "laal_text") for s, b in pairs if m(s, "laal_text") is not None)
    bv = sorted(m(b, "laal_text") for s, b in pairs if m(b, "laal_text") is not None)
    print(f"laal p{int(q*100):<3} present {sv[int(q*(len(sv)-1))]:6.2f}  "
          f"J_40k {bv[int(q*(len(bv)-1))]:6.2f}")

def ngrams(s, n):
    return collections.Counter(s[i:i+n] for i in range(len(s)-n+1))

def recall(preds_refs, n):
    hits = tot = 0
    for p, r in preds_refs:
        rn = ngrams(r, n); pn = ngrams(p, n)
        tot += sum(rn.values())
        hits += sum(min(c, rn[g]) for g, c in pn.items() if g in rn)
    return hits / max(tot, 1)

sp = [(smoke[i]["prediction"], smoke[i]["reference_text"]) for i in ids]
bp = [(base[i]["prediction"], smoke[i]["reference_text"]) for i in ids]
print(f"\n{'char n-gram recall':24}{'present':>10}{'J_40k':>10}")
for n in (1, 2, 4):
    print(f"{n}-gram{'':18}{recall(sp, n):10.3f}{recall(bp, n):10.3f}")

# rep-4gram: within-prediction repetition (degeneration guard)
def rep4(texts):
    vals = []
    for t in texts:
        g = ngrams(t, 4)
        tot = sum(g.values())
        if tot >= 8:
            vals.append(1.0 - len(g) / tot)
    return statistics.mean(vals) if vals else float("nan")

print(f"\n{'rep-4gram (lower=better)':24}"
      f"{rep4([p for p, _ in sp]):10.3f}{rep4([p for p, _ in bp]):10.3f}"
      f"{rep4([smoke[i]['reference_text'] for i in ids]):10.3f}  (ref)")

# length-ratio tail (over-generation guard)
for lo, hi, lab in [(0.0, 0.7, "lr<0.7"), (1.5, 99.0, "lr>1.5")]:
    sc = sum(1 for s, b in pairs if m(s, "length_ratio_ref") is not None
             and lo <= m(s, "length_ratio_ref") < hi)
    bc = sum(1 for s, b in pairs if m(b, "length_ratio_ref") is not None
             and lo <= m(b, "length_ratio_ref") < hi)
    print(f"{lab:24}{sc:10d}{bc:10d}")

PUNCT = set("，。！？、；：“”‘’（）,.!?;:\"'()《》[]【】 —…·-")
def seg(s):
    return [x for x in jieba.lcut(s) if x.strip() and x not in PUNCT]

def prof(texts):
    c = collections.Counter(); nn = 0
    for t in texts:
        toks = seg(t); c.update(toks); nn += len(toks)
    return c, max(nn, 1)

pp, pn = prof([p for p, _ in sp])
cp, cn = prof([p for p, _ in bp])
rp, rn = prof([smoke[i]["reference_text"] for i in ids])
print(f"\nmarkers per 10k words (present / J_40k / frozen-ref):")
for wd in ["因此", "所以", "但是", "正是", "是", "能够", "可以", "如今", "现在",
           "与", "和", "将", "把", "并非", "不是", "或许", "可能", "许多", "很多"]:
    print(f"  {wd:4} {1e4*pp[wd]/pn:7.1f} / {1e4*cp[wd]/cn:7.1f} / {1e4*rp[wd]/rn:7.1f}")

print("\n(step-level changed_vs_baseline rates live in the verbose logs; "
      "grep '\"changed_vs_baseline\": true' if needed)")
