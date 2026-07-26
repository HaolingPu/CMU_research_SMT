#!/usr/bin/env python
"""Paired analysis of an anchor-and-veto smoke run: anchor vs vanilla consensus
(dual-track, same utterances, same futures). Usage: analyze_anchor_smoke.py <run_root>.

Success criteria (wiki: 2026-07-consensus-register-forensics): char-BLEU vs frozen
ref 59 -> 65+, 4-gram overlap 0.48 -> 0.6+, marker profile closer to ref, len
ratio -> 1.0, laal comparable to the baseline track."""
import json, glob, collections, statistics, sys, warnings
warnings.filterwarnings("ignore")
import jieba
jieba.setLogLevel(60)

ROOT = sys.argv[1]
files = sorted(glob.glob(f"{ROOT}/task_*/per_utt/*.json"))
recs = []
for fp in files:
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    if d.get("prediction") is None or not d.get("reference_text"):
        continue
    recs.append(d)
print(f"{len(recs)} utterances\n")
if not recs:
    sys.exit(0)

def agg(key, sub="metrics"):
    vals = [r[sub][key] for r in recs if r.get(sub) and r[sub].get(key) == r[sub].get(key)]
    return statistics.mean(vals) if vals else float("nan")

print(f"{'':22}{'anchor':>10}{'consensus':>11}")
for k in ["bleu_char", "laal_text", "length_ratio_ref"]:
    print(f"{k:22}{agg(k):10.2f}{agg(k, 'consensus_metrics'):11.2f}")

def ngrams(s, n):
    return collections.Counter(s[i:i+n] for i in range(len(s)-n+1))
def overlap(key, n):
    hits = tot = 0
    for r in recs:
        rn = ngrams(r["reference_text"], n); pn = ngrams(r[key], n)
        tot += sum(rn.values())
        hits += sum(min(c, rn[g]) for g, c in pn.items() if g in rn)
    return hits / max(tot, 1)
print(f"\n{'char n-gram recall':22}{'anchor':>10}{'consensus':>11}")
for n in (1, 4):
    print(f"{n}-gram{'':16}{overlap('prediction', n):10.3f}{overlap('consensus_prediction', n):11.3f}")

PUNCT = set("，。！？、；：“”‘’（）,.!?;:\"'()《》[]【】 —…·-")
def seg(s):
    return [w for w in jieba.lcut(s) if w.strip() and w not in PUNCT]
def prof(key):
    c = collections.Counter(); nn = 0
    for r in recs:
        t = seg(r[key]); c.update(t); nn += len(t)
    return c, max(nn, 1)
ap, an = prof("prediction"); cp, cn = prof("consensus_prediction"); rp, rn = prof("reference_text")
print(f"\nmarkers per 10k words (anchor / consensus / frozen-ref):")
for w in ["因此", "所以", "正是", "能够", "如今", "与", "和", "将", "把", "前往", "去", "说道", "而"]:
    print(f"  {w:4} {1e4*ap[w]/an:7.1f} / {1e4*cp[w]/cn:7.1f} / {1e4*rp[w]/rn:7.1f}")

commit_lens = []; veto_toks = collections.Counter(); empty_anchor = 0
for r in recs:
    for dbg in r.get("anchor_debug", []):
        commit_lens.append(dbg.get("commit_len", 0))
        if dbg.get("stop") == "empty_anchor":
            empty_anchor += 1
        v = dbg.get("veto")
        if v:
            veto_toks[v.get("token", "?")] += 1
w = [c for c in commit_lens if c > 0]
print(f"\nveto: chunks={len(commit_lens)} WRITE={len(w)} ({100*len(w)/max(len(commit_lens),1):.0f}%) "
      f"mean commit_len(when>0)={statistics.mean(w) if w else 0:.1f} empty_anchor={empty_anchor}")
print("top vetoed:", ", ".join(f"{t!r}×{c}" for t, c in veto_toks.most_common(10)))

deltas = [r["metrics"]["bleu_char"] - r["consensus_metrics"]["bleu_char"] for r in recs
          if r["metrics"]["bleu_char"] == r["metrics"]["bleu_char"]
          and r["consensus_metrics"]["bleu_char"] == r["consensus_metrics"]["bleu_char"]]
if deltas:
    wins = sum(1 for d in deltas if d > 1); losses = sum(1 for d in deltas if d < -1)
    print(f"\npaired delta bleu_char (anchor-consensus): mean {statistics.mean(deltas):+.2f}  "
          f"wins {wins} / losses {losses} / ties {len(deltas)-wins-losses}")
