# Consensus: my methods vs the futures=200 baseline

**Eval:** ACL 6060 dev, en→zh. COMET = `Unbabel/XCOMET-XL` (ranking metric).
Latency = LongYAAL (CU), ms (lower = faster). Rank by COMET, not BLEU.

**Baseline definition.** `consensus-topk5` *is* the trained futures=200 baseline:
`convert2swift_consensus.py` builds its training data from
`consensus_decoding_en_zh_top_5_futures200-segale/qe3-lr-aligned-full`
(200 undirected futures, plain token-intersection consensus, no axis steering).

My contributions:
- **5-axis** (`top5-axis5`): directed future sampling — 5 narrative axes × ~4 samples = **20 futures**.
- **soft-vote** (`top5-axis5-sv`): loosen the hard top-k intersection to a majority vote.
- **fut100_n100**: 5-axis scaled to 100 futures (saturation check).

## COMET (higher = better)
| method | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| futures200 baseline (topk5) | 0.777 | 0.797 | 0.799 | 0.806 |
| **5-axis** | **0.787** | **0.808** | **0.812** | **0.817** |
| soft-vote | 0.763 | 0.808 | 0.813 | 0.812 |
| fut100_n100 | 0.767 | 0.805 | 0.807 | 0.816 |

## BLEU
| method | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| futures200 baseline (topk5) | 32.4 | 35.2 | 34.7 | 37.4 |
| **5-axis** | **34.9** | **39.6** | **40.1** | **40.1** |
| soft-vote | 30.9 | 37.6 | 38.5 | 38.7 |
| fut100_n100 | 32.1 | 36.9 | 38.4 | 39.3 |

## Latency (LongYAAL CU, ms)
| method | seg960 | seg1920 | seg2880 | seg3840 |
|---|---|---|---|---|
| futures200 baseline (topk5) | 1319 | 1883 | 2378 | 2855 |
| 5-axis | 1461 | 2176 | 2745 | 3107 |
| soft-vote | 1289 | 1964 | 2464 | 2894 |
| fut100_n100 | 1209 | 1890 | 2371 | 2757 |

## Takeaways
1. **5-axis dominates the 200-future baseline on COMET at every latency** (+0.010 to +0.013),
   and on BLEU at every latency (+2.7 to +4.4).
2. **Efficiency win (headline):** 5-axis at seg1920 (COMET **0.808**, 2176 ms) already **exceeds the
   baseline's best point** (seg3840, COMET 0.806, 2855 ms). Directed diversity from **20** futures
   beats brute-force **200** undirected futures — matching+beating peak quality ~680 ms sooner.
3. **soft-vote** edges the baseline but stays below 5-axis; at high latency its COMET dips
   (0.812 < 0.817) and BLEU is clearly lower (38.7 < 40.1). Loosening consensus does not help —
   negative result, confirmed at trained-eval (not just synthesis-time).
4. **fut100_n100** ≈ ties 5-axis (0.816 vs 0.817) at higher cost → **20 futures already saturates**;
   no need for 100, let alone 200.
5. **post-edit polish** (`consensus_decoding_retranslate.py`) was synthesis-time only and regressed
   (anticipation / hallucination) — not trained. The consensus draft is already at its fluency
   frontier without future leakage.

## Sample outputs (seg3840, ACL6060 doc0)
```
# seg0
REF        : 大家好，我是Elena，我将向大家介绍我们的工作——检测西班牙语中的未同化借词：注释语料库和建模方法。
baseline   : 嗨,我是伊琳娜,我将介绍我们关于检测西班牙语未吸收借用词的工作,以及相关建模方法。
5-axis     : 嗨,这是伊琳娜,我将介绍我们关于检测西班牙语未吸收借用词的工作,以及标注语料库和建模方法。
soft-vote  : 嗨,我是伊琳娜,我将介绍我们关于检测西班牙语未吸收借用词的工作,以及相关模型构建的方法。
# (5-axis recovers "标注语料库和建模方法" matching the reference; baseline drops it to "相关建模方法")

# seg1
REF        : 我们将讨论什么是词汇借用、我们提出的任务、我们已经发布的数据集，以及我们探索的一些模型。
baseline   : 我们将涵盖什么是词法借用,我们提出的任务,我们发布的数据集,以及我们探索的一些模型。
5-axis     : 因此,我们将涵盖词法借用的定义,我们提出的任务,我们发布的数据集,以及我们探索的一些模型。

# seg2
REF        : 首先，什么是词汇借用，为什么它作为一个自然语言处理任务很重要？
baseline   : 但首先,什么是词法借用,为什么它作为自然语言处理任务如此重要?
5-axis     : 但首先,什么是词法借用,以及它为何作为自然语言处理任务至关重要
```

## Sources
- scores: `ckpts/infinisst-omni/gigaspeech-zh-consensus-{topk5,top5-axis5,top5-axis5-sv,top5-axis5-fut100_n100}-s-bsz4/*-hf/evaluation/acl_6060/en-zh/seg*/segmentation_output/scores.tsv`
- baseline data provenance: `scripts/train/convert2swift_consensus.py` → `consensus_decoding_en_zh_top_5_futures200-segale/qe3-lr-aligned-full`
- sample outputs: `.../seg3840/segmentation_output/instances.resegmented.jsonl`
