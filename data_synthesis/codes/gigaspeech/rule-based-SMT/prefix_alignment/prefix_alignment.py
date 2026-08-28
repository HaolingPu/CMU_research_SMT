#!/usr/bin/env python3
"""Prefix Alignment offline trajectory synthesis (Kano+ IWSLT 2022).

Reference:
  Kano, Sudoh, Nakamura. "Simultaneous Neural Machine Translation with
  Prefix Alignment." IWSLT 2022.  https://aclanthology.org/2022.iwslt-1.3/

Paper-faithful procedure (4.1 + 4.2):
  1. Translate the full source x  → y                      (free generation)
  2. For each cumulative source chunk x_{≤i}:
     a. translate(x_{≤i}, forced_prefix=committed)         → ȳ_i
     b. ȳ_i_lcp = LCP(y, ȳ_i)
     c. if len(ȳ_i_lcp) > len(committed):                  (strict growth = new pair)
        - extract pair (i, ȳ_i_lcp)
        - committed ← ȳ_i_lcp
        - y ← translate(x, forced_prefix=committed)        (re-translate with forced prefix)
  3. For each extracted (i, ȳ_i_lcp), find ref prefix length j (monotonic in t)
     that maximises BERTScore-F1(ȳ_i_lcp, ref[:j]).

GigaSpeech adaptation:
  Source granularity is fixed by `src_trajectory`; we iterate over chunks
  rather than per-word. Final chunk is force-completed to the full reference
  so target_full is always reconstructible from the emissions.

Output schema (compatible with existing MetricX converter):
  Per-utterance JSON contains the chunk/emission view requested:
    id, source, target, source_full, target_full, full_mt, method
  …plus the wait-k-style fields used by the MetricX QE pipeline:
    utt_id, source_full_text, target_trajectory, actions,
    prediction, reference_text, metrics{bleu_char, laal_text}, decoder_impl

Invariants enforced by validate_example():
  len(source) == len(target)
  source == src_trajectory
  source_full == join_source_chunks(source)
  "".join(target) == target_full   (final step force-completes to ref)
  target_prefixes monotonic; emissions never overwrite committed text
"""
import argparse
import ast
import json
import math
import os
import re
import statistics
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_TSV_PATH = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/"
    "train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
)


# ---------------------------------------------------------------------------
# Environment & CLI
# ---------------------------------------------------------------------------

def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE",
                          "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prefix Alignment trajectory synthesis (IWSLT 2022).")
    # Input
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--input-jsonl", default=None)
    p.add_argument("--id-key", default="id")
    p.add_argument("--src-trajectory-key", default="src_trajectory")
    p.add_argument("--target-key", default="llm_reference_text")
    # MT model
    p.add_argument("--mt-tokenizer-path", required=True)
    p.add_argument("--mt-api-base", required=True)
    p.add_argument("--mt-api-model",
                   default=os.environ.get("MT_API_MODEL", "qwen3-instruct"))
    p.add_argument("--mt-api-timeout", type=float, default=120.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--target-lang", default="Chinese")
    # PA params
    p.add_argument("--target-unit", choices=["char", "word"], default="char",
                   help="Unit for ref-prefix search and emission slicing.")
    p.add_argument("--lcp-mode", choices=["char", "word"], default="char")
    p.add_argument("--scorer", choices=["bertscore", "chrf", "edit"],
                   default="bertscore",
                   help="Aligner for hyp_prefix vs ref_prefix; paper uses BERTScore F1.")
    p.add_argument("--bertscore-lang", default="zh")
    p.add_argument("--bertscore-device", default="cuda:1",
                   help="Put BERTScore on a free GPU (vLLM is on cuda:0).")
    p.add_argument("--bertscore-batch", type=int, default=64)
    # Output / row selection
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=None,
                   help="Alias for --max-rows.")
    p.add_argument("--num-concurrent-cases", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Include debug fields in output.")
    p.add_argument("--run-toy-test", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def parse_trajectory(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(x) for x in ast.literal_eval(raw)]


def join_source_chunks(chunks: List[str]) -> str:
    text = ""
    for raw_piece in chunks:
        piece = str(raw_piece or "")
        if not piece:
            continue
        if not text:
            text = piece
            continue
        if text[-1].isspace() or piece[0].isspace():
            text += piece
        elif piece[0] in ",.!?;:)]}\"'":
            text += piece
        elif text[-1] in "([{\"'":
            text += piece
        else:
            text += " " + piece
    return text.strip()


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def normalize_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def verify_api(api_base: str, timeout: float) -> List[str]:
    req = urllib.request.Request(
        f"{normalize_api_base(api_base)}/models",
        headers={"Authorization": "Bearer dummy"}, method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]


def load_tokenizer(path: str) -> Any:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Translation:
#   - Free generation when committed_text == ""
#   - Prefix-constrained generation when committed_text is set
#     (assistant reply forced to start with committed_text; the model only
#     produces the continuation, which we concatenate back).
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: Any, source: str, target_lang: str,
                  committed_text: str = "") -> str:
    has_committed = bool(str(committed_text or "").strip())
    if has_committed:
        content = (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. Continue from that prefix "
            "and complete the translation. Output only the continuation."
        )
    else:
        content = (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only."
        )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=False, tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    if has_committed:
        prompt += committed_text
    return prompt


def translate(
    tokenizer: Any, source: str, committed_text: str,
    api_base: str, api_model: str, api_timeout: float,
    max_new_tokens: int, target_lang: str,
) -> str:
    """Free translation when committed_text is empty; prefix-constrained otherwise.

    Returns the *full* hypothesis (committed_text + continuation when constrained).
    """
    if not source or not source.strip():
        return committed_text or ""
    has_committed = bool(str(committed_text or "").strip())
    prompt = _build_prompt(tokenizer, source, target_lang, committed_text)
    data = _http_json(
        f"{normalize_api_base(api_base)}/completions",
        payload={
            "model": api_model, "prompt": prompt,
            "max_tokens": max_new_tokens, "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        },
        timeout=api_timeout,
    )
    choices = data.get("choices", [])
    if not choices:
        return committed_text if has_committed else ""
    continuation = clean_model_text(str(choices[0].get("text", "")))
    return (committed_text + continuation) if has_committed else continuation


# ---------------------------------------------------------------------------
# Longest common prefix
# ---------------------------------------------------------------------------

def lcp_chars(a: str, b: str) -> str:
    out: List[str] = []
    for ca, cb in zip(a, b):
        if ca == cb:
            out.append(ca)
        else:
            break
    return "".join(out)


def lcp_words(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    out: List[str] = []
    for ta, tb in zip(aw, bw):
        if ta == tb:
            out.append(ta)
        else:
            break
    return " ".join(out)


def longest_common_prefix(a: str, b: str, mode: str) -> str:
    return lcp_words(a, b) if mode == "word" else lcp_chars(a, b)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def units_of(text: str, unit: str) -> List[str]:
    if unit == "word":
        return text.split()
    return [c for c in text if not c.isspace()]


def assemble(units: List[str], unit: str) -> str:
    return " ".join(units) if unit == "word" else "".join(units)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def edit_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[lb] / max(la, lb)


def chrf_similarity(a: str, b: str, n_max: int = 6, beta: float = 2.0) -> float:
    a_chars = [c for c in a if not c.isspace()]
    b_chars = [c for c in b if not c.isspace()]
    if not a_chars or not b_chars:
        return 0.0
    f_scores: List[float] = []
    beta2 = beta * beta
    for n in range(1, n_max + 1):
        if len(a_chars) < n or len(b_chars) < n:
            continue
        a_ng = Counter(tuple(a_chars[i : i + n]) for i in range(len(a_chars) - n + 1))
        b_ng = Counter(tuple(b_chars[i : i + n]) for i in range(len(b_chars) - n + 1))
        match = sum(min(c, b_ng.get(g, 0)) for g, c in a_ng.items())
        a_total, b_total = sum(a_ng.values()), sum(b_ng.values())
        if a_total == 0 or b_total == 0:
            continue
        p, r = match / a_total, match / b_total
        f_scores.append(0.0 if (p + r) == 0
                        else (1 + beta2) * p * r / (beta2 * p + r))
    return sum(f_scores) / len(f_scores) if f_scores else 0.0


# BERTScore: lazy-loaded singleton, single GPU device, reused across utts.
_BERTSCORER = None
_BERTSCORER_LOCK = Lock()


def _get_bertscorer(lang: str, device: str):
    global _BERTSCORER
    with _BERTSCORER_LOCK:
        if _BERTSCORER is None:
            from bert_score import BERTScorer  # type: ignore
            _BERTSCORER = BERTScorer(
                lang=lang, rescale_with_baseline=False, device=device,
            )
        return _BERTSCORER


def bertscore_f1_batch(hyp: str, refs: List[str], lang: str, device: str,
                        batch_size: int) -> List[float]:
    """Compute BERTScore F1 between hyp and each ref in refs (batched).

    Returns one F1 per ref. Empty hyp or empty ref → 0.0.
    """
    if not hyp.strip() or not refs:
        return [0.0] * len(refs)
    scorer = _get_bertscorer(lang, device)
    hyps_batch = [hyp] * len(refs)
    safe_refs = [r if r.strip() else hyp for r in refs]  # avoid empty-ref crash
    with _BERTSCORER_LOCK:
        _, _, F = scorer.score(hyps_batch, safe_refs, batch_size=batch_size)
    out: List[float] = []
    for i, r in enumerate(refs):
        out.append(0.0 if not r.strip() else float(F[i]))
    return out


# ---------------------------------------------------------------------------
# Reference target extraction
# ---------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def extract_reference_text(row: Dict[str, Any], target_lang: str,
                           target_key: str) -> Optional[str]:
    keys: List[str] = [target_key]
    lang_suffix_map = {"Japanese": "ja", "German": "de",
                       "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    if lang_suffix:
        keys += [f"target_full_{lang_suffix}", f"tgt_text_full_{lang_suffix}",
                 f"llm_reference_text_{lang_suffix}"]
    keys += ["llm_reference_text", "tgt_text_full", "tgt_text",
             "target_text", "translation", "ref_text", "reference", "ref"]
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        raw = row.get(key)
        if _is_missing(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


# ---------------------------------------------------------------------------
# Metrics (BLEU-char + LAAL) — copied from local_agreement to keep numbers comparable.
# ---------------------------------------------------------------------------

def compute_laal(source_chunks: List[str], target_deltas: List[str],
                 actions: List[str], reference: str) -> float:
    timeline: List[int] = []
    source_read = 0
    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        source_read += len(str(chunk).strip().split()) if str(chunk).strip() else 0
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)
    y_len = len("".join(d for d in target_deltas if d))
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())
    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")
    denom = max(y_len, yref_len)
    if denom <= 0 or not timeline:
        return float("nan")
    total = sum(
        (timeline[i - 1] if i <= len(timeline) else x_len) - (i - 1) * x_len / denom
        for i in range(1, denom + 1)
    )
    return total / denom


def compute_bleu_char(hypothesis: str, reference: str,
                      max_order: int = 4, smooth: bool = True) -> float:
    hyp = [c for c in str(hypothesis) if not c.isspace()]
    ref = [c for c in str(reference) if not c.isspace()]
    hyp_len, ref_len = len(hyp), len(ref)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    eff_order = min(max_order, hyp_len, ref_len)
    if eff_order <= 0:
        return float("nan")
    precisions: List[float] = []
    for n in range(1, eff_order + 1):
        hyp_ngrams = Counter(tuple(hyp[i : i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(ref_len - n + 1))
        total = sum(hyp_ngrams.values())
        if total <= 0:
            return float("nan")
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items())
        if smooth:
            precisions.append((clipped + 1.0) / (total + 1.0))
        else:
            if clipped == 0:
                return 0.0
            precisions.append(clipped / total)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / eff_order) * 100.0


# ---------------------------------------------------------------------------
# Paper-faithful PA core
# ---------------------------------------------------------------------------

def best_alignment_length(
    hyp_prefix: str, ref_units: List[str], prev_j: int,
    unit: str, scorer_name: str, args: argparse.Namespace,
) -> int:
    """argmax_j scorer(hyp_prefix, ref[:j])  for j ∈ [prev_j, n_ref].

    For BERTScore we batch all candidate ref-prefixes in one call.
    """
    n = len(ref_units)
    if n == 0 or not hyp_prefix:
        return prev_j
    js = list(range(prev_j, n + 1))
    if scorer_name == "bertscore":
        candidates = [assemble(ref_units[:j], unit) for j in js]
        scores = bertscore_f1_batch(
            hyp_prefix, candidates,
            args.bertscore_lang, args.bertscore_device, args.bertscore_batch,
        )
    else:
        fn = chrf_similarity if scorer_name == "chrf" else edit_similarity
        scores = [fn(hyp_prefix, assemble(ref_units[:j], unit)) for j in js]
    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    return js[best_idx]


def collect_prefix_translation_pairs(
    chunks: List[str], source_full: str, tokenizer: Any,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[int, str]], List[str], List[str], str]:
    """Paper §4.1: extract (i, ȳ_i_lcp) pairs.

    For each source chunk index i:
      - translate(x_{≤i}, forced=committed)  → ȳ_i
      - LCP(y, ȳ_i) > committed   ⇒ extract pair, update committed, re-translate y.

    Returns:
      pairs:        list of (i, hyp_prefix)  in commit order
      prefix_mts:   ȳ_i for every i (for debug)
      ys_seen:      successive y values after each commit (for debug)
      final_y:      the last full-source translation (used for diagnostic)
    """
    n = len(chunks)
    # Initial full translation y0 (free).
    y = translate(
        tokenizer, source_full, "",
        args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
        args.max_new_tokens, args.target_lang,
    )
    committed = ""
    pairs: List[Tuple[int, str]] = []
    prefix_mts: List[str] = []
    ys_seen: List[str] = [y]
    for i in range(n):
        src_obs = join_source_chunks(chunks[: i + 1])
        if not src_obs.strip():
            prefix_mts.append("")
            continue
        y_bar_i = translate(
            tokenizer, src_obs, committed,
            args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
            args.max_new_tokens, args.target_lang,
        )
        prefix_mts.append(y_bar_i)
        new_lcp = longest_common_prefix(y, y_bar_i, args.lcp_mode)
        # Strict growth + must extend committed (true by construction since
        # forced_prefix=committed makes both y and y_bar_i start with committed).
        if (new_lcp.startswith(committed)
                and len(new_lcp) > len(committed)):
            committed = new_lcp
            pairs.append((i, committed))
            # Re-translate y with new committed forced prefix.
            y = translate(
                tokenizer, source_full, committed,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            ys_seen.append(y)
    return pairs, prefix_mts, ys_seen, y


def align_to_reference(
    pairs: List[Tuple[int, str]], target_full: str,
    args: argparse.Namespace,
) -> Dict[int, str]:
    """Paper §4.2: for each (i, ȳ_lcp), find ref_prefix length that maximises
    BERTScore-F1; enforce monotonicity over commit order.

    Returns: map from chunk_idx i → ref_prefix string.
    """
    ref_units = units_of(target_full, args.target_unit)
    out: Dict[int, str] = {}
    prev_j = 0
    for (i, hyp_prefix) in pairs:
        j = best_alignment_length(
            hyp_prefix, ref_units, prev_j,
            args.target_unit, args.scorer, args,
        )
        out[i] = assemble(ref_units[:j], args.target_unit)
        prev_j = j
    return out


def build_chunk_emission_trajectory(
    chunks: List[str], target_full: str,
    pairs: List[Tuple[int, str]], ref_at_commit: Dict[int, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Convert commit-time ref prefixes into chunk/emission trajectory.

    Force-completes the final chunk to target_full so the emissions
    reconstruct the full reference.

    Returns:
      target_emissions  per-step delta (NOT cumulative)
      target_prefixes   cumulative committed ref prefix at each step
      actions           "WRITE"|"READ" per step
    """
    n = len(chunks)
    target_prefixes: List[str] = []
    running = ""
    commit_set = {i for i, _ in pairs}
    for t in range(n):
        if t in commit_set:
            running = ref_at_commit.get(t, running)
        target_prefixes.append(running)
    # Force-complete on final chunk: dump anything still un-emitted.
    if target_prefixes:
        target_prefixes[-1] = target_full

    target_emissions: List[str] = []
    actions: List[str] = []
    for t in range(n):
        cur = target_prefixes[t]
        prev = target_prefixes[t - 1] if t > 0 else ""
        if cur.startswith(prev):
            emit = cur[len(prev):]
        else:
            emit = ""
        target_emissions.append(emit)
        actions.append("WRITE" if emit.strip() else "READ")
    return target_emissions, target_prefixes, actions


def validate_example(example: Dict[str, Any]) -> None:
    src = example["source"]
    tgt = example["target"]
    if len(src) != len(tgt):
        raise AssertionError(f"len(source)={len(src)} != len(target)={len(tgt)}")
    rebuilt_full = join_source_chunks(src)
    if rebuilt_full != example["source_full"]:
        raise AssertionError(
            f"source_full mismatch: rebuilt={rebuilt_full!r} "
            f"vs stored={example['source_full']!r}")
    rebuilt_target = "".join(tgt)
    if rebuilt_target != example["target_full"]:
        raise AssertionError(
            f"target_full mismatch: rebuilt={rebuilt_target!r} "
            f"vs stored={example['target_full']!r}")
    dbg = example.get("debug")
    if dbg is not None:
        prefixes = dbg.get("target_prefixes", [])
        for i in range(1, len(prefixes)):
            if not prefixes[i].startswith(prefixes[i - 1]):
                raise AssertionError(
                    f"non-monotonic target_prefixes at step {i}")
        running = ""
        for t, emit in enumerate(tgt):
            running += emit
            if t < len(prefixes) and running != prefixes[t]:
                raise AssertionError(
                    f"emission/prefix mismatch at step {t}: "
                    f"running={running!r} prefix={prefixes[t]!r}")


def process_example(
    row: Dict[str, Any], args: argparse.Namespace, tokenizer: Any,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_key, row.get("id", "row")))
    chunks = parse_trajectory(row[args.src_trajectory_key])
    chunks = [str(c or "") for c in chunks]

    source_full = join_source_chunks(chunks)
    if not source_full.strip():
        raise ValueError(f"empty source_full for {utt_id}")
    target_full = extract_reference_text(row, args.target_lang, args.target_key)
    if not target_full:
        raise ValueError(f"reference target not found for {utt_id}")

    # §4.1: collect (i, ȳ_lcp) prefix translation pairs with prefix-constrained MT.
    pairs, prefix_mts, ys_seen, final_y = collect_prefix_translation_pairs(
        chunks, source_full, tokenizer, args,
    )

    # §4.2: align each ȳ_lcp to a reference prefix length (BERTScore F1).
    ref_at_commit = align_to_reference(pairs, target_full, args)

    # Convert to chunk/emission trajectory; force-complete last chunk.
    target_emissions, target_prefixes, actions = build_chunk_emission_trajectory(
        chunks, target_full, pairs, ref_at_commit,
    )

    prediction = "".join(target_emissions)  # == target_full by force-complete
    laal_value = float("nan")
    bleu_char_value = float("nan")
    try:
        laal_value = compute_laal(chunks, target_emissions, actions, target_full)
        bleu_char_value = compute_bleu_char(prediction, target_full)
    except Exception:
        pass

    # subsentence list passed through from input row (used by SEGALE prep)
    src_text_full_raw = row.get("src_text_full")
    src_text_full_list: List[str]
    if isinstance(src_text_full_raw, list):
        src_text_full_list = [str(x) for x in src_text_full_raw]
    elif isinstance(src_text_full_raw, str) and src_text_full_raw.strip():
        try:
            parsed = ast.literal_eval(src_text_full_raw)
            src_text_full_list = [str(x) for x in parsed] if isinstance(parsed, list) else [source_full]
        except (ValueError, SyntaxError):
            src_text_full_list = [source_full]
    else:
        src_text_full_list = [source_full]

    pred_chars = sum(1 for c in prediction if not c.isspace())
    ref_chars = sum(1 for c in target_full if not c.isspace())
    src_words = len(source_full.split())
    length_ratio_ref = (pred_chars / ref_chars) if ref_chars > 0 else float("nan")
    length_ratio_src = (pred_chars / src_words) if src_words > 0 else float("nan")

    out: Dict[str, Any] = {
        # Chunk/emission view (user-requested schema)
        "id": utt_id,
        "source": list(chunks),
        "target": list(target_emissions),
        "source_full": source_full,
        "target_full": target_full,
        "full_mt": ys_seen[0] if ys_seen else "",
        "method": "prefix_alignment_trajectory",
        # Wait-k / LA compatible view (consumed by convert_metricx_consensus.py)
        "utt_id": utt_id,
        "src_trajectory": list(chunks),
        "src_text_full": src_text_full_list,
        "source_full_text": source_full,
        "target_trajectory": list(target_emissions),
        "actions": list(actions),
        "prediction": prediction,
        "reference_text": target_full,
        "decoder_impl": {
            "method": "prefix_alignment",
            "scorer": args.scorer,
            "target_unit": args.target_unit,
            "lcp_mode": args.lcp_mode,
        },
        "metrics": {
            "laal_text": laal_value,
            "bleu_char": bleu_char_value,
            "length_ratio_ref": length_ratio_ref,
            "length_ratio_src": length_ratio_src,
            "pred_chars": pred_chars,
            "ref_chars": ref_chars,
            "src_words": src_words,
        },
    }
    if args.debug:
        out["debug"] = {
            "source_prefixes": [join_source_chunks(chunks[: t + 1])
                                 for t in range(len(chunks))],
            "prefix_mt": prefix_mts,
            "extracted_pairs": [{"chunk_idx": i, "hyp_prefix": h} for i, h in pairs],
            "ys_seen": ys_seen,
            "ref_at_commit": {str(k): v for k, v in ref_at_commit.items()},
            "target_prefixes": target_prefixes,
        }
    if not args.debug:
        validation_view = dict(out)
        validation_view["debug"] = {"target_prefixes": target_prefixes}
        validate_example(validation_view)
    else:
        validate_example(out)

    src_len, tgt_len = len(out["source"]), len(out["target"])
    assert src_len == tgt_len, (
        f"len(src_trajectory)={src_len} != len(tgt_trajectory)={tgt_len} "
        f"for utt_id={utt_id}"
    )
    return out


# ---------------------------------------------------------------------------
# Input loading & row selection
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_key].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected
    if args.test_one:
        return df.iloc[[args.row_idx]]
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max(1, int(args.max_rows)))
    return df.iloc[start:end]


# ---------------------------------------------------------------------------
# Toy test (no MT calls; exercises emission decomposition + validation)
# ---------------------------------------------------------------------------

def run_toy_test() -> None:
    chunks = ["I bought", "a red apple", "yesterday"]
    target_full = "我昨天买了一个红苹果"
    pairs = [(0, "我"), (2, "我昨天买了一个红苹果")]
    ref_at_commit = {0: "我", 2: "我昨天买了一个红苹果"}
    target_emissions, target_prefixes, actions = build_chunk_emission_trajectory(
        chunks, target_full, pairs, ref_at_commit,
    )
    example = {
        "id": "toy",
        "source": list(chunks),
        "target": list(target_emissions),
        "source_full": join_source_chunks(chunks),
        "target_full": target_full,
        "full_mt": target_full,
        "method": "prefix_alignment_trajectory",
        "debug": {"target_prefixes": target_prefixes},
    }
    validate_example(example)
    assert "".join(target_emissions) == target_full
    print("[toy_test] OK")
    print(f"  source = {chunks}")
    print(f"  target = {target_emissions}")
    print(f"  target_prefixes = {target_prefixes}")
    print(f"  actions = {actions}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_env()
    args = parse_args()
    if args.run_toy_test:
        run_toy_test()
        return

    if args.max_examples is not None:
        args.max_rows = int(args.max_examples)

    df = load_jsonl(args.input_jsonl) if args.input_jsonl \
        else pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)
    print(
        f"Processing {len(rows)} example(s)  PA  "
        f"unit={args.target_unit}  scorer={args.scorer}  lcp_mode={args.lcp_mode}"
    )

    models = verify_api(args.mt_api_base, args.mt_api_timeout)
    if args.mt_api_model not in models:
        raise RuntimeError(
            f"model '{args.mt_api_model}' not found; available={models}")
    print(f"[MT] model={args.mt_api_model}  "
          f"api={normalize_api_base(args.mt_api_base)}")

    tokenizer = load_tokenizer(args.mt_tokenizer_path)
    if args.scorer == "bertscore":
        # eager-load so we crash early if model download fails
        _get_bertscorer(args.bertscore_lang, args.bertscore_device)
        print(f"[BERTScore] lang={args.bertscore_lang}  "
              f"device={args.bertscore_device}  batch={args.bertscore_batch}")

    out_fh = None
    out_lock = Lock()
    if args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)),
                    exist_ok=True)
        out_fh = open(args.output_jsonl,
                      "w" if args.overwrite else "a", encoding="utf-8")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    stats_lock = Lock()
    n_processed = 0
    n_skipped = 0
    n_failed = 0
    n_nonempty_emissions_total = 0
    bleu_acc: List[float] = []
    laal_acc: List[float] = []

    def _process_one_row(row_idx: int, row_dict: Dict[str, Any]) -> None:
        nonlocal n_processed, n_skipped, n_failed, n_nonempty_emissions_total
        utt_id = str(row_dict.get(args.id_key,
                                   row_dict.get("id", f"row_{row_idx}")))
        out_path = (os.path.join(args.output_dir,
                                  f"{sanitize_filename(utt_id)}.json")
                    if args.output_dir else None)
        if out_path and os.path.exists(out_path) and not args.overwrite:
            with stats_lock:
                n_skipped += 1
            return

        try:
            result = process_example(row_dict, args, tokenizer)
        except Exception as exc:
            with stats_lock:
                n_failed += 1
            print(f"[ERROR] row {row_idx} ({utt_id}): {exc}")
            return

        n_nonempty = sum(1 for t in result["target"] if t and t.strip())
        m = result["metrics"]
        with stats_lock:
            n_processed += 1
            n_nonempty_emissions_total += n_nonempty
            if not (isinstance(m["bleu_char"], float)
                    and math.isnan(m["bleu_char"])):
                bleu_acc.append(float(m["bleu_char"]))
            if not (isinstance(m["laal_text"], float)
                    and math.isnan(m["laal_text"])):
                laal_acc.append(float(m["laal_text"]))
        print(f"  {utt_id}  steps={len(result['target'])}  "
              f"non_empty={n_nonempty}  bleu={m['bleu_char']:.2f}  "
              f"laal={m['laal_text']:.2f}")

        if out_fh:
            with out_lock:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
                fh.write("\n")

    row_items = [(idx, row.to_dict())
                 for idx, (_, row) in enumerate(rows.iterrows())]
    num_concurrent = max(1, int(args.num_concurrent_cases))
    if num_concurrent <= 1:
        for ri, rd in row_items:
            _process_one_row(ri, rd)
    else:
        print(f"[Concurrent] {len(row_items)} rows, {num_concurrent} workers")
        with ThreadPoolExecutor(max_workers=num_concurrent) as ex:
            futs = {ex.submit(_process_one_row, ri, rd): ri for ri, rd in row_items}
            for fut in as_completed(futs):
                fut.result()

    if out_fh:
        out_fh.close()

    avg_emit = (n_nonempty_emissions_total / n_processed) if n_processed else 0.0
    bleu_avg = statistics.fmean(bleu_acc) if bleu_acc else float("nan")
    laal_avg = statistics.fmean(laal_acc) if laal_acc else float("nan")
    print(
        f"Done. processed={n_processed}  skipped={n_skipped}  failed={n_failed}  "
        f"avg_non_empty_emissions={avg_emit:.2f}  "
        f"BLEU_char={bleu_avg:.2f}  LAAL={laal_avg:.2f}"
    )


if __name__ == "__main__":
    main()
