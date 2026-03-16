#!/usr/bin/env python3
"""
Thinking-policy simultaneous interpretation pipeline with simalign post-check.

Per trajectory step:
  1. Accumulate observed English source prefix.
  2. Base model samples N possible future source continuations.
  3. Send to thinking model: observed prefix, list of futures, committed Chinese so far.
  4. Thinking model outputs ONLY the additional Chinese delta that is safe to emit
     (consistent with all possible futures). If no safe delta, output EMPTY → READ.
  5. Run a simalign-based post-check to truncate/reject future-only over-translation.
  6. At end of utterance, force-complete the remaining translation.

No awesome-align, sentence-path, or legacy majority-vote logic.

File structure:
  - CLI, env, text utils: argument parsing, TSV helpers, normalize_zh, clean_llm_output, etc.
  - Base model: load_base_model(), sample_futures() for N future English continuations.
  - Thinking model: build_thinking_prompt(), call_thinking_model() for safe delta; force_complete_translation() at end.
  - Simalign post-check: apply_simalign_delta_check() trims delta to the observed-source boundary.
  - process_one_utterance(): main loop over trajectory chunks; per chunk: sample futures → thinking delta or READ; last chunk → force-complete.
  - I/O: iter_assigned_rows(), get_one_row_by_id(), main() with JSON output.

Assumptions:
  - TSV has columns: id (or --id-column), src_text_full (list or string), src_trajectory (list or string).
  - Thinking model is served via OpenAI-compatible API (e.g. vLLM with chat template).
  - Base model is loadable with vLLM for text generation only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import re
import sys
import textwrap
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI


DEFAULT_TRANSLATION_CACHE_DIR = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "llm_full_translation_cache/train_xl_case_robust_asr_filtered"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thinking-policy simultaneous interpretation with simalign-based over-translation check."
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--thinking-api-base", default="http://localhost:8001/v1")
    p.add_argument("--thinking-model-name", default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8")
    p.add_argument("--thinking-tokenizer-path",
                   default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=6, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=12)
    p.add_argument("--sample-temperature", type=float, default=0.7)
    p.add_argument("--thinking-temperature", type=float, default=0.3)
    p.add_argument("--thinking-max-tokens", type=int, default=16384)
    p.add_argument("--align-device", default="cuda:0",
                   help="Device for simalign check model (e.g. cuda:0 or cpu).")

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-column", default="id")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default=None)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# =============================================================================
# Environment
# =============================================================================

def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Text utilities
# =============================================================================

def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_llm_output(text: str) -> str:
    """Strip thinking tags and stray quotes from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("</think>", "")
    text = (text or "").strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    return text


def clean_continuation(observed: str, raw_output: str, max_words: int = 15) -> str:
    """Extract continuation text after observed prefix, cap length."""
    text = (raw_output or "").strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    obs_lower = (observed or "").lower().strip()
    text_lower = (text or "").lower()
    if obs_lower and text_lower.startswith(obs_lower):
        text = text[len(obs_lower):].strip()
    words = (text or "").split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return (text or "").strip()


_FIRST_SENTENCE_END_RE = re.compile(r'[.!?](?:["\')\]]+)?(?=\s|$)')
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ZH_PUNCT_ONLY_RE = re.compile(r"^[，。！？；：、（）《》【】“”‘’…,.!?;:'\"()\\-]+$")
_ANSWER_LABEL_RE = re.compile(
    r"^(?:safe output|output|answer|final answer|final decision)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
_ANSWER_LABEL_INLINE_RE = re.compile(
    r"(?:safe output|output|answer|final answer|final decision)\s*[:：]\s*[\"“]?([^\"\n”]+)",
    flags=re.IGNORECASE,
)
_QUOTED_ANSWER_RE = re.compile(r"[\"“](EMPTY|[^\"\n”]*[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][^\"\n”]*)[\"”]", flags=re.IGNORECASE)


def truncate_future_to_first_sentence(text: str) -> str:
    """Keep at most the first English sentence to avoid spillover.
    With future_tokens=12, many samples have no period so this often no-ops; still helps when it does."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _FIRST_SENTENCE_END_RE.search(text)
    if not m:
        return text
    return text[: m.end()].strip()


def strip_committed_suffix_from_delta(committed: str, delta: str) -> str:
    """If delta starts with a suffix of committed (overlap), strip it to avoid double-commit.
    Used for step-level delta and final completion."""
    delta = (delta or "").strip()
    if not delta:
        return ""
    if not committed:
        return normalize_zh(delta)
    committed_norm = normalize_zh(committed)
    delta_norm = normalize_zh(delta)
    max_k = min(len(committed_norm), len(delta_norm))
    for k in range(max_k, 0, -1):
        suffix = committed_norm[-k:]
        if delta_norm.startswith(suffix):
            return normalize_zh(delta_norm[k:].strip())
    return delta_norm


def compute_laal(
    source_chunks: List[str],
    target_deltas: List[str],
    actions: List[str],
    reference: str,
) -> float:
    """Text LAAL using source-word count as source-time surrogate."""
    timeline: List[int] = []
    source_read = 0

    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        words_in_chunk = len(str(chunk).strip().split()) if str(chunk).strip() else 0
        source_read += words_in_chunk
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)

    y = "".join(d for d in target_deltas if d)
    y_len = len(y)
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(
        len(str(c).strip().split())
        for c in source_chunks
        if str(c).strip()
    )

    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")

    denom = max(y_len, yref_len)
    if denom <= 0 or len(timeline) == 0:
        return float("nan")

    total_lagging = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        d_star_i = (i - 1) * x_len / denom
        total_lagging += (d_i - d_star_i)

    return total_lagging / denom


def _char_tokens_zh(text: str) -> List[str]:
    """Simple char-level tokenization for Chinese BLEU (remove whitespace)."""
    return [c for c in str(text) if not c.isspace()]


def compute_bleu_char(
    hypothesis: str,
    reference: str,
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """Sentence-level BLEU on character tokens (Chinese-friendly, no external deps)."""
    hyp = _char_tokens_zh(hypothesis)
    ref = _char_tokens_zh(reference)
    hyp_len = len(hyp)
    ref_len = len(ref)

    if hyp_len == 0 or ref_len == 0:
        return float("nan")

    eff_order = min(max_order, hyp_len, ref_len)
    if eff_order <= 0:
        return float("nan")

    precisions: List[float] = []
    for n in range(1, eff_order + 1):
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(ref_len - n + 1))
        total = sum(hyp_ngrams.values())
        if total <= 0:
            return float("nan")
        clipped = 0
        for ng, cnt in hyp_ngrams.items():
            clipped += min(cnt, ref_ngrams.get(ng, 0))
        if smooth:
            p_n = (clipped + 1.0) / (total + 1.0)
        else:
            if clipped == 0:
                return 0.0
            p_n = clipped / total
        precisions.append(p_n)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / hyp_len))

    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / eff_order)
    return bleu * 100.0




def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return (safe or "unknown")[:200]


def _vlog(log_file: Optional[Any], msg: str) -> None:
    if log_file is not None:
        log_file.write(msg)
        if not msg.endswith("\n"):
            log_file.write("\n")
        log_file.flush()


def _format_verbose_paragraph(label: str, text: str, width: int = 100) -> str:
    body = (text or "").strip()
    if not body:
        return f"{label}:"
    body = re.sub(r"\s+", " ", body)
    wrapped = textwrap.fill(
        body,
        width=width,
        initial_indent="    ",
        subsequent_indent="    ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"{label}:\n{wrapped}"


class _TeeWriter:
    def __init__(self, file_obj):
        self._f = file_obj

    def write(self, msg):
        self._f.write(msg)
        sys.stdout.write(msg)

    def flush(self):
        self._f.flush()
        sys.stdout.flush()

    def close(self):
        self._f.close()


def _message_text_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _clean_answer_candidate(text: str) -> str:
    text = (text or "").strip()
    text = _ANSWER_LABEL_RE.sub("", text).strip()
    text = text.strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    return text.strip()


def _looks_like_answer_content(text: str) -> bool:
    text = _clean_answer_candidate(text)
    if not text:
        return False
    if text.upper() == "EMPTY":
        return True
    if _ZH_PUNCT_ONLY_RE.fullmatch(text) and len(text) <= 8:
        return True
    if "\n" in text:
        return False
    if _CJK_CHAR_RE.search(text):
        ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        return ascii_letters <= max(6, len(text) // 4)
    return False


def _extract_answer_candidate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    direct = _clean_answer_candidate(text)
    if _looks_like_answer_content(direct):
        return direct

    inline_matches = list(_ANSWER_LABEL_INLINE_RE.finditer(text))
    for match in reversed(inline_matches):
        candidate = _clean_answer_candidate(match.group(1))
        if _looks_like_answer_content(candidate):
            return candidate

    quoted_matches = list(_QUOTED_ANSWER_RE.finditer(text))
    for match in reversed(quoted_matches):
        candidate = _clean_answer_candidate(match.group(1))
        if _looks_like_answer_content(candidate):
            return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = _clean_answer_candidate(line)
        if _looks_like_answer_content(candidate):
            return candidate

    return ""


def _split_reasoning_and_content(raw_reasoning: str, raw_content: str) -> Tuple[str, str]:
    """Best-effort recovery when the server fails to separate reasoning/content.

    We prefer the server-provided reasoning/content fields. If `message.content`
    still contains `<think>...</think>` (or only an orphan `</think>`), we use
    the suffix after the LAST closing tag as the answer. If the model emits
    plain-English analysis plus a final Chinese answer line, we recover the last
    answer-looking candidate and drop the rest.
    """
    reasoning_text = (raw_reasoning or "").strip()
    content_text = (raw_content or "").strip()
    if not content_text:
        return reasoning_text, ""

    if re.search(r"</think>", content_text, flags=re.IGNORECASE):
        parts = re.split(r"</think>", content_text, flags=re.IGNORECASE)
        if not reasoning_text:
            reasoning_chunks = []
            for part in parts[:-1]:
                cleaned = re.sub(r"<think>", "", part, flags=re.IGNORECASE).strip()
                if cleaned:
                    reasoning_chunks.append(cleaned)
            reasoning_text = "\n\n".join(reasoning_chunks).strip()
        content_text = parts[-1].strip()
    elif re.search(r"<think>", content_text, flags=re.IGNORECASE):
        # Unclosed think block: safer to drop it than write reasoning into output.
        if not reasoning_text:
            reasoning_text = re.sub(r"<think>", "", content_text, flags=re.IGNORECASE).strip()
        content_text = ""

    content_text = _extract_answer_candidate(content_text)
    return reasoning_text, content_text


def _raw_message_debug_fields(message: Any) -> Dict[str, str]:
    """Capture raw reasoning/content fields from the server response before local fallback parsing."""
    return {
        "message.reasoning": _message_text_to_str(getattr(message, "reasoning", None)),
        "message.reasoning_content": _message_text_to_str(getattr(message, "reasoning_content", None)),
        "message.content": _message_text_to_str(getattr(message, "content", None)),
    }


def apply_simalign_delta_check(
    accumulated_source: str,
    futures: List[str],
    committed: str,
    delta: str,
    align_model: Any,
    align_tokenizer: Any,
) -> Tuple[str, Dict[str, Any]]:
    """Use simalign to reject/truncate delta that crosses into future-only content.

    For each sampled future, align the FULL source (`observed + future`) against
    the FULL candidate Chinese (`committed + delta`), then scan target-side
    alignments left-to-right and stop once alignment enters future-only source
    words. The shortest surviving prefix across valid alignment cases is kept.
    """
    committed_norm = normalize_zh(committed)
    delta_norm = normalize_zh(delta)
    if not delta_norm:
        return "", {"status": "empty_delta"}
    if align_model is None or not (accumulated_source or "").strip():
        return delta_norm, {"status": "skip_no_aligner"}
    if not futures:
        return delta_norm, {"status": "skip_no_futures"}

    try:
        import llm_future_sampling_core_v2 as simalign_v2
    except Exception as e:
        return "", {"status": f"reject_import_error:{type(e).__name__}:{e}"}

    candidate_full = committed_norm + delta_norm
    full_srcs = [
        f"{accumulated_source} {future}".strip() if (future or "").strip() else accumulated_source
        for future in futures
    ]
    alignment_pairs = [(full_src, candidate_full) for full_src in full_srcs]

    try:
        batch_alignments = simalign_v2.get_word_alignments_batch(
            alignment_pairs,
            align_model,
            align_tokenizer,
        )
    except Exception as e:
        return "", {"status": f"reject_align_error:{type(e).__name__}:{e}"}

    valid_cases: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, Any]] = []
    for future, full_src, alignments in zip(futures, full_srcs, batch_alignments):
        alignments = alignments or []
        if not alignments:
            skipped_cases.append({
                "future": future,
                "reason": "no_alignments",
            })
            continue

        safe_prefix = simalign_v2.truncate_by_alignment(
            full_src,
            accumulated_source,
            candidate_full,
            alignments,
        )
        if committed_norm and len(safe_prefix) < len(committed_norm):
            safe_prefix = committed_norm
        if committed_norm and not safe_prefix.startswith(committed_norm):
            skipped_cases.append({
                "future": future,
                "reason": "non_monotonic_safe_prefix",
            })
            continue

        valid_cases.append({
            "future": future,
            "alignment_count": len(alignments),
            "safe_prefix": safe_prefix,
            "safe_prefix_chars": len(normalize_zh(safe_prefix)),
        })

    if not valid_cases:
        return "", {
            "status": "reject_no_valid_alignment_cases",
            "candidate_full": candidate_full,
            "skipped_cases": skipped_cases,
        }

    accepted_case = min(valid_cases, key=lambda x: x["safe_prefix_chars"])
    accepted_full = normalize_zh(accepted_case["safe_prefix"])
    committed_chars = len(committed_norm)
    if len(accepted_full) <= committed_chars:
        return "", {
            "status": "reject_no_new_safe_chars",
            "candidate_full": candidate_full,
            "accepted_full": accepted_full,
            "accepted_case": accepted_case,
            "valid_cases": valid_cases,
            "skipped_cases": skipped_cases,
        }

    checked_delta = accepted_full[committed_chars:]
    return checked_delta, {
        "status": "applied",
        "candidate_full": candidate_full,
        "accepted_full": accepted_full,
        "accepted_case": accepted_case,
        "valid_cases": valid_cases,
        "skipped_cases": skipped_cases,
    }




# =============================================================================
# Base model: future sampling
# =============================================================================

def load_base_model(path: str, gpu_memory_utilization: float = 0.85) -> LLM:
    return LLM(
        model=path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def sample_futures(
    base_llm: LLM,
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    temperature: float,
) -> Tuple[List[str], List[str]]:
    """Sample N future English continuations from the base model."""
    if not (observed_source or "").strip():
        return [], []
    params = SamplingParams(
        temperature=temperature,
        max_tokens=future_tokens,
        n=num_futures,
        top_p=0.90,
        top_k=50,
        presence_penalty=0.0,
        stop=["\n"],
    )
    outputs = base_llm.generate([observed_source.strip()], params)
    futures: List[str] = []
    raw_outputs: List[str] = []
    seen = set()
    for out in outputs[0].outputs:
        raw = (out.text or "").strip()
        raw_outputs.append(raw)
        cleaned = clean_continuation(observed_source, raw)
        if cleaned:
            cleaned = truncate_future_to_first_sentence(cleaned)
            cleaned_key = cleaned.lower()
            if cleaned and cleaned_key not in seen:
                seen.add(cleaned_key)
                futures.append(cleaned)
    return futures, raw_outputs


# =============================================================================
# Thinking model: safe delta
# =============================================================================

def build_thinking_prompt(
    observed_source: str,
    futures: List[str],
    committed_chinese: str,
) -> str:
    """Build the user prompt for the thinking model (safe delta only)."""
    futures_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(futures) if (f or "").strip())
    committed_block = committed_chinese if committed_chinese else "(none yet)"
    return (
        "You are a professional simultaneous interpreter (English → Chinese). "
        "You see a PARTIAL English source and several POSSIBLE future continuations. "
        "You have already committed a partial Chinese translation (may be empty). "
        "Your task: output ONLY the additional Chinese text that is SAFE to emit now.\n\n"
        "The output may include:\n"
        "  (a) completion of older observed meaning that is still missing or unfinished in the committed Chinese, and\n"
        "  (b) further safe continuation,\n"
        "as long as the ENTIRE output remains valid under ALL possible futures.\n\n"
        "Rules:\n"
        "- Output a Chinese segment that can be APPENDED directly after the committed Chinese.\n"
        "- First identify the maximal SAFE Chinese segment that can be appended now.\n"
        "- It must first repair any missing or unfinished meaning already supported by the observed English, and only then continue further when that continuation remains valid under ALL possible futures.\n"
        "- The entire new text must remain consistent with ALL possible futures (no contradiction).\n"
        "- The committed Chinese may lag behind the observed English.\n"
        "- Do NOT assume the committed Chinese already covers all previously observed English.\n"
        "- Before translating newer material, check whether the committed Chinese ends in an unfinished word, phrase, or clause relative to the already observed English. If yes, complete that unfinished part first whenever it is safely supported.\n"
        "- If earlier observed meaning is still missing, do NOT jump ahead and translate only the newest boundary.\n"
        "- The new text must APPEND to the committed Chinese; do not revise, replace, or paraphrase already committed content.\n"
        "- Repair missing meaning only by continuing from the committed tail, not by rewriting earlier committed words.\n"
        "- Prefer the longest segment that is strictly supported by the observed English and ALL possible futures. Do NOT extend beyond what is jointly supported.\n"
        "- If a Chinese punctuation mark is already forced by the observed source and remains valid under ALL futures, include it now.\n"
        "- Do NOT output explanation, reasoning, or summary.\n"
        "- If no safe new Chinese characters can be appended now, output exactly: EMPTY\n\n"
        "Example 1:\n"
        "Observed English: he is\n"
        "Possible futures:\n"
        "  1. a worker at the local school.\n"
        "  2. a teacher who later became a principal.\n"
        "Committed Chinese: (none yet)\n"
        "Safe output: 他是\n\n"
        "Example 2:\n"
        "Observed English: I went to the\n"
        "Possible futures:\n"
        "  1. bank to deposit some cash.\n"
        "  2. beach to watch the sunset.\n"
        "Committed Chinese: 我去了\n"
        "Safe output: EMPTY\n\n"
        "Example 3:\n"
        "Observed English: it was over.\n"
        "Possible futures:\n"
        "  1. Then we left.\n"
        "  2. After that, we slept.\n"
        "Committed Chinese: 这件事结束了\n"
        "Safe output: 。\n\n"
        "Example 4:\n"
        "Observed English: he is the editor, and not\n"
        "Possible futures:\n"
        "  1. the author.\n"
        "  2. the author of the preface.\n"
        "Committed Chinese: 他\n"
        "Safe output: 是编辑，而不是作者\n\n"
        "Example 5:\n"
        "Observed English: she said that he was\n"
        "Possible futures:\n"
        "  1. innocent, and should be released.\n"
        "  2. innocent, but still under suspicion.\n"
        "Committed Chinese: 她说他\n"
        "Safe output: 是无辜的\n\n"
        "Partial English source so far:\n"
        f"{observed_source}\n\n"
        "Possible future continuations (any of these may follow):\n"
        f"{futures_block}\n\n"
        "Committed Chinese translation so far (do not repeat):\n"
        f"{committed_block}\n\n"
        "Output ONLY the next safe Chinese segment to emit, or EMPTY if none. "
        "Use natural Chinese punctuation as soon as it is safely determined."
    )


def call_thinking_model(
    client: OpenAI,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    """Call thinking model via chat endpoint; return content delta and debug payload."""
    messages = [{"role": "user", "content": user_content}]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    choice = resp.choices[0]
    message = choice.message
    raw_message_fields = _raw_message_debug_fields(message)
    reasoning_text, content_text = _split_reasoning_and_content(
        raw_message_fields["message.reasoning_content"] or raw_message_fields["message.reasoning"],
        raw_message_fields["message.content"],
    )
    delta = "" if (not content_text or content_text.upper() == "EMPTY") else normalize_zh(content_text)
    return delta, {
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": delta,
        "finish_reason": getattr(choice, "finish_reason", None),
    }


# =============================================================================
# End-of-utterance: force-complete remaining translation
# =============================================================================

def build_final_completion_prompt(full_source: str, committed_chinese: str) -> str:
    """Prompt to complete the rest of the translation from committed."""
    return (
        "You are a professional translator. Complete the Chinese translation of the following English text. "
        "A partial Chinese translation is already committed; output ONLY the continuation (the part that comes after).\n"
        "The final result must be fluent, natural Chinese with proper punctuation.\n"
        "If the already committed text is missing a comma or period right before the continuation, "
        "you may begin your continuation with that punctuation mark.\n\n"
        f"English (full):\n{full_source}\n\n"
        f"Chinese already committed (do NOT repeat):\n{committed_chinese or '(none)'}\n\n"
        "Output ONLY the new Chinese characters that follow the committed part. "
        "No explanation. Natural punctuation is required."
    )


def force_complete_translation(
    client: OpenAI,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    """Get final translation using chat endpoint; return full translation and debug payload."""
    messages = [{"role": "user", "content": build_final_completion_prompt(full_source, committed_chinese)}]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    message = choice.message
    raw_message_fields = _raw_message_debug_fields(message)
    _, content_text = _split_reasoning_and_content(
        raw_message_fields["message.reasoning_content"] or raw_message_fields["message.reasoning"],
        raw_message_fields["message.content"],
    )
    continuation = "" if (not content_text or content_text.upper() == "EMPTY") else normalize_zh(content_text)

    committed_norm = normalize_zh(committed_chinese)
    new_part = strip_committed_suffix_from_delta(committed_chinese, continuation)
    new_part = normalize_zh(new_part)
    full_translation = committed_norm + new_part if committed_chinese else continuation
    return full_translation, {
        "raw_message_fields": raw_message_fields,
        "reasoning_text": "",
        "content_text": content_text,
        "cleaned_content": continuation,
        "finish_reason": getattr(choice, "finish_reason", None),
        "full_translation": full_translation,
    }


# =============================================================================
# Process one utterance
# =============================================================================

def process_one_utterance(
    base_llm: LLM,
    client: OpenAI,
    align_model: Any,
    align_tokenizer: Any,
    thinking_model: str,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    row: Dict[str, str],
    args: argparse.Namespace,
    translation_cache: Optional[Dict[str, str]] = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run thinking-policy pipeline for one utterance.
    Returns dict with utt_id, source_future_sampling (chunks), target_future_sampling (deltas), actions, etc.
    """
    full_source = " ".join(sentences)
    n_chunks = len(trajectory)
    timing: Dict[str, float] = {
        "step1_future_sampling_s": 0.0,
        "step2_thinking_delta_s": 0.0,
        "step2_alignment_check_s": 0.0,
        "step3_final_complete_s": 0.0,
    }

    source_chunks: List[str] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    committed = ""
    accumulated_source = ""

    _vlog(verbose_log_file, f"\n{'#' * 60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source}")
    _vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# Thinking model: {thinking_model}")
    _vlog(verbose_log_file, f"{'#' * 60}")

    for chunk_idx, chunk in enumerate(trajectory):
        chunk_str = (chunk or "").strip()
        if chunk_str:
            accumulated_source = (accumulated_source + " " + chunk_str).strip()
        source_chunks.append(chunk_str)

        is_last = chunk_idx == n_chunks - 1

        _vlog(verbose_log_file, f"\n{'=' * 60}")
        _vlog(verbose_log_file, f"Chunk {chunk_idx + 1}/{n_chunks}: {chunk_str!r}")
        _vlog(verbose_log_file, f"  accumulated_source: {accumulated_source!r}")
        _vlog(verbose_log_file, f"  committed_before:  {committed!r}")

        # --- Last chunk: force-complete remainder ---
        if is_last:
            _vlog(verbose_log_file, f"  [last] force-complete from committed len={len(normalize_zh(committed))}")
            t0 = time.perf_counter()
            full_translation, final_debug = force_complete_translation(
                client, thinking_model, full_source, committed
            )
            timing["step3_final_complete_s"] += time.perf_counter() - t0
            _vlog(verbose_log_file, _format_verbose_paragraph("  [last] reasoning", final_debug["reasoning_text"]))
            if final_debug.get("raw_message_fields"):
                raw_fields = final_debug["raw_message_fields"]
                _vlog(verbose_log_file, f"  [last][raw] message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
                _vlog(verbose_log_file, f"  [last][raw] message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
                _vlog(verbose_log_file, f"  [last][raw] message.content: {raw_fields.get('message.content', '')!r}")
            _vlog(verbose_log_file, f"  [last] content_text: {final_debug['content_text']!r}")
            _vlog(verbose_log_file, f"  [last] cleaned_content: {final_debug['cleaned_content']!r}")
            _vlog(verbose_log_file, f"  [last] full_translation: {full_translation!r}")
            committed_norm = normalize_zh(committed)
            full_norm = normalize_zh(full_translation)
            if len(full_norm) > len(committed_norm):
                remaining = full_norm[len(committed_norm):]
                target_deltas.append(remaining)
                actions.append("WRITE")
                committed = full_translation
                _vlog(verbose_log_file, f"  -> WRITE (end) delta={remaining!r}")
            else:
                target_deltas.append("")
                actions.append("READ")
                _vlog(verbose_log_file, "  -> READ (end, nothing new)")
            continue

        # --- Future sampling ---
        t1_0 = time.perf_counter()
        futures, future_raw_outputs = sample_futures(
            base_llm,
            accumulated_source,
            args.num_futures,
            args.future_tokens,
            args.sample_temperature,
        )
        timing["step1_future_sampling_s"] += time.perf_counter() - t1_0

        _vlog(verbose_log_file, f"  step1_future_raw_outputs: {json.dumps(future_raw_outputs, ensure_ascii=False, indent=2)}")
        _vlog(verbose_log_file, f"  step1_futures_cleaned: {json.dumps(futures, ensure_ascii=False, indent=2)}")

        if len(futures) < 2:
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        # --- Thinking model: safe delta ---
        user_content = build_thinking_prompt(accumulated_source, futures, committed)
        t2_0 = time.perf_counter()
        delta, thinking_debug = call_thinking_model(
            client,
            thinking_model,
            user_content,
            committed_chinese=committed,
            temperature=args.thinking_temperature,
            max_tokens=args.thinking_max_tokens,
        )
        timing["step2_thinking_delta_s"] += time.perf_counter() - t2_0
        _vlog(verbose_log_file, _format_verbose_paragraph("  step2_reasoning", thinking_debug["reasoning_text"]))
        if thinking_debug.get("raw_message_fields"):
            raw_fields = thinking_debug["raw_message_fields"]
            _vlog(verbose_log_file, f"  step2_raw_message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
            _vlog(verbose_log_file, f"  step2_raw_message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
            _vlog(verbose_log_file, f"  step2_raw_message.content: {raw_fields.get('message.content', '')!r}")
        _vlog(verbose_log_file, f"  step2_content_text: {thinking_debug['content_text']!r}")
        _vlog(verbose_log_file, f"  step2_cleaned_content: {thinking_debug['cleaned_content']!r}")
        raw_delta = delta
        t2a_0 = time.perf_counter()
        delta, alignment_debug = apply_simalign_delta_check(
            accumulated_source=accumulated_source,
            futures=futures,
            committed=committed,
            delta=delta,
            align_model=align_model,
            align_tokenizer=align_tokenizer,
        )
        timing["step2_alignment_check_s"] += time.perf_counter() - t2a_0
        _vlog(verbose_log_file, f"  step2_delta_raw: {raw_delta!r}")
        _vlog(verbose_log_file, f"  step2_alignment_check_status: {alignment_debug.get('status', '')!r}")
        if alignment_debug.get("accepted_case"):
            accepted_case = alignment_debug["accepted_case"]
            _vlog(verbose_log_file, f"  step2_alignment_selected_future: {accepted_case.get('future', '')!r}")
            _vlog(verbose_log_file, f"  step2_alignment_selected_safe_prefix: {accepted_case.get('safe_prefix', '')!r}")
        if alignment_debug.get("valid_cases"):
            compact_valid = [
                {
                    "future": c.get("future", ""),
                    "alignment_count": c.get("alignment_count", 0),
                    "safe_prefix": c.get("safe_prefix", ""),
                    "safe_prefix_chars": c.get("safe_prefix_chars", 0),
                }
                for c in alignment_debug["valid_cases"]
            ]
            _vlog(verbose_log_file, f"  step2_alignment_valid_cases: {json.dumps(compact_valid, ensure_ascii=False, indent=2)}")
        if alignment_debug.get("skipped_cases"):
            _vlog(verbose_log_file, f"  step2_alignment_skipped_cases: {json.dumps(alignment_debug['skipped_cases'], ensure_ascii=False, indent=2)}")
        _vlog(verbose_log_file, f"  step2_delta_checked: {delta!r}")

        if delta:
            target_deltas.append(delta)
            actions.append("WRITE")
            committed = (committed or "") + delta
            _vlog(verbose_log_file, f"  -> WRITE delta={delta!r}")
        else:
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (empty/invalid content)")

    system_output = "".join(d for d in target_deltas if d)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source,
        "input_sentences": sentences,
        "source_future_sampling": source_chunks,
        "target_future_sampling": target_deltas,
        "actions": actions,
        "system_output_text": system_output,
        "config": {
            "version": "thinking_policy",
            "num_futures": args.num_futures,
            "future_tokens": args.future_tokens,
            "thinking_model": thinking_model,
        },
        "timing": timing,
    }

    laal_reference_text = ""
    laal_value = float("nan")
    laal_error: Optional[str] = None
    bleu_char_value = float("nan")
    bleu_char_error: Optional[str] = None
    laal_reference_mode = "manifest_reference"

    ref_text = (translation_cache or {}).get(utt_id)
    if ref_text:
        laal_reference_mode = "cache"
    else:
        ref_text = _get_reference_from_row(row)
    if ref_text is not None:
        laal_reference_text = ref_text
        result["reference_text"] = ref_text
        try:
            laal_value = compute_laal(
                source_chunks,
                target_deltas,
                actions,
                ref_text,
            )
            bleu_char_value = compute_bleu_char(
                system_output,
                ref_text,
            )
        except Exception as e:
            laal_error = str(e)
            bleu_char_error = str(e)
    else:
        laal_error = "reference_text_unavailable"
        bleu_char_error = "reference_text_unavailable"

    result["laal_reference_text"] = laal_reference_text
    result["metrics"] = {
        "laal_text": laal_value,
        "laal_reference_mode": laal_reference_mode,
        "bleu_char": bleu_char_value,
        "bleu_reference_mode": laal_reference_mode,
        "effective_source_chunks": sum(1 for c in source_chunks if str(c).strip()),
        "system_output_chars": len(system_output),
        "reference_chars": len(laal_reference_text.replace(" ", "")) if laal_reference_text else 0,
        "laal_error": laal_error,
        "bleu_char_error": bleu_char_error,
    }
    return result


def _get_reference_from_row(row: Dict[str, str]) -> Optional[str]:
    for key in ("tgt_text_full", "tgt_text", "target_text", "translation", "ref_text", "reference"):
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        vals = parse_list_column(raw)
        if vals:
            text = "".join(str(v).strip() for v in vals if str(v).strip())
        else:
            text = str(raw).strip()
        if text:
            return text
    return None


def load_translation_cache(cache_dir: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not cache_dir or not os.path.isdir(cache_dir):
        return cache
    jsonl_files = sorted(glob.glob(os.path.join(cache_dir, "task_*.jsonl")))
    print(f"[Cache] Loading translation cache from {cache_dir} ({len(jsonl_files)} files) ...")
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    uid = str(entry.get("utt_id", "")).strip()
                    tl = str(entry.get("llm_full_translation", "")).strip()
                    if uid and tl:
                        cache[uid] = tl
                except Exception:
                    pass
    print(f"[Cache] Loaded {len(cache)} entries.")
    return cache


# =============================================================================
# Data I/O
# =============================================================================

def iter_assigned_rows(
    input_tsv: str,
    task_id: int,
    num_tasks: int,
    max_rows: Optional[int] = None,
):
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row
            count += 1
            if max_rows is not None and count >= max_rows:
                break


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    n = 0
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        for row_idx, _ in enumerate(csv.DictReader(f, delimiter="\t")):
            if row_idx % num_tasks == task_id:
                n += 1
    return n


def get_one_row_by_id(
    input_tsv: str,
    utt_id: str,
    id_column: str = "id",
) -> Optional[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if str(row.get(id_column, "")).strip() == str(utt_id).strip():
                return row_idx, row
    return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_env()

    os.makedirs(args.output_root, exist_ok=True)

    try:
        import llm_future_sampling_core_v2 as simalign_v2
    except Exception as e:
        print(f"ERROR: Cannot import simalign helpers from llm_future_sampling_core_v2.py: {e}")
        sys.exit(1)

    # Thinking model (OpenAI-compatible)
    client = OpenAI(base_url=args.thinking_api_base, api_key="dummy")
    try:
        _ = client.models.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to thinking API at {args.thinking_api_base}: {e}")
        sys.exit(1)

    # Alignment model for post-output safety check
    print(f"[Align] Loading simalign on {args.align_device} ...")
    align_model, align_tokenizer = simalign_v2.load_align_model(
        cache_dir=os.environ.get("HF_HOME"),
        device=args.align_device,
    )
    print("[Align] Loaded.")

    # Base model for future sampling
    print(f"[Base] Loading {args.base_model_path} ...")
    base_llm = load_base_model(args.base_model_path, args.gpu_memory_utilization)
    print("[Base] Loaded.")

    # Row iteration (test_one => verbose for per-step debug)
    if args.test_one:
        args.verbose = True
        if args.utt_id:
            one = get_one_row_by_id(args.input_tsv, args.utt_id, args.id_column)
            if one is None:
                print(f"utt-id '{args.utt_id}' not found.")
                return
            row_list = [one]
        else:
            row_list = list(iter_assigned_rows(
                args.input_tsv, args.task_id, args.num_tasks, max_rows=1
            ))
        total = len(row_list)
    else:
        total = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        if args.max_rows is not None:
            total = min(total, args.max_rows)
        row_list = list(iter_assigned_rows(
            args.input_tsv, args.task_id, args.num_tasks, max_rows=args.max_rows
        ))

    if not row_list:
        print("No rows to process.")
        return

    print(f"[Task {args.task_id}] Processing {total} rows | thinking={args.thinking_model_name} | N={args.num_futures}")
    translation_cache = load_translation_cache(DEFAULT_TRANSLATION_CACHE_DIR)

    written = 0
    for row_idx, row in tqdm(row_list, desc="utterances"):
        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx}"
        out_path = os.path.join(args.output_root, f"{sanitize_filename(utt_id)}.json")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        sentences = parse_list_column(row.get("src_text_full"))
        trajectory = parse_list_column(row.get("src_trajectory"))
        if not sentences:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": "Empty src_text_full"}, f, ensure_ascii=False, indent=2)
            written += 1
            continue
        if not trajectory:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": "Empty src_trajectory"}, f, ensure_ascii=False, indent=2)
            written += 1
            continue

        try:
            verbose_log_file = None
            if args.verbose:
                verbose_log_path = os.path.join(
                    args.output_root,
                    f"verbose_{sanitize_filename(utt_id)}.log",
                )
                raw_file = open(verbose_log_path, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file)
                _vlog(verbose_log_file, f"[VerboseLog] writing to {verbose_log_path}")
            try:
                result = process_one_utterance(
                    base_llm,
                    client,
                    align_model,
                    align_tokenizer,
                    args.thinking_model_name,
                    utt_id,
                    sentences,
                    trajectory,
                    row,
                    args,
                    translation_cache=translation_cache,
                    verbose_log_file=verbose_log_file,
                )
            finally:
                if verbose_log_file is not None:
                    verbose_log_file.close()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": str(e)}, f, ensure_ascii=False, indent=2)
            written += 1

    print(f"[Task {args.task_id}] Done. Written {written} outputs to {args.output_root}")


if __name__ == "__main__":
    main()
