#!/usr/bin/env python3
"""Local Agreement (LA-N) rule-based simultaneous MT baseline.

Reference: TAF paper (arXiv 2410.22499), Section 3 baselines.

At each step, generate a full translation hypothesis (prefix-constrained
to committed text) using an offline MT model.  Commit the longest common
prefix (LCP) of the last N hypotheses.  At the final chunk, force-complete.

Usage:
  python local_agreement.py \
    --mt-api-base http://localhost:8100 \
    --mt-api-model qwen3-instruct \
    --mt-tokenizer-path /path/to/tokenizer \
    --la-n 2 --segment-size 1 --target-lang Chinese \
    --test-one --overwrite \
    --output-jsonl output/la_n2_seg1.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

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
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LA-N simultaneous MT baseline.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # MT model
    p.add_argument("--mt-tokenizer-path", required=True)
    p.add_argument("--mt-api-base", required=True)
    p.add_argument("--mt-api-model", default=os.environ.get("MT_API_MODEL", "qwen3-instruct"))
    p.add_argument("--mt-api-timeout", type=float, default=120.0)
    # LA parameters
    p.add_argument("--la-n", type=int, default=2)
    p.add_argument("--segment-size", type=int, default=1)
    p.add_argument("--lcp-mode", choices=["char", "word"], default="char")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--target-lang", default="Chinese")
    # Output / row selection
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def parse_trajectory(raw: str) -> List[str]:
    return ast.literal_eval(raw)


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


def build_source_observed(chunks: List[str], t: int) -> str:
    return join_source_chunks(chunks[: t + 1])


def get_full_source_text(row: Dict[str, Any]) -> str:
    raw = row.get("src_text")
    if raw is None or pd.isna(raw):
        raise ValueError("src_text missing")
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("src_text is empty")
    return text


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
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
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
# Translation (MT model calls)
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: Any, source: str, target_lang: str,
                  committed_text: str = "") -> str:
    """Build a translation prompt, prefix-constrained if committed_text is given."""
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


def translate_source_prefix(
    tokenizer: Any, source_observed: str, committed_text: str,
    api_base: str, api_model: str, api_timeout: float,
    max_new_tokens: int, target_lang: str,
) -> str:
    """Generate full hypothesis, prefix-constrained to committed_text if non-empty."""
    has_committed = bool(str(committed_text or "").strip())
    prompt = _build_prompt(tokenizer, source_observed, target_lang, committed_text)
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


def force_complete_translation(
    tokenizer: Any, full_source: str, committed_text: str,
    api_base: str, api_model: str, api_timeout: float,
    max_new_tokens: int, target_lang: str,
) -> str:
    """Generate remaining translation beyond committed_text using full source."""
    prompt = _build_prompt(tokenizer, full_source, target_lang, committed_text)
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
        return ""
    return clean_model_text(str(choices[0].get("text", "")))


# ---------------------------------------------------------------------------
# LCP
# ---------------------------------------------------------------------------

def longest_common_prefix_chars(hypotheses: List[str]) -> str:
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]
    prefix: List[str] = []
    for chars in zip(*hypotheses):
        if len(set(chars)) == 1:
            prefix.append(chars[0])
        else:
            break
    return "".join(prefix)


def longest_common_prefix_words(hypotheses: List[str]) -> str:
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]
    token_lists = [h.split() for h in hypotheses]
    prefix: List[str] = []
    for tokens in zip(*token_lists):
        if len(set(tokens)) == 1:
            prefix.append(tokens[0])
        else:
            break
    return " ".join(prefix)


def compute_lcp(hypotheses: List[str], mode: str = "char") -> str:
    if mode == "word":
        return longest_common_prefix_words(hypotheses)
    return longest_common_prefix_chars(hypotheses)


# ---------------------------------------------------------------------------
# Core: run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(row: Dict[str, Any], args: argparse.Namespace,
                      mt_tokenizer: Any) -> Dict[str, Any]:
    """LA-N core loop.

    At each step, translate the observed source prefix (prefix-constrained),
    maintain a sliding window of the last N hypotheses, and commit the LCP
    when it extends beyond the current committed text.
    """
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = parse_trajectory(row["src_trajectory"])
    full_source_text = get_full_source_text(row)
    n_chunks = len(chunks)

    committed_text = ""
    target_deltas: List[str] = [""] * n_chunks
    actions: List[str] = ["READ"] * n_chunks
    recent_hypotheses: List[str] = []

    step_start = 0
    while step_start < n_chunks:
        step_end = min(step_start + args.segment_size, n_chunks)
        is_last_step = (step_end == n_chunks)
        last_chunk_idx = step_end - 1
        source_observed = build_source_observed(chunks, last_chunk_idx)

        if is_last_step:
            delta = force_complete_translation(
                mt_tokenizer, full_source_text, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            if delta:
                committed_text += delta
            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = "WRITE" if delta else "READ"

        elif not source_observed.strip():
            step_start = step_end
            continue

        else:
            hyp = translate_source_prefix(
                mt_tokenizer, source_observed, committed_text,
                args.mt_api_base, args.mt_api_model, args.mt_api_timeout,
                args.max_new_tokens, args.target_lang,
            )
            recent_hypotheses.append(hyp)
            if len(recent_hypotheses) > args.la_n:
                recent_hypotheses.pop(0)

            lcp = compute_lcp(recent_hypotheses, mode=args.lcp_mode)

            if lcp.startswith(committed_text) and len(lcp) > len(committed_text):
                delta = lcp[len(committed_text):]
                committed_text = lcp
            else:
                delta = ""

            target_deltas[last_chunk_idx] = delta
            actions[last_chunk_idx] = "WRITE" if delta else "READ"

        step_start = step_end

    return {
        "utt_id": utt_id,
        "src_trajectory": chunks,
        "source_full_text": full_source_text,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {
            "method": "local_agreement",
            "la_n": args.la_n,
            "segment_size": args.segment_size,
            "lcp_mode": args.lcp_mode,
        },
    }


# ---------------------------------------------------------------------------
# Row selection & main
# ---------------------------------------------------------------------------

def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_column].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected
    if args.test_one:
        return df.iloc[[args.row_idx]]
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max(1, int(args.max_rows)))
    return df.iloc[start:end]


def main() -> None:
    setup_env()
    args = parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)
    print(f"Processing {len(rows)} row(s)  LA-{args.la_n}  segment_size={args.segment_size}")

    models = verify_api(args.mt_api_base, args.mt_api_timeout)
    if args.mt_api_model not in models:
        raise RuntimeError(f"model '{args.mt_api_model}' not found; available={models}")
    print(f"[MT] model={args.mt_api_model}  api={normalize_api_base(args.mt_api_base)}")

    mt_tokenizer = load_tokenizer(args.mt_tokenizer_path)

    out_fh = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        out_fh = open(args.output_jsonl, "w" if args.overwrite else "a", encoding="utf-8")

    for _, row in rows.iterrows():
        result = run_one_utterance(row.to_dict(), args, mt_tokenizer)
        print(f"  {result['utt_id']}")
        if out_fh:
            out_fh.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            out_fh.flush()

    if out_fh:
        out_fh.close()
    print("Done.")


if __name__ == "__main__":
    main()
