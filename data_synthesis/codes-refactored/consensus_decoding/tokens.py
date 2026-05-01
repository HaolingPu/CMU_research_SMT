"""Token-ID-level utilities: filtering, decoding, suspicious-content detection,
and trimming the pending buffer to a clean Unicode boundary before commit.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _disallowed_generation_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    for frag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if frag in token_text:
            return "special_token_text"
    if re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    # NOTE: Do NOT filter replacement_char (U+FFFD) here. Byte-level BPE tokens
    # for rare characters (e.g. Japanese kanji "繰") decode to U+FFFD individually
    # but combine into valid characters when decoded together in the pending buffer.
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width_or_variation_selector"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control_or_surrogate"
    return None


def filter_distribution_token_ids(
    tokenizer: Any,
    id_distribution: Dict[int, float],
    dist_debug: Dict[str, Any],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    filtered: Dict[int, float] = {}
    removed: List[Dict[str, Any]] = []
    for tok_id, prob in id_distribution.items():
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None:
            removed.append({"token_id": int(tok_id), "token_text": _single_token_text(tokenizer, tok_id),
                            "prob": round(float(prob), 6), "reason": reason})
            continue
        filtered[int(tok_id)] = float(prob)

    out_debug = dict(dist_debug)
    out_debug["filtered_disallowed_tokens"] = removed
    if filtered:
        out_debug["topk_token_ids"] = list(filtered.keys())
        out_debug["topk_token_texts"] = [_single_token_text(tokenizer, t) for t in filtered]
        out_debug["topk_true_probs"] = [round(float(filtered[t]), 6) for t in filtered]
    elif removed and out_debug.get("reason") == "ok":
        out_debug["reason"] = "all_top_tokens_filtered_as_disallowed"
    return filtered, out_debug


# ---------------------------------------------------------------------------
# Pending-buffer management
# ---------------------------------------------------------------------------

def decode_token_ids_to_text(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def inspect_token_ids(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
    # Decode the entire token ID buffer to text for debugging and logging.  This is NOT used for any model input or state management, to avoid drift from re-encoding incomplete token sequences.
    decoded_text = decode_token_ids_to_text(tokenizer, token_ids)
    last_token_id = token_ids[-1] if token_ids else None
    last_token_text = _single_token_text(tokenizer, last_token_id) if last_token_id is not None else ""
    return {"decoded_text": decoded_text, "last_token_id": last_token_id, "last_token_text": last_token_text}


def has_suspicious_content(text: str) -> bool:
    if not text:
        return False
    # Check the ENTIRE decoded text for replacement characters, not just tail.
    # Byte-level BPE tokens may form broken sequences anywhere in the buffer
    # when consensus picks a byte token but fails to pick its pair.
    if "\ufffd" in text or "�" in text:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_pending_token_ids(
    tokenizer: Any, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]]]:
# Filter out any disallowed tokens from the pending token ID buffer, and log the removed tokens with reasons.  This ensures that we never commit or feed back into the model any token that could cause decoding issues or model confusion.
    kept: List[int] = []
    removed: List[Dict[str, Any]] = []
    for idx, tok_id in enumerate(pending_token_ids):
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is None:
            kept.append(tok_id)
        else:
            removed.append({"position": idx, "token_id": int(tok_id),
                            "token_text": _single_token_text(tokenizer, tok_id), "reason": reason})
    return kept, removed


def trim_pending_tokens_to_complete_boundary(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[int], Dict[str, Any]]:
    work, removed_disallowed = sanitize_pending_token_ids(tokenizer, pending_token_ids)
    if not work:
        return [], removed_disallowed, [], {
            "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
        }
    # Commit only an exact prefix of the original token-id buffer.
    # This preserves the true model state and avoids drift from re-encoding text
    # such as ["而", "且"] into a merged token like ["而且"].
    last_committable_idx = -1 #从前往后逐个decode，找到最后一个"解码后没有乱码"的位置
    for i in range(len(work)):
        partial = decode_token_ids_to_text(tokenizer, work[:i + 1])
        if not has_suspicious_content(partial): #没有U+FFFD等异常字符，说明到这里的token组合是完整的
            last_committable_idx = i
    if last_committable_idx >= 0:
        clean_ids = work[:last_committable_idx + 1]
        removed_tail = work[last_committable_idx + 1:]
        view = inspect_token_ids(tokenizer, clean_ids)
        view["full_text"] = committed_text + view["decoded_text"]
        return clean_ids, removed_disallowed, removed_tail, view
    return [], removed_disallowed, work, {
        "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
    }


def finalize_external_commit(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[str, str, List[int], Dict[str, Any]]:
    trimmed, removed_disallowed, removed_tail, view = trim_pending_tokens_to_complete_boundary(
        tokenizer, committed_text, pending_token_ids)
    commit_text = view["decoded_text"]
    return committed_text + commit_text, commit_text, trimmed, {
        "pending_before_trim": decode_token_ids_to_text(tokenizer, pending_token_ids),
        "pending_after_trim": commit_text,
        "removed_disallowed_tokens": removed_disallowed,
        "removed_tail_text": decode_token_ids_to_text(tokenizer, removed_tail),
    }
