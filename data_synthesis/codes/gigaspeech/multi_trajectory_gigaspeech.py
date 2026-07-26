#!/usr/bin/env python3
"""
Build streaming trajectory JSONs for GigaSpeech using:
  - merged LLM outputs (with Source/Target schema and propagated src_trajectory)
  - the manifest's src_trajectory itself as the timing grid (no MFA)

The src_trajectory column is a list of streaming ASR chunks emitted on a fixed
~960 ms grid. We treat each entry's index as its emission window and align the
LLM Source chunks back to those windows by greedy token matching.

This replaces the previous MFA TextGrid alignment step entirely.
"""

import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


DEFAULT_CHUNK_MS = 960

# Target languages whose written form uses inter-word spaces.
SPACE_JOIN_LANGS = {"de", "en", "fr", "es"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate streaming trajectories from LLM output + manifest src_trajectory "
            "for GigaSpeech (no MFA required)."
        )
    )
    parser.add_argument("--llm-dir", required=True, help="LLM JSON directory (recursive).")
    parser.add_argument("--output-dir", required=True, help="Output directory for trajectory JSON files.")
    parser.add_argument(
        "--good-jsonl",
        default=None,
        help=(
            "Optional good-list jsonl path. If omitted, every JSON under --llm-dir "
            "that has a non-empty src_trajectory will be processed."
        ),
    )
    parser.add_argument(
        "--src-trajectory-key",
        default="src_trajectory",
        help="Top-level field name in the LLM JSON that holds the trajectory list.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help=(
            "Streaming chunk duration in milliseconds. Used only for output bookkeeping; "
            "the actual emission grid is determined by src_trajectory entry indices."
        ),
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write files directly under output-dir (default: subdir by recording id).",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Only process first N ids.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    n = normalize_text(s)
    return n.split() if n else []


def parse_trajectory(value: Any) -> List[str]:
    """Coerce the manifest src_trajectory cell into a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]

    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        # Not a python-literal list; treat as a single chunk.
        return [raw]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def build_trajectory_token_grid(traj: List[str]) -> Tuple[List[str], List[int]]:
    """
    Flatten the trajectory into a per-token sequence and remember which trajectory
    chunk (== emission window index) each token belongs to.

    Returns:
        tokens         : flat list of normalized tokens
        chunk_of_token : same length; chunk_of_token[i] is the trajectory-entry index
                         (== emission window index) that contributed token i
    """
    tokens: List[str] = []
    chunk_of_token: List[int] = []
    for ci, chunk in enumerate(traj):
        for tok in tokenize(chunk):
            tokens.append(tok)
            chunk_of_token.append(ci)
    return tokens, chunk_of_token


def align_llm_chunks_to_trajectory(
    llm_chunks: List[str],
    traj_tokens: List[str],
    chunk_of_token: List[int],
) -> List[Optional[int]]:
    """
    For each LLM Source chunk, find the trajectory chunk index at which the LLM
    chunk's last matched token appears. That index is the streaming window in
    which the chunk becomes safe to emit.

    Mono-global cursor: the same cursor advances across all LLM chunks so we
    never re-match earlier trajectory positions (handles repeated words).

    Returns: list of emit_chunk_index per LLM chunk (None if nothing matched).
    """
    results: List[Optional[int]] = []
    cursor = 0

    for chunk in llm_chunks:
        toks = tokenize(chunk)
        last_matched_pos: Optional[int] = None

        for t in toks:
            for i in range(cursor, len(traj_tokens)):
                if traj_tokens[i] == t:
                    last_matched_pos = i
                    cursor = i + 1
                    break
            # If this token never matched, just keep going; we still emit on whatever
            # we already matched. Strict pre-filtering should catch egregious cases.

        if last_matched_pos is None:
            results.append(None)
        else:
            results.append(chunk_of_token[last_matched_pos])

    return results


def assign_chunks_by_window(
    emit_indices: List[Optional[int]],
    num_windows: int,
) -> List[Dict[str, Any]]:
    """
    Group LLM-chunk indices by the trajectory window in which they're emitted.

    `emit_indices[i]` is the trajectory window index (0..num_windows-1) at which
    LLM chunk `i` becomes ready. Any chunk that did not match is appended to the
    last window so no source/target content is dropped.
    """
    windows: List[List[int]] = [[] for _ in range(max(num_windows, 1))]
    unmatched: List[int] = []

    for li, w in enumerate(emit_indices):
        if w is None or w < 0:
            unmatched.append(li)
            continue
        if w >= len(windows):
            w = len(windows) - 1
        windows[w].append(li)

    if unmatched:
        windows[-1].extend(unmatched)

    return [{"chunk_index": i, "emit_idx": idxs} for i, idxs in enumerate(windows)]


def build_final_segments(
    timeline: List[Dict[str, Any]],
    src_chunks: List[str],
    tgt_chunks: List[str],
    tgt_sep: str,
) -> Tuple[List[str], List[str]]:
    sources, targets = [], []
    for entry in timeline:
        idxs = entry["emit_idx"]
        s = " ".join(src_chunks[i] for i in idxs).strip()
        t = tgt_sep.join(tgt_chunks[i] for i in idxs).strip()
        sources.append(s)
        targets.append(t)
    return sources, targets


def target_join_sep(target_lang: Optional[str]) -> str:
    if target_lang and target_lang.lower() in SPACE_JOIN_LANGS:
        return " "
    return ""


def list_files(root: str, suffix: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(suffix):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def build_index(root: str, suffix: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for path in list_files(root, suffix):
        utt_id = os.path.basename(path)[: -len(suffix)]
        if utt_id not in idx:
            idx[utt_id] = path
    return idx


def load_good_ids(good_jsonl: str) -> List[str]:
    good_ids: List[str] = []
    with open(good_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utt = obj.get("file") or obj.get("utt_id")
            if isinstance(utt, str) and utt.strip():
                good_ids.append(utt.strip())
    seen: Set[str] = set()
    ordered = []
    for x in good_ids:
        if x in seen:
            continue
        seen.add(x)
        ordered.append(x)
    return ordered


def recording_id_from_utt(utt_id: str) -> str:
    if "_" not in utt_id:
        return utt_id
    return utt_id.rsplit("_", 1)[0]


def output_path_for_utt(output_dir: str, utt_id: str, flat_output: bool) -> str:
    if flat_output:
        out_dir = output_dir
    else:
        out_dir = os.path.join(output_dir, recording_id_from_utt(utt_id))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{utt_id}.json")


def main() -> None:
    args = parse_args()
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be > 0")

    chunk_seconds = args.chunk_ms / 1000.0
    os.makedirs(args.output_dir, exist_ok=True)

    llm_index = build_index(args.llm_dir, ".json")

    if args.good_jsonl:
        ids = load_good_ids(args.good_jsonl)
    else:
        ids = sorted(llm_index.keys())

    if args.max_items is not None:
        ids = ids[: args.max_items]

    print(f"chunk_ms={args.chunk_ms} ({chunk_seconds:.3f}s) — alignment grid is src_trajectory itself")
    print(f"ids={len(ids)}, llm_index={len(llm_index)}")

    ok = 0
    skipped_missing = 0
    skipped_existing = 0
    failed = 0

    for utt_id in tqdm(ids, desc="Trajectory"):
        llm_path = llm_index.get(utt_id)
        out_path = output_path_for_utt(args.output_dir, utt_id, args.flat_output)

        if llm_path is None:
            skipped_missing += 1
            continue

        if os.path.exists(out_path) and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            with open(llm_path, "r", encoding="utf-8") as f:
                seg = json.load(f)

            if not isinstance(seg, dict):
                raise ValueError("LLM JSON is not dict")
            if "error" in seg:
                raise ValueError(f"LLM JSON has error field: {seg.get('error')}")

            traj = parse_trajectory(seg.get(args.src_trajectory_key))
            if not traj:
                raise ValueError(f"Empty/missing src_trajectory under key '{args.src_trajectory_key}'")

            traj_tokens, chunk_of_token = build_trajectory_token_grid(traj)
            if not traj_tokens:
                raise ValueError("src_trajectory contains no tokens after normalization")

            target_lang = seg.get("target_lang") or seg.get("tgt_lang")
            tgt_sep = target_join_sep(target_lang)

            if "offline" in seg:
                levels = ["offline"]
            else:
                levels = ["low_latency", "medium_latency", "high_latency"]

            out: Dict[str, Any] = {
                "utt_id": utt_id,
                "original_text": seg.get("input", ""),
                "target_lang": target_lang,
                "chunk_ms": args.chunk_ms,
                "num_windows": len(traj),
            }

            for level in levels:
                level_obj = seg.get(level)
                if not isinstance(level_obj, dict):
                    raise ValueError(f"Missing or non-dict level: {level}")
                src_chunks = level_obj.get("Source", [])
                tgt_chunks = level_obj.get("Target", [])
                if len(src_chunks) != len(tgt_chunks):
                    raise ValueError(
                        f"{level}: Source/Target length mismatch "
                        f"({len(src_chunks)} vs {len(tgt_chunks)})"
                    )

                emit_indices = align_llm_chunks_to_trajectory(
                    src_chunks, traj_tokens, chunk_of_token
                )
                timeline = assign_chunks_by_window(emit_indices, num_windows=len(traj))
                src_out, tgt_out = build_final_segments(
                    timeline, src_chunks=src_chunks, tgt_chunks=tgt_chunks, tgt_sep=tgt_sep
                )

                out[f"source_{level}"] = src_out
                out[f"target_{level}"] = tgt_out

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            ok += 1
        except Exception as e:
            failed += 1
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utt_id": utt_id,
                        "error": str(e),
                        "llm_path": llm_path,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print("\n========== Done ==========")
    print(f"processed_ok    : {ok}")
    print(f"skipped_missing : {skipped_missing}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"failed          : {failed}")
    print(f"output_dir      : {args.output_dir}")


if __name__ == "__main__":
    main()
