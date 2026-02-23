#!/usr/bin/env python3
"""
Build streaming trajectory JSONs for GigaSpeech using:
  - merged LLM outputs
  - MFA TextGrid alignments
  - strict good-id list

Important: default chunk size is 960 ms (0.96 s), not 1.0 s.
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import tgt
from tqdm import tqdm


DEFAULT_CHUNK_MS = 960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate streaming trajectories from LLM + MFA for GigaSpeech."
    )
    parser.add_argument("--llm-dir", required=True, help="LLM JSON directory (recursive).")
    parser.add_argument("--mfa-dir", required=True, help="MFA TextGrid directory (recursive).")
    parser.add_argument("--good-jsonl", required=True, help="Good-list jsonl path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for trajectory JSON files.")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="Streaming chunk duration in milliseconds (default: 960).",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write files directly under output-dir (default: subdir by recording id).",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Only process first N good ids.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_word_alignment(textgrid_path: str) -> Tuple[List[Dict[str, Any]], float]:
    tg = tgt.read_textgrid(textgrid_path)
    audio_duration = float(tg.end_time)
    words = tg.get_tier_by_name("words").intervals
    out = []
    for w in words:
        raw = (w.text or "").strip()
        if not raw:
            continue
        out.append(
            {
                "word": raw,
                "start": float(w.start_time),
                "end": float(w.end_time),
            }
        )
    return out, audio_duration


def match_llm_chunks_to_mfa(llm_chunks: List[str], mfa_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mono-global matching:
    Keep one cursor across all chunks to prevent matching repeated words to earlier positions.
    """
    mfa_tokens = [normalize_text(w["word"]) for w in mfa_words]
    results = []
    cursor = 0

    for chunk in llm_chunks:
        tokens = normalize_text(chunk).split()
        matched = []

        for t in tokens:
            found = False
            for i in range(cursor, len(mfa_tokens)):
                if mfa_tokens[i] == t:
                    matched.append(i)
                    cursor = i + 1
                    found = True
                    break
            if not found:
                # Keep scanning next token; strict filter should catch bad cases before this step.
                continue

        if not matched:
            results.append({"start": None, "end": None})
            continue

        results.append(
            {
                "start": mfa_words[matched[0]]["start"],
                "end": mfa_words[matched[-1]]["end"],
            }
        )

    return results


def assign_chunks_by_window(
    aligned_chunks: List[Dict[str, Any]],
    chunk_seconds: float,
    audio_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    valid = [x for x in aligned_chunks if x["end"] is not None]
    if not valid:
        return [{"chunk_index": 0, "emit_idx": []}]

    last_word_end = max(x["end"] for x in valid)
    max_time = audio_duration if (audio_duration is not None and audio_duration > last_word_end) else last_word_end
    eps = 1e-6
    max_chunk_idx = int(math.ceil((max_time / chunk_seconds) - eps))

    emitted: Set[int] = set()
    timeline = []

    for idx in range(max_chunk_idx):
        chunk_end = (idx + 1) * chunk_seconds
        emit_idx = []

        for i, item in enumerate(aligned_chunks):
            if i in emitted:
                continue
            if item["end"] is not None and item["end"] <= chunk_end:
                emit_idx.append(i)
                emitted.add(i)

        timeline.append({"chunk_index": idx, "emit_idx": emit_idx})

    return timeline


def build_final_segments(
    timeline: List[Dict[str, Any]],
    eng_chunks: List[str],
    zh_chunks: List[str],
) -> Tuple[List[str], List[str]]:
    sources = []
    targets = []

    for entry in timeline:
        idxs = entry["emit_idx"]
        src = " ".join(eng_chunks[i] for i in idxs).strip()
        tgt = "".join(zh_chunks[i] for i in idxs).strip()
        sources.append(src)
        targets.append(tgt)

    return sources, targets


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
    # keep order, dedup
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
    mfa_index = build_index(args.mfa_dir, ".TextGrid")
    good_ids = load_good_ids(args.good_jsonl)
    if args.max_items is not None:
        good_ids = good_ids[: args.max_items]

    print(f"chunk_ms={args.chunk_ms} ({chunk_seconds:.3f}s)")
    print(f"good_ids={len(good_ids)}, llm_index={len(llm_index)}, mfa_index={len(mfa_index)}")

    ok = 0
    skipped_missing = 0
    skipped_existing = 0
    failed = 0

    for utt_id in tqdm(good_ids, desc="Trajectory"):
        llm_path = llm_index.get(utt_id)
        tg_path = mfa_index.get(utt_id)
        out_path = output_path_for_utt(args.output_dir, utt_id, args.flat_output)

        if llm_path is None or tg_path is None:
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

            mfa_words, audio_duration = load_word_alignment(tg_path)
            if not mfa_words:
                raise ValueError("Empty MFA words")

            if "offline" in seg:
                levels = ["offline"]
            else:
                levels = ["low_latency", "medium_latency", "high_latency"]

            out = {
                "utt_id": utt_id,
                "original_text": seg.get("input", ""),
            }

            for level in levels:
                eng = seg[level]["English"]
                zh = seg[level]["Chinese"]

                aligned = match_llm_chunks_to_mfa(eng, mfa_words)
                timeline = assign_chunks_by_window(aligned, chunk_seconds=chunk_seconds, audio_duration=audio_duration)
                src, tgt = build_final_segments(timeline, eng_chunks=eng, zh_chunks=zh)

                out[f"source_{level}"] = src
                out[f"target_{level}"] = tgt

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
                        "textgrid_path": tg_path,
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