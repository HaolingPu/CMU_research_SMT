#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DIR = Path("/data/user_data/haolingp/data_synthesis/codes/gigaspeech/hibiki/output/hibiki-100")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Hibiki trajectory JSON reconstruction consistency.")
    p.add_argument("path", nargs="?", default=str(DEFAULT_DIR))
    p.add_argument("--target-lang", choices=["zh", "de", "ja"], default="")
    return p.parse_args()


def maybe_add_space(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if left[-1].isspace() or right[0].isspace():
        return ""
    if left[-1].isalnum() and right[0].isalnum():
        return " "
    return ""


def maybe_add_space_target(left: str, right: str, target_lang: str) -> str:
    if not left or not right:
        return ""
    if target_lang in {"zh", "ja"}:
        return ""
    if left[-1].isspace() or right[0].isspace():
        return ""
    no_space_before = set(",.!?:;%)]}\"'»”’")
    no_space_after = set("([{\"'«“‘")
    if right[0] in no_space_before:
        return ""
    if left[-1] in no_space_after:
        return ""
    return " "


def join_source_pieces(pieces: List[Any]) -> str:
    running = ""
    for piece in pieces:
        piece = str(piece)
        running = running + maybe_add_space(running, piece) + piece
    return running


def join_target_pieces(pieces: List[Any], target_lang: str) -> str:
    running = ""
    for piece in pieces:
        piece = str(piece)
        running = running + maybe_add_space_target(running, piece, target_lang) + piece
    return running


def iter_records(path: Path) -> Iterable[tuple[Path, Dict[str, Any]]]:
    if path.is_dir():
        for json_path in sorted(path.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield json_path, item
            elif isinstance(data, dict):
                yield json_path, data
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield path, item
        elif isinstance(data, dict):
            yield path, data


def resolve_target_full(item: Dict[str, Any], target_lang: str = "") -> tuple[str, Optional[str]]:
    candidates: List[tuple[str, Any]] = []
    if target_lang:
        candidates.append((f"target_full_{target_lang}", item.get(f"target_full_{target_lang}")))
    candidates.extend(
        [
            ("target_full", item.get("target_full")),
            ("target_full_zh", item.get("target_full_zh")),
            ("target_full_de", item.get("target_full_de")),
            ("target_full_ja", item.get("target_full_ja")),
        ]
    )
    for key, value in candidates:
        if value is not None and str(value) != "":
            return str(value), key
    return "", None


def check_record(json_path: Path, item: Dict[str, Any], target_lang: str = "") -> List[str]:
    errors: List[str] = []
    uid = str(item.get("id", ""))
    src_full = str(item.get("source_text_full", ""))
    tgt_full, tgt_key = resolve_target_full(item, target_lang)
    src_traj = item.get("src_trajectory", [])
    tgt_traj = item.get("target_trajectory", [])

    if not isinstance(src_traj, list):
        errors.append(f"[BAD SRC TYPE] {uid} file={json_path} src_trajectory is not a list")
        return errors
    if not isinstance(tgt_traj, list):
        errors.append(f"[BAD TGT TYPE] {uid} file={json_path} target_trajectory is not a list")
        return errors

    if len(src_traj) != len(tgt_traj):
        errors.append(
            f"[LEN MISMATCH] {uid} file={json_path} len(src_trajectory)={len(src_traj)} "
            f"len(target_trajectory)={len(tgt_traj)}"
        )

    effective_target_lang = target_lang or str(item.get("target_lang", "") or "zh")
    src_joined = join_source_pieces(src_traj)
    tgt_joined = join_target_pieces(tgt_traj, effective_target_lang)

    if src_joined != src_full:
        errors.append(
            f"[SRC MISMATCH] {uid} file={json_path}\n"
            f"  full:   {repr(src_full)}\n"
            f"  joined: {repr(src_joined)}"
        )

    if tgt_joined != tgt_full:
        errors.append(
            f"[TGT MISMATCH] {uid} file={json_path}\n"
            f"  target_field: {tgt_key or 'NONE'}\n"
            f"  full:   {repr(tgt_full)}\n"
            f"  joined: {repr(tgt_joined)}"
        )

    return errors


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    total = 0
    failed = 0
    len_fail = 0
    src_fail = 0
    tgt_fail = 0
    missing_tgt_field = 0

    for json_path, item in iter_records(path):
        total += 1
        tgt_full, tgt_key = resolve_target_full(item, args.target_lang)
        if tgt_key is None:
            missing_tgt_field += 1
        errors = check_record(json_path, item, args.target_lang)
        if errors:
            failed += 1
            for err in errors:
                print(err)
                if err.startswith("[LEN MISMATCH]"):
                    len_fail += 1
                elif err.startswith("[SRC MISMATCH]"):
                    src_fail += 1
                elif err.startswith("[TGT MISMATCH]"):
                    tgt_fail += 1

    print("\n=== Summary ===")
    print(f"Path: {path}")
    print(f"Total records: {total}")
    print(f"Failed records: {failed}")
    print(f"Length mismatches: {len_fail}")
    print(f"Source reconstruction mismatches: {src_fail}")
    print(f"Target reconstruction mismatches: {tgt_fail}")
    print(f"Missing target_full field: {missing_tgt_field}")
    if failed == 0:
        print("All OK!")


if __name__ == "__main__":
    main()
