#!/usr/bin/env python3
"""Install a comparison UI, publish the complete v3 arm, or add any extra arm (complete or partial) without replacing old data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
NEW_ID = "suffix_icl_v3"
OLD_ID = "source_only_boundary"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_ui(root, arms):
    for name in ("experiments.html", "experiments.js", "experiments.css"):
        shutil.copy2(HERE / name, root / name)
    for arm in arms:
        target = root / arm
        if not target.is_dir():
            continue
        # Only presentation assets are replaced, never review.json or raw evidence.
        for name in ("app.js", "styles.css"):
            shutil.copy2(HERE / "static" / name, target / name)
        html = (HERE / "static/index.html").read_text().replace(
            "</body>", '<script src="experiment-nav.js"></script>\n</body>')
        (target / "index.html").write_text(html)
        shutil.copy2(HERE / "experiment-nav.js", target / "experiment-nav.js")
    index = root / "index.html"
    if index.exists() and 'href="experiments.html"' not in index.read_text():
        html = index.read_text().replace("<body>", '<body>\n<nav style="padding:20px 36px"><a href="experiments.html">Experiments / 新旧 50-case 对比</a></nav>', 1)
        index.write_text(html)


def by_id(payload, ids):
    cases = payload["cases"]
    result = {item["utt_id"]: item for item in cases}
    if len(cases) != len(ids) or len(result) != len(ids) or set(result) != set(ids):
        raise ValueError("Expected exactly the frozen unique IDs, not a partial or duplicate bundle")
    return result


def validate_new(root, incoming, registry):
    ready = read(incoming / "READY.json")
    entry = next(e for e in registry["experiments"] if e["id"] == NEW_ID)
    if ready["run_name"] != entry["run_name"] or ready["input_sha256"] != registry["input_sha256"]:
        raise ValueError("Wrong run or frozen input checksum")
    if sha(incoming / "input_50.tsv") != registry["input_sha256"]:
        raise ValueError("Input TSV bytes changed")
    summary = read(incoming / "pilot_summary.json")
    if summary["summary"][NEW_ID]["valid_cases"] != 50:
        raise ValueError("Pilot report is not complete")
    old = by_id(read(root / OLD_ID / "data/review.json"), registry["case_ids"])
    new = by_id(read(incoming / NEW_ID / "data/review.json"), registry["case_ids"])
    for uid, b in new.items():
        a = old[uid]
        for key in ("source_full_text", "src_text_full", "reference_text"):
            if a[key] != b[key]:
                raise ValueError(f"{uid}: changed {key}")
        if [(s["step"], s["source_chunk"]) for s in a["steps"]] != [(s["step"], s["source_chunk"]) for s in b["steps"]]:
            raise ValueError(f"{uid}: changed or incomplete chunks")
        settings = dict(b["decoder_settings"])
        if settings.pop("targeted_prompt_version", None) != "future_set_v3_suffix_icl":
            raise ValueError(f"{uid}: not produced with v3")
        before = dict(a["decoder_settings"])
        before.pop("targeted_prompt_version", None)
        if settings != before:
            raise ValueError(f"{uid}: non-prompt decoder settings changed")
        if b["prediction"] != "".join(s["translation_delta"] for s in b["steps"]):
            raise ValueError(f"{uid}: incomplete prediction")
        raw = incoming / NEW_ID / "raw/per_utt" / f"{uid}.json"
        record = read(raw)
        if record["utt_id"] != uid or record["prediction"] != b["prediction"]:
            raise ValueError(f"{uid}: raw evidence does not match viewer")
        if (record["decoder_settings"] != b["decoder_settings"]
                or record["src_trajectory"] != [s["source_chunk"] for s in b["steps"]]
                or record["target_trajectory"] != [s["translation_delta"] for s in b["steps"]]
                or record["actions"] != [s["action"] for s in b["steps"]]):
            raise ValueError(f"{uid}: raw trajectory/settings do not match viewer")
        log = incoming / NEW_ID / "raw/verbose" / f"verbose_{uid}.log"
        detail = incoming / NEW_ID / "data/consensus" / f"{uid}.json"
        if not log.is_file() or not log.stat().st_size or not detail.is_file():
            raise ValueError(f"{uid}: missing verbose log or consensus detail")
        read(detail)
    return entry


def validate_arm_cases(root, bundle, registry, prompt_version):
    """Every case present in an extra arm must share input, reference and chunks with the old arm."""
    old = by_id(read(root / OLD_ID / "data/review.json"), registry["case_ids"])
    cases = {item["utt_id"]: item for item in read(bundle / "data/review.json")["cases"]}
    if not cases or len(cases) > len(registry["case_ids"]) or set(cases) - set(old):
        raise ValueError("Arm must cover a non-empty subset of the frozen 50 IDs")
    for uid, b in cases.items():
        a = old[uid]
        for key in ("source_full_text", "src_text_full", "reference_text"):
            if a[key] != b[key]:
                raise ValueError(f"{uid}: changed {key}")
        if [(s["step"], s["source_chunk"]) for s in a["steps"]] != [(s["step"], s["source_chunk"]) for s in b["steps"]]:
            raise ValueError(f"{uid}: changed or incomplete chunks")
        if b["decoder_settings"].get("targeted_prompt_version") != prompt_version:
            raise ValueError(f"{uid}: not produced with {prompt_version}")
        if b["prediction"] != "".join(s["translation_delta"] for s in b["steps"]):
            raise ValueError(f"{uid}: incomplete prediction")
        if not (bundle / "raw/verbose" / f"verbose_{uid}.log").is_file() or not (bundle / "data/consensus" / f"{uid}.json").is_file():
            raise ValueError(f"{uid}: missing verbose log or consensus detail")
    return len(cases)


def add_arm(root, args):
    path = root / "experiments.json"
    registry = read(path)
    if any(e["id"] == args.id for e in registry["experiments"]) or (root / args.id).exists():
        raise FileExistsError(f"Arm {args.id} already exists; do not overwrite a published experiment")
    count = validate_arm_cases(root, args.incoming, registry, args.prompt_version)
    status = "complete" if count == len(registry["case_ids"]) else "partial"
    temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=root))
    shutil.copytree(args.incoming, temporary / args.id)
    (temporary / args.id).rename(root / args.id)
    temporary.rmdir()
    registry["experiments"].append({
        "id": args.id, "label": args.label, "bundle_url": args.id + "/", "run_name": args.run_name,
        "prompt_version": args.prompt_version, "status": status, "cases": count,
        "description": args.description, "published_at": stamp(),
    })
    install_ui(root, [args.id])
    registry["updated_at"] = stamp()
    write(path, registry)
    return status, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "publish", "ui", "add-arm"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--incoming", type=Path)
    parser.add_argument("--id", help="add-arm: registry id and folder name (letters, digits, _ -)")
    parser.add_argument("--label", help="add-arm: label shown in the selectors")
    parser.add_argument("--description", default="", help="add-arm: one-line description")
    parser.add_argument("--run-name", default="", help="add-arm: decode run name")
    parser.add_argument("--prompt-version", default="future_set_v3_suffix_icl", help="add-arm: expected sampler prompt version")
    args = parser.parse_args()
    path = args.root / "experiments.json"
    if args.command == "add-arm":
        if not (args.id and args.label and args.incoming) or not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
            parser.error("add-arm needs --id, --label and --incoming")
        status, count = add_arm(args.root, args)
        print(json.dumps({"root": str(args.root), "id": args.id, "status": status, "cases": count, "command": "add-arm"}))
        return
    if args.command == "init":
        if path.exists():
            raise FileExistsError("Registry already exists; use ui or publish instead")
        manifest = read(args.manifest)
        selection = read(args.root / "cases.json")
        ids = [case["utt_id"] for case in selection["cases"]]
        if len(ids) != 50 or len(set(ids)) != 50 or sha(args.root / "input_50.tsv") != manifest["input_sha256"]:
            raise ValueError("Frozen selection mismatch")
        for arm in ("baseline", OLD_ID):
            by_id(read(args.root / arm / "data/review.json"), ids)
        registry = {"schema_version": 1, "updated_at": stamp(), "case_ids": ids,
            "input_sha256": manifest["input_sha256"], "default_left": OLD_ID, "default_right": NEW_ID,
            "experiments": [
                {"id": "baseline", "label": "历史 baseline / v2", "bundle_url": "baseline/",
                 "run_name": manifest["old_run"], "prompt_version": "future_set_v2_two_groups", "status": "complete",
                 "description": "最早的对照组；保留原来的 sampler 窗口及边界设置。"},
                {"id": OLD_ID, "label": "旧版 · Source-only boundary / v2", "bundle_url": OLD_ID + "/",
                 "run_name": manifest["old_run"], "prompt_version": "future_set_v2_two_groups", "status": "complete",
                 "description": "已完成的 50 cases。句子锚点、source-only、句末跳过 futures 并匹配标点。"},
                {"id": NEW_ID, "label": "新版 · Suffix ICL / v3", "bundle_url": NEW_ID + "/",
                 "run_name": manifest["run_name"], "prompt_version": "future_set_v3_suffix_icl", "status": "pending",
                 "jobs": manifest["jobs"],
                 "description": "同输入、同模型、同 seed、同句末策略。加入 5 个歧义示例和严格续写要求，允许不足 20 条，配套分组解析。"},
            ]}
        install_ui(args.root, [e["id"] for e in registry["experiments"]])
        write(path, registry)
    elif args.command == "publish":
        registry = read(path)
        entry = validate_new(args.root, args.incoming, registry)
        destination = args.root / NEW_ID
        if destination.exists():
            raise FileExistsError("New arm already exists; do not overwrite a published experiment")
        temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=args.root))
        shutil.copytree(args.incoming / NEW_ID, temporary / NEW_ID)
        for name in ("READY.json", "pilot_summary.json", "case_comparison.csv"):
            shutil.copy2(args.incoming / name, temporary / NEW_ID / name)
        (temporary / NEW_ID).rename(destination)
        temporary.rmdir()
        install_ui(args.root, [NEW_ID])
        entry["status"] = "complete"
        entry["published_at"] = stamp()
        registry["updated_at"] = stamp()
        write(path, registry)
    else:
        registry = read(path)
        install_ui(args.root, [e["id"] for e in registry["experiments"]])
    print(json.dumps({"root": str(args.root), "registry": str(path), "command": args.command}))


if __name__ == "__main__":
    main()
