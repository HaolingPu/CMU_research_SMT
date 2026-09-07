#!/usr/bin/env python3
"""Archive an old review and stage an isolated, same-input v3 prompt pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[5]
    fs = repo / "data_synthesis/codes/gigaspeech/future_sampling"
    old = args.old_bundle
    selection = json.loads((old / "cases.json").read_text())
    assert selection["count"] == 50 and len(selection["cases"]) == 50
    assert len({c["utt_id"] for c in selection["cases"]}) == 50
    assert sha(old / "input_50.tsv") == selection["input_sha256"]
    expected_ids = [c["utt_id"] for c in selection["cases"]]
    for arm in ("baseline", "source_only_boundary"):
        data = json.loads((old / arm / "data/review.json").read_text())
        assert [c["utt_id"] for c in data["cases"]] == expected_ids
    if args.stage.exists() or args.archive.exists():
        raise FileExistsError("Staging/archive already exists; refusing to overwrite")
    hashes = {str(p.relative_to(old)): sha(p) for p in sorted(old.rglob("*")) if p.is_file()}
    shutil.copytree(old, args.archive)
    assert hashes == {str(p.relative_to(args.archive)): sha(p)
                      for p in sorted(args.archive.rglob("*")) if p.is_file()}
    write_json(args.archive.parent / (args.archive.name + ".sha256.json"), hashes)

    runtime = args.stage / "runtime"
    runtime.mkdir(parents=True)
    for relative in ("ambiguity_sampler_prompt.py", "sentence_boundary_helpers.py",
                     "consensus_decoding_token_id_level_instruct.py",
                     "experimental/pilot_case_guard.py", "experimental/run_single_case_ab.sbatch"):
        shutil.copy2(fs / relative, runtime / Path(relative).name)
    viewer = repo / "data_synthesis/tools/trajectory_viewer"
    shutil.copy2(viewer / "build_review_bundle.py", runtime / "build_review_bundle.py")
    shutil.copytree(viewer / "static", runtime / "static")
    shutil.copy2(Path(__file__).with_name("report_prompt_pilot.sbatch"), args.stage / "report.sbatch")
    remote = f"/home/haolingp/slurm_runs/{args.run_name}"
    output = f"/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_pilots/{args.run_name}"
    (args.stage / "selection").mkdir()
    shutil.copy2(old / "input_50.tsv", args.stage / "selection/input_50.tsv")
    selection["input_tsv"] = remote + "/selection/input_50.tsv"
    write_json(args.stage / "selection/cases.json", selection)
    extra = ("--targeted-sampler-context source-only --future-source-window-mode sentence-anchor "
             "--sentence-end-completion --sentence-end-boundary-mode conservative "
             "--sentence-end-punctuation match-source --targeted-prompt-version future_set_v3_suffix_icl")
    manifest = {
        "run_name": args.run_name, "run_dir": remote, "output_root": output,
        "selection_dir": remote + "/selection", "input_sha256": selection["input_sha256"],
        "variants": {"suffix_icl_v3": extra}, "sampler_seed": 1015,
        "case_count": 50, "groups": 4, "gpus_per_group": 2, "max_pilot_gpus": 8,
        "old_run": "source-only-sentence-boundary-50cases-20260907-023037",
        "comparison_arm": "source_only_boundary", "old_bundle_archive": str(args.archive),
        "old_bundle_hash_index": str(args.archive.parent / (args.archive.name + ".sha256.json")),
        "treatment": "v3 suffix-fit ICL prompt (five ambiguity examples) and variable-length grouped parser",
        "unchanged": "same 50 ordered inputs, seed, models, source-only sentence anchor, boundary closure, filters and strict consensus; no training",
        "local_base_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_sha256": {str(p.relative_to(runtime)): sha(p) for p in runtime.rglob("*") if p.is_file()},
        "models": {"gemma": "/data/user_data/haolingp/models/gemma-4-E2B-it",
                   "qwen_sampler": "/data/user_data/haolingp/models/Qwen3.8-27B-FP8",
                   "translator": "/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8"},
        "status": "staged_not_submitted", "jobs": {},
    }
    write_json(args.stage / "run_manifest.json", manifest)
    print(json.dumps({"stage": str(args.stage), "archive_files_verified": len(hashes),
                      "archive": str(args.archive), "run_dir": remote}, indent=2))


if __name__ == "__main__":
    main()
