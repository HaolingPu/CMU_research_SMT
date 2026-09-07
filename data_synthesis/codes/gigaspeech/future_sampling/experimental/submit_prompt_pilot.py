#!/usr/bin/env python3
"""Submit the staged eight-GPU pilot without touching existing jobs or outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def call(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["run_dir"])
    if manifest["jobs"]:
        raise RuntimeError("Already submitted; inspect recorded jobs instead of submitting twice")
    queue = call(["squeue", "-h", "-r", "-u", "haolingp", "-o", "%i|%T|%b|%j"])
    for row in queue.splitlines():
        job, state, gres, name = row.split("|", 3)
        if name.startswith("icl50_"):
            raise RuntimeError("Existing pilot job found; recover its ID before proceeding")
        if "gpu:" in gres and job.split("_")[0] not in {"10329388", "10329390", "10329392"}:
            raise RuntimeError(f"Unreviewed GPU job {job}: re-audit the cap")
    generation = call(["scontrol", "show", "job", "10329388_23"])
    if "ArrayTaskThrottle=16" not in generation or "gres/gpu=1," not in generation:
        raise RuntimeError("Generation throttle/resources changed; re-audit the GPU cap")
    for relative, expected in manifest["runtime_sha256"].items():
        assert hashlib.sha256((root / "runtime" / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root / "selection/input_50.tsv").read_bytes()).hexdigest() == manifest["input_sha256"]
    (root / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"INPUT_TSV": str(root / "selection/input_50.tsv"), "OUTPUT_ROOT": manifest["output_root"],
                "DECODER": str(root / "runtime/consensus_decoding_token_id_level_instruct.py"),
                "CASE_GUARD": str(root / "runtime/pilot_case_guard.py"),
                "PILOT_CASE_COUNT": "50", "PILOT_GROUPS": "4", "SAMPLER_SEED": "1015",
                "VARIANTS": ";".join(f"{name}={extra}" for name, extra in manifest["variants"].items()),
                "RUN_DIR": str(root)})
    manifest["gpu_audit"] = {"queue": queue, "generation": generation,
        "budget": "Simul generation <=16 or later stages <=16; pilot 4x2=8; combined <=24"}
    manifest["exclude"] = args.exclude
    manifest["remote_base_commit"] = call(["git", "-C", "/home/haolingp/CMU_research_SMT", "rev-parse", "HEAD"])

    def record():
        temporary = args.manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.manifest)

    common = ["sbatch", "--parsable", "--partition=preempt", "--requeue", "--export=ALL"]
    generation_id = subprocess.check_output(common + ["--qos=preempt_qos", "--array=0-3%4",
        "--time=08:00:00", "--time-min=01:00:00", "--job-name=icl50_v3",
        "--exclude=" + args.exclude, "--output=" + str(root / "logs/decode_%A_%a.out"),
        "--error=" + str(root / "logs/decode_%A_%a.err"), str(root / "runtime/run_single_case_ab.sbatch")],
        env=env, text=True).strip().split(";")[0]
    manifest["jobs"]["decode"] = generation_id
    manifest["status"] = "submitted"
    record()
    report_id = subprocess.check_output(common + ["--qos=preempt_cpu_qos", "--job-name=icl50_report",
        "--dependency=afterok:" + generation_id, "--output=" + str(root / "logs/report_%j.out"),
        "--error=" + str(root / "logs/report_%j.err"), str(root / "report.sbatch")],
        env=env, text=True).strip().split(";")[0]
    manifest["jobs"]["report"] = report_id
    record()
    print(json.dumps(manifest["jobs"]))


if __name__ == "__main__":
    main()
