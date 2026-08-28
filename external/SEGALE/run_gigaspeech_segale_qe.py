#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICX_REPO = "/data/user_data/haolingp/tools/metricx"


def iter_consensus_jsons(job_dir: Path) -> Iterable[Path]:
    yield from sorted(job_dir.glob("task_*/*.json"))


def pick_job_dir(root: Path) -> Path:
    if (root / "task_0").is_dir() or list(root.glob("task_*/*.json")):
        return root

    candidates: List[Tuple[int, float, Path]] = []
    for path in sorted(root.glob("job_*")):
        if not path.is_dir():
            continue
        count = sum(1 for _ in iter_consensus_jsons(path))
        if count:
            candidates.append((count, path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No consensus JSON files found under {root}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip bad json {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "error" in data:
        return None
    return data


def build_segale_inputs(job_dir: Path, work_dir: Path, sys_id: str) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    system_file = work_dir / "system.jsonl"
    ref_file = work_dir / "ref.jsonl"

    seen = set()
    docs = 0
    rows = 0
    fallback_src = 0
    missing_pred = 0
    missing_ref = 0
    policies = Counter()

    with system_file.open("w", encoding="utf-8") as fsys, ref_file.open(
        "w", encoding="utf-8"
    ) as fref:
        for path in iter_consensus_jsons(job_dir):
            data = load_json(path)
            if not data:
                continue

            utt_id = str(data.get("utt_id", "")).strip()
            if not utt_id or utt_id in seen:
                continue
            seen.add(utt_id)

            decoder = data.get("decoder_impl") or {}
            policies[str(decoder.get("candidate_policy", "unknown"))] += 1

            src_segments = data.get("src_text_full")
            if not isinstance(src_segments, list) or not any(str(x).strip() for x in src_segments):
                src_segments = [data.get("source_full_text", "")]
                fallback_src += 1
            src_segments = [str(x).strip() for x in src_segments if str(x).strip()]
            if not src_segments:
                continue

            prediction = str(data.get("prediction", "")).strip()
            reference = str(data.get("reference_text", "")).strip()
            if not prediction:
                missing_pred += 1
            if not reference:
                missing_ref += 1

            for seg_id, src in enumerate(src_segments):
                sys_row = {
                    "doc_id": utt_id,
                    "sys_id": sys_id,
                    "seg_id": seg_id,
                    "src": src,
                    "tgt": prediction if seg_id == 0 else "",
                }
                ref_row = {
                    "doc_id": utt_id,
                    "seg_id": seg_id,
                    "src": src,
                    "tgt": reference if seg_id == 0 else "",
                }
                fsys.write(json.dumps(sys_row, ensure_ascii=False) + "\n")
                fref.write(json.dumps(ref_row, ensure_ascii=False) + "\n")
                rows += 1
            docs += 1

    return {
        "system_file": str(system_file),
        "ref_file": str(ref_file),
        "docs": docs,
        "rows": rows,
        "fallback_src": fallback_src,
        "missing_pred": missing_pred,
        "missing_ref": missing_ref,
        "policies": dict(policies),
    }


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def segale_align(
    system_file: Path,
    ref_file: Path,
    segmenter: str,
    task_lang: str,
    proc_device: str,
    embedding_model: str,
    max_size: int,
    verbose: int,
) -> Path:
    cmd = [
        "segale-align",
        "--system_file",
        str(system_file),
        "--ref_file",
        str(ref_file),
        "--segmenter",
        segmenter,
        "--task_lang",
        task_lang,
        "--proc_device",
        proc_device,
        "--embedding_model",
        embedding_model,
        "--max_size",
        str(max_size),
    ]
    if verbose > 0:
        cmd.append("-" + ("v" * verbose))
    run(cmd)

    aligned = system_file.with_suffix("") / f"aligned_{segmenter}_{system_file.stem}.jsonl"
    if not aligned.is_file():
        raise FileNotFoundError(f"Expected aligned output not found: {aligned}")
    return aligned


def metricx_env(metricx_repo: str) -> Dict[str, str]:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = metricx_repo + (os.pathsep + old if old else "")
    return env


def run_metricx_qe(
    aligned_file: Path,
    output_dir: Path,
    metricx_repo: str,
    tokenizer: str,
    model_name_or_path: str,
    max_input_length: int,
    batch_size: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / f"eval_metricx_qe_{aligned_file.name}"
    result_file = output_dir / f"result_metricx_qe_{aligned_file.name}"
    raw_metricx_output = output_dir / f"metricx_qe_predictions_{aligned_file.stem}.jsonl"

    aligned_rows = [json.loads(line) for line in aligned_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    zero_windows: List[int] = []
    none_windows: List[int] = []
    scored_indices: List[int] = []

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as fin:
        metricx_input = Path(fin.name)
        for idx, item in enumerate(aligned_rows):
            src = str(item.get("src", "")).strip()
            tgt = str(item.get("tgt", "")).strip()
            if src and tgt:
                fin.write(json.dumps({"source": src, "hypothesis": tgt, "reference": ""}, ensure_ascii=False) + "\n")
                scored_indices.append(idx)
            elif src or tgt:
                zero_windows.append(idx)
            else:
                none_windows.append(idx)

    try:
        run(
            [
                sys.executable,
                "-m",
                "metricx24.predict",
                "--tokenizer",
                tokenizer,
                "--model_name_or_path",
                model_name_or_path,
                "--max_input_length",
                str(max_input_length),
                "--batch_size",
                str(batch_size),
                "--input_file",
                str(metricx_input),
                "--output_file",
                str(raw_metricx_output),
                "--qe",
            ],
            env=metricx_env(metricx_repo),
        )
    finally:
        metricx_input.unlink(missing_ok=True)

    predictions: List[float] = []
    with raw_metricx_output.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            predictions.append(float(json.loads(line).get("prediction", 0)))
    if len(predictions) != len(scored_indices):
        raise RuntimeError(
            f"MetricX prediction count mismatch: got {len(predictions)}, expected {len(scored_indices)}"
        )

    scores = [-1.0] * len(aligned_rows)
    for idx, score in zip(scored_indices, predictions):
        scores[idx] = score
    for idx in zero_windows:
        scores[idx] = 25.0
    for idx in none_windows:
        scores[idx] = -1.0

    for item, score in zip(aligned_rows, scores):
        item["metricx-qe"] = score

    with eval_file.open("w", encoding="utf-8") as fout:
        for item in aligned_rows:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in aligned_rows:
        grouped[str(item.get("doc_id", ""))].append(item)

    doc_rows: List[Dict[str, Any]] = []
    for doc_id, items in grouped.items():
        items.sort(key=lambda x: x.get("seg_id", 0))
        valid = [float(item["metricx-qe"]) for item in items if float(item["metricx-qe"]) >= 0]
        avg = sum(valid) / len(valid) if valid else 0.0
        doc_rows.append(
            {
                "doc_id": doc_id,
                "sys_id": items[0].get("sys_id", ""),
                "metricx-qe": avg,
                "total_seg": len(valid),
                "misaligned_seg": sum(1 for v in valid if v == 25.0),
                "src": "\n".join(str(item.get("src", "")) for item in items),
                "ref": "\n".join(str(item.get("ref", "")) for item in items),
                "tgt": "\n".join(str(item.get("tgt", "")) for item in items),
            }
        )
    doc_rows.sort(key=lambda x: x["doc_id"])
    with result_file.open("w", encoding="utf-8") as fout:
        for item in doc_rows:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    values = [float(item["metricx-qe"]) for item in doc_rows]
    summary = {
        "aligned_windows": len(aligned_rows),
        "scored_windows": len(scored_indices),
        "one_sided_windows": len(zero_windows),
        "empty_windows": len(none_windows),
        "docs": len(doc_rows),
        "metricx_qe_avg": statistics.mean(values) if values else None,
        "metricx_qe_median": statistics.median(values) if values else None,
        "metricx_qe_min": min(values) if values else None,
        "metricx_qe_max": max(values) if values else None,
        "qe_leq_3_count": sum(v <= 3.0 for v in values),
        "qe_leq_3_pct": (sum(v <= 3.0 for v in values) / len(values) * 100.0) if values else None,
    }
    return eval_file, result_file, summary


def write_summary(summary: Dict[str, Any], path_prefix: Path) -> Tuple[Path, Path]:
    json_path = path_prefix.with_suffix(".json")
    tsv_path = path_prefix.with_suffix(".tsv")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8") as fout:
        fout.write("docs\taligned_windows\tmetricx_qe_avg\tqe_leq_3_pct\tqe_leq_3_count\n")
        fout.write(
            f"{summary.get('docs')}\t{summary.get('aligned_windows')}\t"
            f"{summary.get('metricx_qe_avg'):.6f}\t{summary.get('qe_leq_3_pct'):.4f}\t"
            f"{summary.get('qe_leq_3_count')}\n"
        )
    return json_path, tsv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SEGALE inputs from GigaSpeech consensus JSONs, align, run MetricX-QE, and summarize."
    )
    parser.add_argument("--consensus-root", required=True, help="Experiment root or a job_* directory.")
    parser.add_argument("--job-dir", default="", help="Optional explicit job_* directory.")
    parser.add_argument("--work-dir", default="", help="Working directory for generated SEGALE files.")
    parser.add_argument("--sys-id", default="", help="sys_id to write into system.jsonl.")
    parser.add_argument("--metricx-repo", default=DEFAULT_METRICX_REPO)
    parser.add_argument("--segmenter", default="spacy", choices=["spacy", "ersatz"])
    parser.add_argument("--task-lang", default="zh")
    parser.add_argument("--proc-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--align-verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--tokenizer", default="google/mt5-large")
    parser.add_argument("--metricx-model", default="google/metricx-24-hybrid-large-v2p6")
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clean", action="store_true", help="Remove previous work-dir before running.")
    parser.add_argument("--remove-intermediate", action="store_true", help="Remove work-dir after summary is written.")
    args = parser.parse_args()

    consensus_root = Path(args.consensus_root).resolve()
    job_dir = Path(args.job_dir).resolve() if args.job_dir else pick_job_dir(consensus_root)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else consensus_root / "segale_metricx_qe"
    sys_id = args.sys_id or f"{consensus_root.name}_{job_dir.name}"

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"[info] consensus_root={consensus_root}")
    print(f"[info] job_dir={job_dir}")
    print(f"[info] work_dir={work_dir}")

    input_info = build_segale_inputs(job_dir, work_dir, sys_id)
    print("[info] built inputs: " + json.dumps(input_info, ensure_ascii=False))

    aligned_file = segale_align(
        Path(input_info["system_file"]),
        Path(input_info["ref_file"]),
        args.segmenter,
        args.task_lang,
        args.proc_device,
        args.embedding_model,
        args.max_size,
        args.align_verbose,
    )
    print(f"[info] aligned_file={aligned_file}")

    eval_file, result_file, qe_summary = run_metricx_qe(
        aligned_file,
        aligned_file.parent,
        args.metricx_repo,
        args.tokenizer,
        args.metricx_model,
        args.max_input_length,
        args.batch_size,
    )
    print(f"[info] eval_file={eval_file}")
    print(f"[info] result_file={result_file}")

    summary = {
        "consensus_root": str(consensus_root),
        "job_dir": str(job_dir),
        "work_dir": str(work_dir),
        "system_file": input_info["system_file"],
        "ref_file": input_info["ref_file"],
        "aligned_file": str(aligned_file),
        "eval_file": str(eval_file),
        "result_file": str(result_file),
        "input": input_info,
        "metricx_qe": qe_summary,
    }
    prefix = consensus_root / "segale_metricx_qe_summary"
    json_path, tsv_path = write_summary(
        {
            **qe_summary,
            "consensus_root": str(consensus_root),
            "job_dir": str(job_dir),
            "result_file": str(result_file),
        },
        prefix,
    )
    (consensus_root / "segale_metricx_qe_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== MetricX-QE Summary =====")
    print(f"Docs          : {qe_summary['docs']}")
    print(f"Aligned win   : {qe_summary['aligned_windows']}")
    print(f"QE avg        : {qe_summary['metricx_qe_avg']:.6f}")
    print(f"QE <= 3.0     : {qe_summary['qe_leq_3_count']}/{qe_summary['docs']} ({qe_summary['qe_leq_3_pct']:.4f}%)")
    print(f"Summary JSON  : {json_path}")
    print(f"Summary TSV   : {tsv_path}")

    if args.remove_intermediate:
        shutil.rmtree(work_dir)
        print(f"[info] removed intermediate work_dir={work_dir}")


if __name__ == "__main__":
    main()
