#!/usr/bin/env bash
#SBATCH --job-name=verify_decode_root
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
# Gate between decode and post-processing: exactly TOTAL_ROWS distinct utterance JSONs under
# DECODE_ROOT/task_*/per_utt, each loadable with a complete trajectory and the expected prompt
# version. Appends the counts to MANIFEST and exits non-zero on any mismatch.
set -euo pipefail
: "${DECODE_ROOT:?}" "${TOTAL_ROWS:?}" "${PROMPT_VERSION:?}" "${MANIFEST:?}"
python3 - "$DECODE_ROOT" "$TOTAL_ROWS" "$PROMPT_VERSION" "$MANIFEST" <<'PY'
import glob, json, os, sys
root, total, prompt, manifest = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
paths = sorted(glob.glob(os.path.join(root, "task_*", "per_utt", "*.json")))
seen, dupes, bad, wrong_prompt = {}, [], [], []
for p in paths:
    utt = os.path.splitext(os.path.basename(p))[0]
    try:
        d = json.load(open(p, encoding="utf-8"))
        n = len(d["src_trajectory"])
        assert d["utt_id"] == utt and len(d["target_trajectory"]) == n and len(d["actions"]) == n and n > 0
    except Exception as exc:
        bad.append(f"{p}: {exc}"); continue
    if d.get("decoder_settings", {}).get("targeted_prompt_version") != prompt:
        wrong_prompt.append(p)
    if utt in seen:
        dupes.append(p)
    else:
        seen[utt] = p
lines = [f"decode_files={len(paths)}", f"decode_unique_utts={len(seen)}", f"decode_duplicates={len(dupes)}",
         f"decode_unreadable={len(bad)}", f"decode_wrong_prompt={len(wrong_prompt)}"]
ok = len(seen) == total and not dupes and not bad and not wrong_prompt
lines.append("decode_gate=" + ("PASSED" if ok else "FAILED"))
with open(manifest, "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
for label, items in (("duplicate", dupes), ("unreadable", bad), ("wrong_prompt", wrong_prompt)):
    for item in items[:20]:
        print(f"[GATE] {label}: {item}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
