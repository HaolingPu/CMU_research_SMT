#!/usr/bin/env bash
#SBATCH --job-name=verify_decode_root
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
# Gate between decode and post-processing. Verifies that the decode root holds exactly the
# expected utterance set (the first TOTAL_ROWS ids of INPUT_TSV), each JSON loadable with a
# complete trajectory and the expected sampler prompt version. Utterances listed in
# ALLOW_MISSING_UTTS (comma-separated ids) may be absent; any other gap, duplicate, unexpected
# id, unreadable file or wrong prompt version fails the gate. Counts and the exact missing ids
# are appended to MANIFEST.
set -euo pipefail
: "${DECODE_ROOT:?}" "${TOTAL_ROWS:?}" "${PROMPT_VERSION:?}" "${MANIFEST:?}" "${INPUT_TSV:?}"
ALLOW_MISSING_UTTS="${ALLOW_MISSING_UTTS:-}"
python3 - "$DECODE_ROOT" "$TOTAL_ROWS" "$PROMPT_VERSION" "$MANIFEST" "$INPUT_TSV" "$ALLOW_MISSING_UTTS" <<'PY'
import csv, glob, json, os, sys
root, total, prompt, manifest, input_tsv, allow_raw = sys.argv[1:7]
total = int(total)
allowed = {u.strip() for u in allow_raw.split(",") if u.strip()}

csv.field_size_limit(10**9)
expected = []
with open(input_tsv, newline="", encoding="utf-8") as stream:
    for i, row in enumerate(csv.DictReader(stream, delimiter="\t")):
        if i >= total:
            break
        expected.append(row["id"])
expected_set = set(expected)

paths = sorted(glob.glob(os.path.join(root, "task_*", "per_utt", "*.json")))
seen, dupes, bad, wrong_prompt = {}, [], [], []
for p in paths:
    utt = os.path.splitext(os.path.basename(p))[0]
    try:
        d = json.load(open(p, encoding="utf-8"))
        n = len(d["src_trajectory"])
        assert d["utt_id"] == utt and len(d["target_trajectory"]) == n and len(d["actions"]) == n and n > 0
    except Exception as exc:
        bad.append(f"{p}: {exc}")
        continue
    if d.get("decoder_settings", {}).get("targeted_prompt_version") != prompt:
        wrong_prompt.append(p)
    if utt in seen:
        dupes.append(p)
    else:
        seen[utt] = p

missing = sorted(expected_set - set(seen))
unexpected = sorted(set(seen) - expected_set)
unapproved = sorted(set(missing) - allowed)
ok = (len(expected) == total and not dupes and not bad and not wrong_prompt
      and not unexpected and not unapproved)

lines = [f"decode_expected_utts={len(expected)}", f"decode_files={len(paths)}",
         f"decode_unique_utts={len(seen)}", f"decode_duplicates={len(dupes)}",
         f"decode_unreadable={len(bad)}", f"decode_wrong_prompt={len(wrong_prompt)}",
         f"decode_unexpected_utts={len(unexpected)}", f"decode_missing_utts={len(missing)}",
         f"decode_missing_list={','.join(missing) if missing else 'none'}",
         f"decode_allowed_missing={','.join(sorted(allowed)) if allowed else 'none'}",
         "decode_gate=" + ("PASSED" if ok else "FAILED")]
with open(manifest, "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
for label, items in (("duplicate", dupes), ("unreadable", bad), ("wrong_prompt", wrong_prompt),
                     ("unexpected", unexpected), ("missing_not_approved", unapproved)):
    for item in items[:20]:
        print(f"[GATE] {label}: {item}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
