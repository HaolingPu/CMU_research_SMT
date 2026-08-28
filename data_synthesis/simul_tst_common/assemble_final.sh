#!/bin/bash
# Final assembly: chunk outputs -> ordered translations -> tgt edits -> dataset dir.
set -e

BUILD=/data/user_data/haolingp/data_synthesis/simul_tst_common
PY=$BUILD/whisper_venv/bin/python
DEST=/data/user_data/haolingp/datasets/simul_tst_common
WAV_DIR=/data/group_data/li_lab/siqiouya/datasets/must-c/v2.0/en-zh/data/tst-COMMON/wav
cd "$BUILD"

echo "=== 1. restore source order from $(ls batch/chunks/chunk_*.output.jsonl | wc -l) chunk outputs"
$PY restore_order.py \
    --source Simul-tst-COMMON.en \
    --batch-output batch/chunks/chunk_*.output.jsonl \
    --out-prefix Simul-LLM
mv Simul-LLM.zh.sep.en Simul-LLM.sep.en 2>/dev/null || true

echo "=== 2. apply tgt edits (fidelity checkpoint)"
$PY repo/scripts/apply_edits.py \
    --input Simul-LLM.zh \
    --edits repo/data/tgt/zh/edits.json \
    --output Simul-tst-COMMON.zh 2>&1 | grep -vE "line/s|operation/s"

paste -d'\n' /dev/null /dev/null < /dev/null > /dev/null # noop
n_src=$(wc -l < Simul-tst-COMMON.en)
n_tgt=$(wc -l < Simul-tst-COMMON.zh)
echo "src lines: $n_src, tgt lines: $n_tgt"
if [ "$n_src" != "$n_tgt" ]; then
    echo "WARNING: line count mismatch"
fi

echo "=== 3. install dataset dir"
mkdir -p "$DEST"
cp Simul-tst-COMMON.en "$DEST/tst.en"
cp Simul-tst-COMMON.zh "$DEST/tst.zh"
cp Simul-LLM.zh "$DEST/tst.unpatched.zh"
ls "$WAV_DIR"/ted_*.wav | sort -t_ -k2 -n > "$DEST/tst.source"

# per-talk concatenated zh reference (one line per wav) for simuleval --target
$PY - << 'PYEOF'
import json
refs = [l.rstrip("\n") for l in open("Simul-tst-COMMON.zh", encoding="utf-8")]
by_talk = json.load(open("sentences_by_talk.json", encoding="utf-8"))
talks = sorted(by_talk, key=int)
with open("/data/user_data/haolingp/datasets/simul_tst_common/tst.target.zh", "w", encoding="utf-8") as f:
    for t in talks:
        f.write("".join(refs[idx] for idx, _ in by_talk[t]) + "\n")
print(f"tst.target.zh: {len(talks)} talk-level reference lines")
PYEOF
if [ -s tst.yaml ]; then
    cp tst.yaml "$DEST/tst.yaml"
else
    echo "(tst.yaml not built yet — run parse_mfa.py after MFA finishes)"
fi
echo "installed to $DEST"
