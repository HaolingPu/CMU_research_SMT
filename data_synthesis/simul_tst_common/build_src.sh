#!/bin/bash
# Concatenate whisper transcripts (numeric talk order) -> split -> apply src edits.
set -e

BUILD=/data/user_data/haolingp/data_synthesis/simul_tst_common
PY=$BUILD/whisper_venv/bin/python
cd "$BUILD"

: > raw_concat.txt
for f in $(ls asr_out/ted_*.txt | sort -t_ -k2 -n); do
    cat "$f" >> raw_concat.txt
    echo >> raw_concat.txt
done
echo "raw_concat: $(wc -l < raw_concat.txt) lines from $(ls asr_out/ted_*.txt | wc -l) talks"

$PY repo/scripts/split.py --input raw_concat.txt --output split.txt
echo "split: $(wc -l < split.txt) sentences"

$PY repo/scripts/apply_edits.py \
    --input split.txt \
    --edits repo/data/src/edits.json \
    --output Simul-tst-COMMON.en
echo "final src: $(wc -l < Simul-tst-COMMON.en) sentences"
