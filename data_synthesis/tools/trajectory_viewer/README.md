# Future Consensus Trajectory Viewer

This tool packages completed per-utterance consensus outputs into a mentor-facing,
read-only review site. It shows the source ASR chunks, committed translation deltas,
READ/WRITE actions, selected Gemma/Qwen futures, final prediction, reference, metrics,
and the extracted GigaSpeech audio segment.

## Packaged 100-case review set

The shared bundle is stored at:

```text
/data/group_data/li_lab/haolingp/data_synthesis/trajectory_reviews/ambiguity-q38-gemma-q36-first100
```

It contains:

- `raw/per_utt/`: generated trajectory JSONs.
- `raw/verbose/`: complete future-generation and filtering logs.
- `audio/`: extracted source clips.
- `data/review.json`: browser-ready data for all 100 cases.
- `index.html`, `styles.css`, and `app.js`: the review website.

`/data/group_data` is mounted on compute nodes, not `login2`. From the login node,
use `sbatch` as shown below or enter an existing allocation with
`srun --jobid=<job_id> --overlap --pty bash` before listing the bundle.

## Build a 100-case bundle on a BABEL compute node

```bash
python3 data_synthesis/tools/trajectory_viewer/build_review_bundle.py \
  --input-tsv /data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv \
  --decode-root /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831 \
  --output-dir /data/group_data/li_lab/haolingp/data_synthesis/trajectory_reviews/ambiguity-q38-gemma-q36-first100 \
  --limit 100 \
  --run-name ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831 \
  --overwrite
```

The selection is deterministic: the first 100 TSV rows that have both a complete JSON
and a matching verbose log. Raw JSONs and logs are retained under `raw/`.

## Launch on BABEL

Submit the CPU-only server:

```bash
mkdir -p ~/slurm_logs
sbatch --export=ALL,BUNDLE_DIR=/data/group_data/li_lab/haolingp/data_synthesis/trajectory_reviews/ambiguity-q38-gemma-q36-first100,PORT=8765 \
  data_synthesis/tools/trajectory_viewer/launch_server.sbatch
```

Read the job output to get the assigned compute node. If it reports `babel-x9-32`,
open a tunnel from the laptop:

```bash
ssh -N -L 8765:babel-x9-32:8765 babel
```

Then visit `http://127.0.0.1:8765`.

## Launch locally

If the bundle is copied to the laptop:

```bash
python3 data_synthesis/tools/trajectory_viewer/server.py /path/to/bundle --port 8765
```
