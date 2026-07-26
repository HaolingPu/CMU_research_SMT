# Raw Sources (read-only)

This is the **source of truth** layer. The LLM reads these but NEVER edits them.

- `papers/` — papers as PDF or extracted text/notes.
- `experiments/` — dropped experiment snapshots (a config + its results), one file or folder per run.

## The codebase is also a raw source

`../../codes/` (i.e. `data_synthesis/codes/`) is a third raw source, but it is **referenced by
path, not copied here**. Wiki pages cite code as `../codes/<path>:<line>` in their `sources:`
frontmatter, keeping a single source of truth.

## Trained models & performance are raw sources too

`ckpts/infinisst-omni/` (repo root, i.e. `../../../ckpts/`) holds the trained checkpoints and
their `evaluation/.../scores.tsv` results — also **referenced by path, not copied** (2.2 TB of
weights). The measured performance is summarized in `comparisons/scoreboard.md`.

## To ingest a source

Drop a file into `papers/` or `experiments/`, then run the `wiki` skill's **ingest** flow.
