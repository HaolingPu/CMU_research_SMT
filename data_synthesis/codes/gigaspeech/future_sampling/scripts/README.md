# Future Sampling Scripts

New submit/wrapper scripts should live under this directory instead of the
`future_sampling/` root.

Current layout:

- `topk/`: top-k consensus decoding pipelines.

The older root-level sbatch files are intentionally left in place for now
because some running or pending Slurm jobs still reference them by absolute
path.
