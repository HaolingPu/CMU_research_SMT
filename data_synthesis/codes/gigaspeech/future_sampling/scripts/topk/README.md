# Top-k Consensus Runs

Each file here is a self-contained `sbatch` script for a 40k English -> Chinese
consensus decoding run on the `preempt` partition.

Submit one run:

```bash
sbatch /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/run_topk1_40k_preempt.sbatch
```

Submit top-k=5 with 8 GPUs total on `general`:

```bash
sbatch /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/run_topk5_40k_8gpu_general.sbatch
```

Submit the comparison sweep:

```bash
for k in 1 5 10 20 50; do
  sbatch /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/run_topk${k}_40k_preempt.sbatch
done
```

Outputs go to:

```text
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_<k>
```

These jobs set `SKIP_EXISTING=1`, so re-submitting the same script will skip
already-written per-case JSON files under the same output root.

Run QE<=3 filter after generation. This submits the local top-k QE pipeline in
this folder: prepare, 8-GPU MetricX predict, then finalize/filter.

```bash
bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/submit_qe_after_generation.sh <top_k> <generation_job_id>
```

Example:

```bash
bash /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/scripts/topk/submit_qe_after_generation.sh 1 7040000
```

QE outputs go to:

```text
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_<k>/job_<generation_job_id>-metricx
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/topk/consensus_decoding_en_zh_top_<k>/job_<generation_job_id>-qe3
```
