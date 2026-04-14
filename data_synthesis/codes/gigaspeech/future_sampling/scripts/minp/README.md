# Min-P Consensus Decoding Ablation

Generation scripts:

```bash
sbatch scripts/minp/run_minp0p001_40k_preempt.sbatch
sbatch scripts/minp/run_minp0p005_40k_preempt.sbatch
sbatch scripts/minp/run_minp0p01_40k_preempt.sbatch
sbatch scripts/minp/run_minp0p05_40k_preempt.sbatch
sbatch scripts/minp/run_minp0p1_40k_preempt.sbatch
sbatch scripts/minp/run_minp0p3_40k_preempt.sbatch
```

Or submit all:

```bash
bash scripts/minp/submit_all_minp_ablation.sh
```

Output directories:

```text
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.001
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.005
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.01
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.05
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.1
/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_debug/minp/consensus_decoding_en_zh_minp_0.3
```

After generation finishes, run QE filtering:

```bash
bash scripts/minp/submit_qe_after_generation.sh 0.05 <generation_job_id>
```
