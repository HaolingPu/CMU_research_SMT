# External code the jobs depend on (not vendored in full)

These live on Babel outside this repo; jobs reference them by absolute path (see
`data_synthesis/migration/BABEL_PATHS.md`). Pinned versions as of 2026-08-28:

| Path on Babel | Upstream | Commit | Local changes |
|---|---|---|---|
| `/data/user_data/haolingp/code/Megatron-LM` | https://github.com/NVIDIA/Megatron-LM.git | `377af02ad` | none |
| `/data/user_data/haolingp/code/OmniSTEval` | https://github.com/pe-trik/OmniSTEval.git | `5a6cc9b` | none |
| `/data/user_data/haolingp/tools/metricx` | https://github.com/google-research/metricx.git | `fc4978e` | none (same code also tracked at `data_synthesis/codes/metricx`) |
| `/data/user_data/haolingp/tools/SEGALE` | https://github.com/NVlabs/SEGALE.git | `bc19b2b` | **yes — see `SEGALE/`** |

## Re-creating `tools/SEGALE` from scratch
```bash
git clone https://github.com/NVlabs/SEGALE.git tools/SEGALE && cd tools/SEGALE && git checkout bc19b2b
git apply /home/haolingp/CMU_research_SMT/external/SEGALE/segale_align.patch      # or copy segale_align.py.modified over segale_align.py
cp /home/haolingp/CMU_research_SMT/external/SEGALE/run_gigaspeech_segale_qe.py .
# build in the `segale` conda env (torch cu124, sm_89 only: L40S / 6000Ada)
```
The other three are plain clones at the pinned commit.
