# What is NOT on the Mac, and why

Machine-readable list: `exclude.txt` (used verbatim by `stage_to_home.sh`).
Remote-path metadata for the excluded data: `BABEL_PATHS.md`.

| Excluded | Size | Why | If you ever need it |
|---|---|---|---|
| `.git/` | 966 G (1.1 MB real) | 843 G orphaned `tmp_pack_*` + ~123 G loose objects | history via GitHub clone; junk is safe to `rm` on Babel (not done — your call) |
| `ckpts/` | 3.1 T | trained/pretrained checkpoints | stay on Babel; eval reads them in place |
| `models/` | 514 G | model weights | re-download from HF if ever needed elsewhere |
| `hf_cache/`, `.cache/`, `pip/uv/conda` caches, `vscode-server/` | ~170 G | caches, rebuildable | — |
| `conda_envs/`, `whisper_venv/` | 27 G | Linux binaries, not portable to macOS | text captures in `environment/` |
| `data_synthesis/outputs/` | 1.8 T | generated synthesis data (EAST/SALAMI/gigaspeech eval TSVs, many files >100 MB) | regenerate via pipeline or read on Babel |
| `data_synthesis/MFA_backup/` | 54 G | MFA alignment state backup | Babel only |
| `codes/gigaspeech/hibiki/output/` | 3.7 G | hibiki decode dumps | Babel only |
| `simul_tst_common/{models,mfa_tmp}/` | 1.8 G | whisper/MFA models + temp | re-download |
| `slurm_logs/` (all, incl. `scripts/infer/slurm_logs/` 1.2 G), `*.log` | ~1.4 G | job logs | read via ssh when debugging |
| `code/` (Megatron-LM, OmniSTEval) | 111 M | external clones | re-clone upstream; paths in BABEL_PATHS.md |
| `tools/` (SEGALE, metricx) | 124 M | tool checkouts | re-clone; SEGALE env pinned to sm_89 GPUs anyway |
| `.env`, `.codex`, `~/.keys/*` | — | secrets (OpenAI/DeepSeek keys, wandb/HF tokens) | recreate from `.env.example` |
| `ANLP/`, `IDL_final_3small*`, `scripts_2026-03-28.tar.gz`, `test.txt` | ~850 M | old course work / stale snapshots (Category D) | say so and they'll be added to the include set |

Kept on purpose despite "data-ish" names: top-level `datasets/` (985 K, eval-set metadata)
and `data_synthesis/datasets/` (54 M, one granary manifest TSV) — small, code references them.
Small result artifacts (plots, `ckpts*.txt` registries, `reports/*.html`, wiki scoreboard,
JSON result summaries inside `codes/`) are all INCLUDED for local research reasoning.
