# Migration Plan — Babel → Mac (local Claude/Codex controller)

Prepared 2026-08-24 on compute node `babel-p5-24`. Goal: a local Mac copy that can
understand/edit the project, reproduce experiment configs, and drive Babel via SLURM —
without pulling datasets, checkpoints, or model weights.

## Phase 1 — Project identification

| Item | Value |
|---|---|
| Hostname | `babel-p5-24` (Babel compute node; `/data/user_data` mounted) |
| Working dir | `/data/user_data/haolingp/data_synthesis` |
| **Git root** | `/data/user_data/haolingp` — the git repo is the WHOLE user_data dir, not just data_synthesis |
| Repo | yes; remote `origin` = `git@github.com:HaolingPu/CMU_research_SMT.git` |
| Branch | `feature/llm-wiki` @ `30099bb`, **in sync with origin** (no unpushed commits) |
| Dirty state | 4 modified files + ~195 untracked paths (see `migration/GIT_STATE.md`) |
| Tracked content | 461 files, **~20 MB total** (401 under `data_synthesis/`, 55 under `scripts/`, plus `docs/`, `README.md`, `install.sh`) |
| Submodule | `data_synthesis/codes/metricx` is a gitlink with **no `.gitmodules`** — a fresh clone will NOT fetch it; rsync brings it |

### Critical finding: `.git` is 966 GB of junk

`du -sh .git` = **966 GB**, but the real pack is **1.1 MB**. 843 GB is 10 orphaned
`.git/objects/pack/tmp_pack_*` files (known quota-bloat issue) plus ~123 GB of loose
objects from aborted adds of large files. **Never rsync `.git` to the Mac.**
Strategy instead: `git clone` from GitHub (full history, a few MB), then rsync the
uncommitted/untracked working files on top. See `migration/PULL_TO_MAC.md`.
(The tmp_pack files are safe to delete on Babel, but nothing is deleted as part of this plan.)

### Top-level disk usage (git root `/data/user_data/haolingp`)

| Dir | Size | Note |
|---|---|---|
| ckpts/ | 3.1 T | trained checkpoints (Cat. B) |
| .git/ | 966 G | 1.1 MB real + junk (see above) |
| models/ | 514 G | base/QE model weights (Cat. B) |
| hf_cache/ | 127 G | HF cache (Cat. C) |
| conda_envs/ | 27 G | binary envs (Cat. C — captured as text in migration/environment/) |
| uv_cache/, pip_cache/, conda_pkgs/, conda_user/, vscode-server/ | ~18 G | caches (Cat. C) |
| data_synthesis/ | ~1.9 T | the project — outputs/ 1.8 T + MFA_backup 54 G + hibiki/output 3.7 G (Cat. B/C); code itself ~200 MB |
| scripts/ | 1.3 G | 1.2 G is `infer/slurm_logs/` (Cat. C); scripts themselves ~100 MB |
| IDL_final_3small/ (+tar.gz) | 513 M | old course project (Cat. D) |
| ANLP/ | 320 M | old course dir, gitignored (Cat. D — default: skip) |
| tools/ | 124 M | SEGALE + metricx tool checkouts (Cat. D) |
| code/ | 111 M | Megatron-LM + OmniSTEval external clones (Cat. B — record paths, re-clone if needed) |
| EMNLP_2026_Future_Aware_Data_Synthesis/ | 42 M | paper latex/figures/survey (Cat. A) |
| datasets/ (top-level) | 985 K | small (acl_6060 etc.) — Cat. A |
| docs/, slurm_logs/, .claude/ | small | docs+.claude Cat. A; slurm_logs Cat. C |

## Phase 2 — Classification

### Category A — MUST copy to Mac (~450 MB total)

- `data_synthesis/codes/` — all synthesis/eval source (EAST, SALAMI, future_sampling,
  rule-based-SMT, yodas, metricx submodule) **excluding** `hibiki/output/` (3.7 G)
- `data_synthesis/codes-refactored/` — consensus decoding, wait-k, LA (803 K)
- `data_synthesis/wiki/` — the LLM Wiki: the project's knowledge base (1.5 M) — highest-value item
- `data_synthesis/reports/`, `data_synthesis/refactor_view/`, `data_synthesis/.claude/`
- `data_synthesis/datasets/` — 54 M, only a manifest TSV (small metadata, NOT the ignored big datasets)
- `scripts/` — train/infer/synth/debug/babysitter sbatch + driver scripts, plots, ckpts*.txt
  registries **excluding** `scripts/infer/slurm_logs/` (1.2 G)
- `simul_tst_common/` — eval-set build scripts + small text/json artifacts, **excluding**
  `models/` (1.5 G), `mfa_tmp/` (285 M), `whisper_venv/` (101 M)
- `EMNLP_2026_Future_Aware_Data_Synthesis/` — paper draft, figures, literature survey (42 M)
- `datasets/` (top-level, 985 K), `docs/`, `README.md`, `install.sh`, `.gitignore`,
  `.env.example` (NOT `.env`), `vllm_requirements.txt`
- `.claude/` (repo root) — Claude Code project config
- `data_synthesis/migration/` — everything produced by this plan
- Git history — via clone from GitHub (not via rsync)

### Category B — record metadata only (see `migration/BABEL_PATHS.md`)

- `ckpts/` 3.1 T — trained checkpoints (`infinisst-omni/<exp>/v*-hf/` incl. per-ckpt `evaluation/` results) + `pretrained/`
- `models/` 514 G — Qwen3-Omni-30B, Qwen3-30B FP8 variants, Qwen3.5-122B, metricx-23/24, mt5-xl, LaBSE, …
- `hf_cache/` 127 G
- `data_synthesis/outputs/` — synthesized training data (EAST/Refined_EAST/SALAMI/gigaspeech, MFA textgrids)
- `/data/group_data/li_lab/siqiouya/datasets/gigaspeech/` — source manifests + audio (group share)
- `code/Megatron-LM`, `code/OmniSTEval` — external repos; re-clone on demand, no need to mirror
- `data_synthesis/MFA_backup/` 54 G — MFA alignment backup

### Category C — do NOT copy

`.git/` junk objects; all caches (`hf_cache/`, `.cache/`, `conda_pkgs/`, `pip_cache/`,
`uv_cache/`, `conda_user/`, `vscode-server/`); binary conda envs (`conda_envs/`,
`whisper_venv/`); `__pycache__/`, `*.pyc`; `slurm_logs/` everywhere (incl.
`scripts/infer/slurm_logs/` 1.2 G); `data_synthesis/codes/gigaspeech/hibiki/output/` 3.7 G;
`simul_tst_common/mfa_tmp/`; `.env` (contains API keys — names documented, values excluded);
`/home/haolingp/.keys/` (wandb + HF tokens, referenced by train scripts — never copy).

### Category D — review manually (excluded by default; say the word and they go in include)

| Item | Size | What it is |
|---|---|---|
| `ANLP/` | 320 M | old course dir, gitignored |
| `IDL_final_3small/` + `IDL_final_3small_deploy.tar.gz` | 513 M | old IDL course project, untracked |
| `tools/SEGALE`, `tools/metricx` | 124 M | tool checkouts used by QE pipeline — likely re-cloneable; paths recorded |
| `simul_tst_common/repo/`, `simul_tst_common/batch/` | ~7 M | small — currently INCLUDED; drop if noise |
| `scripts_2026-03-28.tar.gz` | 243 K | old scripts snapshot — currently EXCLUDED |
| `data_synthesis/MFA_backup/` | 54 G | backup of MFA state — assumed regenerable, excluded |
| `.claude/` at repo root vs `~/.claude` memory | — | project config included; global memory/skills live in `/home/haolingp/.claude` (outside this repo — copy separately if you want session memory on the Mac) |

## Phase 3 — Large files / directories

| Path | Size | What it is | Copy? |
|---|---|---|---|
| `.git/objects/pack/tmp_pack_*` (10 files) | 843 G | orphaned git pack temp files | NO |
| `.git/objects/` loose | ~123 G | aborted adds of large files | NO (clone instead) |
| `ckpts/` | 3.1 T | trained + pretrained checkpoints | NO (metadata) |
| `models/` | 514 G | model weights | NO (metadata) |
| `hf_cache/` | 127 G | HF cache | NO |
| `data_synthesis/outputs/` | 1.8 T | synthesized training data + eval TSVs | NO (metadata) |
| `data_synthesis/MFA_backup/` | 54 G | MFA backup | NO (review) |
| `conda_envs/` | 27 G | binary envs | NO (env captured as text) |
| `uv_cache/` | 12 G | cache | NO |
| `data_synthesis/codes/gigaspeech/hibiki/output/` | 3.7 G | hibiki decode outputs | NO |
| `vscode-server/`, `pip_cache/`, `conda_pkgs/` | ~5.7 G | caches | NO |
| `simul_tst_common/models/` | 1.5 G | whisper/MFA models | NO |
| `scripts/infer/slurm_logs/` | 1.2 G | old job logs | NO |
| `IDL_final_3small*` | 513 M | old course project | REVIEW |
| `ANLP/` | 320 M | old course dir | REVIEW |
| `simul_tst_common/mfa_tmp/` | 285 M | MFA temp | NO |
| `tools/` | 124 M | SEGALE/metricx checkouts | REVIEW |
| `code/` | 111 M | Megatron-LM/OmniSTEval clones | NO (re-clone) |
| `simul_tst_common/whisper_venv/` | 101 M | venv | NO |

No single tracked file exceeds 100 MB (tracked total is 20 MB).

## 2026-08-28 addendum — `/home` checkout

Code now lives in `/home/haolingp/CMU_research_SMT` (clone + overlay, code paths rewritten);
see `migration/HOME_CHECKOUT.md`. The Mac pulls from that clone. Data paths unchanged.

## Phases 4–10 outputs

- `migration/environment/` — system/GPU/conda/pip captures (Phase 4)
- `migration/PROJECT_WORKFLOW.md` — how the project actually runs (Phase 5)
- `migration/BABEL_PATHS.md` — remote-only paths + where they're hardcoded (Phase 6)
- `migration/include.txt`, `migration/exclude.txt`, `migration/EXCLUDED_FILES.md` (Phase 7)
- `migration/PULL_TO_MAC.md` — clone + rsync commands to run FROM the Mac (Phase 8)
- `migration/GIT_STATE.md` — branch/HEAD/dirty-state snapshot (Phase 9)
- `migration/PREEMPTION_READINESS.md` + tools/babel design (Phase 10)

Estimated total local transfer: **~175 MB** (measured staged size) (plus a few-MB git clone).
