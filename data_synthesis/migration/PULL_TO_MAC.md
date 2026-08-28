# Pulling the project to your Mac

> **2026-08-28 update:** the canonical code copy is now the clone at `/home/haolingp/CMU_research_SMT`
> (see `HOME_CHECKOUT.md`). Commands below pull from it directly; `migration_staging/` +
> `stage_to_home.sh` are superseded. Branch to check out is `feature/home-checkout` once pushed.

## Why not a plain `rsync babel:/data/user_data/...`

Two Babel-specific facts shape the strategy:

1. **Login nodes do not mount `/data/user_data`** — your Mac's ssh lands on a login node,
   so a direct rsync of the project path fails (or sees an empty AutoFS stub). The
   include-set is therefore **staged into `/home/haolingp/migration_staging/`** (home is
   mounted everywhere). The staging copy already exists and can be refreshed any time with:
   `ssh babel "~/bin/ondata 'bash /data/user_data/haolingp/data_synthesis/migration/stage_to_home.sh'"`
2. **`.git` on Babel is 966 GB of junk** (843 GB orphaned tmp_pack files + ~123 GB loose
   objects; real pack = 1.1 MB) — never rsync it. Since `feature/llm-wiki` is fully pushed,
   **git history comes from GitHub via clone**, and rsync only layers the uncommitted /
   untracked working files on top. After the rsync, `git status` on the Mac shows exactly
   the same dirty state as Babel — nothing is lost, nothing is duplicated.

## Step 1 — clone (history, ~a few MB)

```bash
mkdir -p /Users/haolingpu/Desktop/research
git clone git@github.com:HaolingPu/CMU_research_SMT.git /Users/haolingpu/Desktop/research/CMU_research_SMT
cd /Users/haolingpu/Desktop/research/CMU_research_SMT
git checkout feature/home-checkout   # canonical branch (paths rewritten to /home checkout)
```

## Step 2 — dry-run the overlay rsync (from the Mac)

```bash
rsync -avzn \
  --exclude .git babel:CMU_research_SMT/ \
  /Users/haolingpu/Desktop/research/CMU_research_SMT/
```

Expect ~175 MB of mostly untracked research files (EMNLP paper dir, east de/ja pipeline
scripts, migration/, wiki working copies, plots). No `--delete` — the clone's files are
never removed, identical tracked files are simply skipped/overwritten in place.

## Step 3 — real transfer

```bash
rsync -avz \
  --exclude .git babel:CMU_research_SMT/ \
  /Users/haolingpu/Desktop/research/CMU_research_SMT/
```

## Step 4 — verify on the Mac

```bash
cd /Users/haolingpu/Desktop/research/CMU_research_SMT
git status          # should mirror migration/GIT_STATE.md (4 modified + ~195 untracked)
ls data_synthesis/migration/environment/   # env captures present
```

Note: `data_synthesis/codes/metricx` is a gitlink without `.gitmodules`; after the overlay
its files are present but git shows it as a modified submodule — same as on Babel. Leave it.

## Assumptions

- `babel` is an ssh alias to a login node with your key loaded
  (`Host babel → HostName login.babel.cs.cmu.edu, User haolingp`).
- If you prefer a single-shot pull without staging, the alternative is ProxyJump to a
  compute node running your anchor job, but node names change — staging is the stable path.

## Keeping in sync later

Re-run the staging refresh + Step 3 whenever you want Babel-side changes pulled down.
For the reverse direction (Mac → Babel), see the `tools/babel/sync.sh` design in
`PREEMPTION_READINESS.md` — push through git (preferred) or rsync Mac→`/home` staging →
`ondata` rsync into `/data/user_data`.
