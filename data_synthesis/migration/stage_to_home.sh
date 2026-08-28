#!/bin/bash
# SUPERSEDED 2026-08-28: canonical code copy is /home/haolingp/CMU_research_SMT (see HOME_CHECKOUT.md). Kept for reference.
# Stage the Mac-bound subset of the repo into /home (mounted on ALL Babel nodes,
# including login nodes — /data/user_data is NOT). Run on a compute node, or via:
#   ~/bin/ondata 'bash /data/user_data/haolingp/data_synthesis/migration/stage_to_home.sh'
# Re-run any time to refresh the staging copy (no --delete; additive only).
set -e
SRC=/data/user_data/haolingp
DEST=/home/haolingp/migration_staging/CMU_research_SMT
EXCLUDES=$SRC/data_synthesis/migration/exclude.txt
mkdir -p "$DEST"
rsync -a --exclude-from="$EXCLUDES" "$SRC/" "$DEST/"
du -sh "$DEST"
echo "staged OK: $DEST"
