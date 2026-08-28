"""Smoke test for the indexed-citation prompt format.

Runs ONE Stage 2 narrative call on the smallest non-skipped cluster (typically
~13 papers) with the synthesis prompts, then audits the output:
  1. Are there any `[Author YYYY](#p_NNN)` markdown links?
  2. Do the indices in those links land within the cluster's paper_indices?
  3. Are there leftover unlinked "Author et al. YYYY" citations the prompt
     should have caught?

Cost: ~5K input + ~3K output tokens on Opus 4.7 ≈ $0.10. Quick and cheap.

Run AFTER `synthesize_field.py` has populated Stage 1 of `synthesis.cache.json`:

    ANTHROPIC_API_KEY=$(cat ~/.keys/claude) \\
    PYTHONPATH=~/.local/lib/python3.9/site-packages \\
    python smoke_test_citations.py
"""

from __future__ import annotations

import asyncio
import json
import re
import site
import sys
from pathlib import Path

_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

# Import the orchestrator pieces we need
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from synthesize_field import (  # noqa: E402
    Cluster,
    load_papers,
    stage2_narrative,
)
from anthropic import AsyncAnthropic  # noqa: E402


CACHE_PATH = HERE / "synthesis.cache.json"

LINK_RE = re.compile(r"\[([^\]]+)\]\(#p_(\d+)\)")
BARE_CITE_RE = re.compile(
    r"\b[A-Z][a-zA-Z'\-]+(?:\s+et\s+al\.?|(?:,\s*[A-Z][a-zA-Z'\-]+)*\s*&\s*[A-Z][a-zA-Z'\-]+)"
    r"(?:,)?\s*\(?\d{4}\)?"
)


async def main() -> int:
    if not CACHE_PATH.exists():
        print(f"ERROR: {CACHE_PATH} not found — run synthesize_field.py first "
              "to populate Stage 1 cache.", file=sys.stderr)
        return 1
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if "1" not in cache:
        print("ERROR: synthesis.cache.json has no Stage 1 result", file=sys.stderr)
        return 1

    papers = load_papers()
    clusters_dict = cache["1"]["clusters"]

    # Smallest non-off-topic cluster
    candidates = []
    for i, c in enumerate(clusters_dict):
        name_lc = c["name"].lower()
        if any(s in name_lc for s in ("off-topic", "out-of-scope",
                                       "out of scope", "off topic")):
            continue
        candidates.append((len(c["paper_indices"]), i, c))
    candidates.sort()
    if not candidates:
        print("ERROR: no non-skipped clusters in cache", file=sys.stderr)
        return 1
    _, target_idx, target_dict = candidates[0]
    target = Cluster(**target_dict)

    print(f"Smoke test target: cluster {target_idx} = "
          f"{target.name!r} ({len(target.paper_indices)} papers)\n",
          file=sys.stderr)

    client = AsyncAnthropic(max_retries=5)
    totals = {"in": 0, "out": 0, "thinking_blocks": 0}

    def log_usage(u) -> None:
        totals["in"] += getattr(u, "input_tokens", 0) or 0
        totals["out"] += getattr(u, "output_tokens", 0) or 0

    n = await stage2_narrative(client, target_idx, target, papers, log_usage)

    full = "\n\n".join([
        "## seminal_works_md", n.seminal_works_md,
        "## evolution_md", n.evolution_md,
        "## recent_thrust_md", n.recent_thrust_md,
        "## open_problems_md", n.open_problems_md,
    ])

    print("=" * 70)
    print("NARRATIVE OUTPUT")
    print("=" * 70)
    print(full)
    print("=" * 70)

    # Audit: indexed links
    links = LINK_RE.findall(full)
    print(f"\nFound {len(links)} indexed links of form [Text](#p_NNN)")

    cluster_set = set(target.paper_indices)
    in_cluster = 0
    out_of_cluster_examples: list[tuple[str, int, str]] = []
    for text, idx_str in links:
        idx = int(idx_str)
        if idx in cluster_set:
            in_cluster += 1
        else:
            actual_title = (papers[idx]["title"] or "(no title)")[:80] if 0 <= idx < len(papers) else "(out-of-range)"
            out_of_cluster_examples.append((text, idx, actual_title))

    print(f"  ✓ {in_cluster}/{len(links)} indices land within the cluster")
    if out_of_cluster_examples:
        print(f"  ✗ {len(out_of_cluster_examples)} indices outside cluster (examples):")
        for text, idx, title in out_of_cluster_examples[:5]:
            print(f"      [{idx}] {text!r} → {title}")

    # Audit: bare unlinked citations the prompt should have linked
    full_minus_links = LINK_RE.sub("", full)
    bare = BARE_CITE_RE.findall(full_minus_links)
    print(f"\n{len(bare)} unlinked bare citations remaining (sample):")
    for b in bare[:10]:
        print(f"      {b!r}")

    print(f"\nTokens: input={totals['in']:,}  output={totals['out']:,}",
          file=sys.stderr)

    # Summary verdict
    print("\n" + "=" * 70)
    if links and in_cluster == len(links) and len(bare) <= 2:
        print("VERDICT: ✓ Prompt format works. "
              "All indices in-cluster, no significant bare citations.")
        return 0
    else:
        print("VERDICT: ⚠ Review output above before re-running full pipeline.")
        if not links:
            print("  - No indexed links found (model didn't follow format)")
        if links and in_cluster < len(links):
            print("  - Some indices land outside the cluster (model may be guessing)")
        if len(bare) > 2:
            print(f"  - {len(bare)} unlinked citations leaked through")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
