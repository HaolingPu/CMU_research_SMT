"""Synthesize the development trajectory of the survey topic.

Three Opus 4.7 stages:
  1. cluster all papers into 6-10 technique-family threads
  2. per-cluster narrative (sequential) — seminal works, evolution, recent thrust, open problems
  3. cross-cutting synthesis — cross-pollination, eras, eval gaps, future directions

The topic header (TOPIC_HEADER in config.py) is the system prompt for all three stages.

Outputs:
  - synthesis.md             — final document
  - synthesis.cache.json     — intermediate state, supports incremental re-runs

Run:
    ANTHROPIC_API_KEY=$(cat ~/.keys/claude) \\
    PYTHONPATH=~/.local/lib/python3.9/site-packages \\
    python synthesize_field.py
    # re-run from a specific stage:
    python synthesize_field.py --force-stage 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import site
import sys
import time
from pathlib import Path

_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

from anthropic import AsyncAnthropic  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config import TOPIC_HEADER, TOPIC_SHORT, VENUE_DESCRIPTION  # noqa: E402


HERE = Path(__file__).parent
RESULTS_JSON = HERE / "results.json"
CACHE_PATH = HERE / "synthesis.cache.json"
OUTPUT_MD = HERE / "synthesis.md"

MODEL_ID = "claude-opus-4-7"
# Adaptive thinking is off by default on Opus 4.7 — enable explicitly for
# synthesis-quality reasoning. effort="high" is the recommended minimum for
# intelligence-sensitive work; effort matters more on 4.7 than any prior Opus.
THINKING = {"type": "adaptive"}
EFFORT = "high"
ABSTRACT_BUDGET = 1200            # Stage 1 (whole-corpus clustering)
ABSTRACT_BUDGET_S2 = 600          # Stage 2 (per-cluster narrative)
SDK_MAX_RETRIES = 5
# Per-minute input-token rate limits are model-specific. 90s pacing is
# conservative for the 30K tok/min observed limit; tighten if no 429s.
INTER_CALL_SLEEP_S2 = 90.0


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class Cluster(BaseModel):
    name: str
    description: str
    paper_indices: list[int]


class ClusterSet(BaseModel):
    clusters: list[Cluster]


class ClusterNarrative(BaseModel):
    cluster_index: int
    seminal_works_md: str
    evolution_md: str
    recent_thrust_md: str
    open_problems_md: str


class CrossCutting(BaseModel):
    cross_pollination_md: str
    eras_md: str
    eval_gaps_md: str
    future_directions_md: str


# ---------------------------------------------------------------------------
# Paper formatting
# ---------------------------------------------------------------------------


def load_papers() -> list[dict]:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def format_papers_for_clustering(papers: list[dict]) -> str:
    """Compact textual form for Stage 1 — minimize tokens."""
    lines = []
    for i, p in enumerate(papers):
        title = (p.get("title") or "(no title)").strip()
        year = p.get("year") or "?"
        venue = p.get("_venue_group") or p.get("venue") or "?"
        cites = p.get("citationCount") or 0
        abstract = (p.get("abstract") or "").strip()[:ABSTRACT_BUDGET]
        if abstract:
            lines.append(f"[{i}] ({year}, {venue}, {cites}c) {title}\n  {abstract}")
        else:
            lines.append(f"[{i}] ({year}, {venue}, {cites}c) {title}")
    return "\n\n".join(lines)


def format_papers_for_narrative(papers: list[dict], indices: list[int],
                                 abstract_budget: int = ABSTRACT_BUDGET_S2) -> str:
    """Format only the papers in this cluster, sorted chronologically with citations."""
    subset = [(i, papers[i]) for i in indices if 0 <= i < len(papers)]
    subset.sort(key=lambda t: ((t[1].get("year") or 0),
                                -(t[1].get("citationCount") or 0)))
    lines = []
    for i, p in subset:
        title = (p.get("title") or "(no title)").strip()
        year = p.get("year") or "?"
        venue = p.get("_venue_group") or p.get("venue") or "?"
        cites = p.get("citationCount") or 0
        authors_list = p.get("authors") or []
        authors = ", ".join(a.get("name", "") for a in authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        abstract = (p.get("abstract") or "").strip()[:abstract_budget]
        head = f"[{i}] ({year}, {venue}, {cites}c) {authors}. {title}"
        if abstract:
            lines.append(f"{head}\n  {abstract}")
        else:
            lines.append(head)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 — clustering
# ---------------------------------------------------------------------------


async def stage1_cluster(client: AsyncAnthropic, papers: list[dict],
                         log_usage) -> ClusterSet:
    text = format_papers_for_clustering(papers)
    user_msg = f"""Below are {len(papers)} papers from this survey. Each is `[index] (year, venue, cites) Title` followed by abstract.

Group them into **6 to 10 clusters** organized by *technique family* (NOT by venue, year, or task). Each cluster needs:
- `name`: short, technical (e.g. "Contextual biasing in E2E ASR")
- `description`: one sentence on what unifies the cluster
- `paper_indices`: integer indices [0..{len(papers)-1}] in this cluster

Hard rules:
- Every paper goes into exactly ONE cluster — no orphans, no duplicates.
- A cluster needs to be a meaningful research thread, not a catch-all. If a paper is genuinely off-topic (e.g., pulled in by citation expansion but unrelated to the survey topic), put it in a final cluster called "Other / off-topic / out of scope".
- Aim for clusters of 30-100 papers. Avoid one giant 200+ cluster — split by technique sub-family if needed.
- Names should be technique-specific. Pick technique-specific phrasing rather than broad task names.

Papers:

{text}"""

    # Stage 1 is bookkeeping (assign N indices to clusters) — not
    # intelligence-sensitive. Keep all output tokens for the structured payload.
    response = await client.messages.parse(
        model=MODEL_ID,
        max_tokens=16000,
        output_config={"effort": EFFORT},
        system=[{"type": "text", "text": TOPIC_HEADER,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=ClusterSet,
    )
    log_usage(response.usage)
    if response.parsed_output is None:
        raise RuntimeError(
            f"Stage 1 parse failed: stop_reason={response.stop_reason!r}, "
            f"content_types={[b.type for b in response.content]}, "
            f"usage={response.usage}")
    return response.parsed_output


# ---------------------------------------------------------------------------
# Stage 2 — per-cluster narrative
# ---------------------------------------------------------------------------


async def stage2_narrative(client: AsyncAnthropic,
                           cluster_idx: int, cluster: Cluster,
                           papers: list[dict], log_usage) -> ClusterNarrative:
    text = format_papers_for_narrative(papers, cluster.paper_indices)
    user_msg = f"""Cluster: **{cluster.name}**
Description: {cluster.description}
Papers in cluster ({len(cluster.paper_indices)}, sorted by year then citations):

{text}

Write a markdown narrative section about how this thread of research evolved. Output exactly four fields:

`seminal_works_md` — Identify 3-5 most influential papers (high citations OR field-defining). For each: one paragraph noting authors+year, key contribution, and why it mattered to this cluster.

`evolution_md` — A chronological prose narrative (~250-400 words) tracing how techniques in this cluster shifted over time. Cite papers inline by author+year. Mention specific techniques. No bullet lists. Connect papers as a story — what did one paper enable the next to do?

`recent_thrust_md` — ~100-150 words. What's distinctively new in the most recent 1-2 years of work? What problems are recent papers attacking that earlier work in this cluster could not?

`open_problems_md` — ~80-120 words. What do recent papers in this cluster explicitly call out as unsolved? Be specific — name evaluation gaps, scaling barriers, or methodology limits.

**CITATION FORMAT — STRICT.** Every paper you mention must be cited using a markdown link with the paper's index, like `[Pundak et al. 2018](#p_42)`. The index is the number in square brackets at the start of each paper's entry above (so `[42] (2018, Interspeech, ...)` means use `(#p_42)` for that paper). Apply this in ALL four sub-sections including `seminal_works_md`. NEVER cite a paper that isn't in the cluster list above — only cite papers you can find an index for. If you'd want to mention a foundational reference outside this cluster, prefer a generic technique name without a citation rather than an unindexed citation.

Be concrete. Avoid filler ("Many papers have studied X."). The cluster_index field of your output should be {cluster_idx}."""

    # Adaptive thinking adds output overhead — give 16K headroom to ensure the
    # structured payload still emits after the thinking block.
    response = await client.messages.parse(
        model=MODEL_ID,
        max_tokens=16000,
        thinking=THINKING,
        output_config={"effort": EFFORT},
        system=[{"type": "text", "text": TOPIC_HEADER,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=ClusterNarrative,
    )
    log_usage(response.usage)
    if response.parsed_output is None:
        raise RuntimeError(
            f"Stage 2 cluster {cluster_idx} parse failed: "
            f"stop_reason={response.stop_reason!r}, "
            f"content_types={[b.type for b in response.content]}, "
            f"usage={response.usage}")
    return response.parsed_output


# ---------------------------------------------------------------------------
# Stage 3 — cross-cutting synthesis
# ---------------------------------------------------------------------------


async def stage3_cross_cutting(client: AsyncAnthropic,
                                clusters: list[Cluster],
                                narratives: list[ClusterNarrative],
                                log_usage) -> CrossCutting:
    summary_blob = "\n\n".join(
        f"### Cluster {i}: {c.name} ({len(c.paper_indices)} papers)\n"
        f"Description: {c.description}\n"
        f"Recent thrust: {n.recent_thrust_md}\n"
        f"Open problems: {n.open_problems_md}"
        for i, (c, n) in enumerate(zip(clusters, narratives))
    )
    user_msg = f"""Below are summaries of {len(clusters)} clusters from this survey. Each cluster is a research thread.

{summary_blob}

Write four cross-cutting markdown sub-sections:

`cross_pollination_md` — ~250 words. Which techniques have crossed between sub-areas of this field? Where did methods originate, and how were they adapted? Cite specific cluster names and exemplar papers (author+year if you can recall from the summaries).

`eras_md` — ~250 words. Identify 3-5 chronological eras visible across multiple clusters. For each era: name dominant approaches and the conceptual shift that ended it. Eras overlap — don't put hard year boundaries.

`eval_gaps_md` — ~150 words. What evaluation methodology gaps does the field collectively share? Be specific — datasets, metrics, missing benchmarks.

`future_directions_md` — ~150 words. Synthesize the open-problem statements across clusters into 3-5 concrete future directions. Be specific — concrete benchmarks, methods, or evaluations rather than "more data".

**CITATION FORMAT — STRICT.** The cluster summaries above already contain markdown citation links of the form `[Author YYYY](#p_NNN)`. When you cite a specific paper in your output, you MUST use that exact link form (copy the `(#p_NNN)` from the cluster summary). Never write a paper citation as bare `Author YYYY` without the link — if you don't have the index from the summaries, refer to the technique by name and the cluster by name instead (e.g. "retrieval-augmented LLM correction (Cluster 1)") rather than emitting an unlinked citation."""

    response = await client.messages.parse(
        model=MODEL_ID,
        max_tokens=12000,
        thinking=THINKING,
        output_config={"effort": EFFORT},
        system=[{"type": "text", "text": TOPIC_HEADER,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=CrossCutting,
    )
    log_usage(response.usage)
    if response.parsed_output is None:
        raise RuntimeError(
            f"Stage 3 parse failed: stop_reason={response.stop_reason!r}, "
            f"content_types={[b.type for b in response.content]}, "
            f"usage={response.usage}")
    return response.parsed_output


# ---------------------------------------------------------------------------
# Stage 4 — assemble synthesis.md
# ---------------------------------------------------------------------------


def assemble_markdown(papers: list[dict], clusters: list[Cluster],
                      narratives: list[ClusterNarrative],
                      cross: CrossCutting) -> str:
    lines: list[str] = []
    lines.append(f"# Development trajectory: {TOPIC_SHORT}")
    lines.append("")
    years = [p.get("year") for p in papers if p.get("year")]
    year_range = f"{min(years)}–{max(years)}" if years else "?"
    lines.append(f"_Synthesized from {len(papers)} papers across {VENUE_DESCRIPTION}, "
                 f"{year_range}. Generated by {MODEL_ID}._")
    lines.append("")

    lines.append("## Taxonomy at a glance")
    lines.append("")
    for c in clusters:
        lines.append(f"- **{c.name}** ({len(c.paper_indices)} papers) — {c.description}")
    lines.append("")

    lines.append("## Cross-cutting trends")
    lines.append("")
    lines.append("### How techniques cross between sub-areas")
    lines.append("")
    lines.append(cross.cross_pollination_md)
    lines.append("")
    lines.append("### Eras of dominant approaches")
    lines.append("")
    lines.append(cross.eras_md)
    lines.append("")
    lines.append("### Evaluation gaps the field shares")
    lines.append("")
    lines.append(cross.eval_gaps_md)
    lines.append("")
    lines.append("### Future directions")
    lines.append("")
    lines.append(cross.future_directions_md)
    lines.append("")

    # Per-cluster sections, ordered by paper count descending
    pairs = sorted(zip(clusters, narratives),
                   key=lambda t: -len(t[0].paper_indices))
    lines.append("## Per-thread deep dives")
    lines.append("")
    for c, n in pairs:
        lines.append(f"### {c.name} ({len(c.paper_indices)} papers)")
        lines.append("")
        lines.append(f"_{c.description}_")
        lines.append("")
        lines.append("**Seminal works**")
        lines.append("")
        lines.append(n.seminal_works_md)
        lines.append("")
        lines.append("**Evolution**")
        lines.append("")
        lines.append(n.evolution_md)
        lines.append("")
        lines.append("**Recent thrust**")
        lines.append("")
        lines.append(n.recent_thrust_md)
        lines.append("")
        lines.append("**Open problems**")
        lines.append("")
        lines.append(n.open_problems_md)
        lines.append("")

    # Appendix: cluster assignment table for spot-checking
    paper_to_cluster: dict[int, str] = {}
    for c in clusters:
        for idx in c.paper_indices:
            paper_to_cluster[idx] = c.name

    lines.append("## Appendix: cluster assignments")
    lines.append("")
    lines.append("| # | Cluster | Year | Venue | Title |")
    lines.append("|---|---|---|---|---|")
    for i, p in enumerate(papers):
        cn = paper_to_cluster.get(i, "_unassigned_")
        title = (p.get("title") or "(no title)").replace("|", "\\|")[:120]
        year = p.get("year") or "?"
        venue = p.get("_venue_group") or "?"
        lines.append(f"| {i} | {cn} | {year} | {venue} | {title} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-stage", type=int, choices=[1, 2, 3],
                    help="Re-run from this stage onwards (busts cache)")
    args = ap.parse_args()

    papers = load_papers()
    print(f"Loaded {len(papers)} papers from {RESULTS_JSON}", file=sys.stderr)

    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            print(f"Loaded cache: stages {sorted(cache.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"Cache unreadable ({e}), starting fresh", file=sys.stderr)

    if args.force_stage is not None:
        cleared = []
        for s in (1, 2, 3):
            if s >= args.force_stage and str(s) in cache:
                del cache[str(s)]
                cleared.append(s)
        if cleared:
            print(f"--force-stage={args.force_stage}: cleared cache for stages {cleared}",
                  file=sys.stderr)

    client = AsyncAnthropic(max_retries=SDK_MAX_RETRIES)
    totals = {"writes": 0, "reads": 0, "uncached_in": 0, "out": 0}

    def log_usage(u) -> None:
        totals["writes"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        totals["reads"] += getattr(u, "cache_read_input_tokens", 0) or 0
        totals["uncached_in"] += getattr(u, "input_tokens", 0) or 0
        totals["out"] += getattr(u, "output_tokens", 0) or 0

    def save_cache() -> None:
        CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    # Snapshot: was Stage 1 already cached when we started?
    cache_at_start = set(cache.keys())
    stage1_started_at = 0.0

    # ---- Stage 1: cluster ----
    if "1" in cache:
        cluster_set = ClusterSet(**cache["1"])
        print(f"Stage 1 cached: {len(cluster_set.clusters)} clusters",
              file=sys.stderr)
    else:
        print("Stage 1: clustering...", file=sys.stderr)
        stage1_started_at = time.monotonic()
        cluster_set = await stage1_cluster(client, papers, log_usage)
        stage1_started_at = time.monotonic()  # treat as the "last activity" marker
        cache["1"] = cluster_set.model_dump()
        save_cache()
        print(f"Stage 1 complete: {len(cluster_set.clusters)} clusters",
              file=sys.stderr)

    for i, c in enumerate(cluster_set.clusters):
        print(f"  [{i}] {c.name}  ({len(c.paper_indices)} papers)",
              file=sys.stderr)

    # validation
    seen: dict[int, int] = {}
    duplicates: list[tuple[int, int, int]] = []
    for ci, c in enumerate(cluster_set.clusters):
        for idx in c.paper_indices:
            if idx in seen:
                duplicates.append((idx, seen[idx], ci))
            seen[idx] = ci
    missing = [i for i in range(len(papers)) if i not in seen]
    if duplicates:
        print(f"  WARN: {len(duplicates)} papers in multiple clusters "
              f"(first 5: {duplicates[:5]})", file=sys.stderr)
    if missing:
        print(f"  WARN: {len(missing)} papers unassigned (first 10: {missing[:10]})",
              file=sys.stderr)

    # Track wall-clock time of last live API call for rate-limit pacing.
    last_api_activity = stage1_started_at if "1" not in cache_at_start else 0.0

    async def pace_before_call() -> None:
        nonlocal last_api_activity
        elapsed = time.monotonic() - last_api_activity
        if elapsed < INTER_CALL_SLEEP_S2:
            wait = INTER_CALL_SLEEP_S2 - elapsed
            print(f"  sleeping {wait:.0f}s to respect 30K tok/min limit "
                  f"(last live call {elapsed:.0f}s ago)...", file=sys.stderr)
            await asyncio.sleep(wait)

    # ---- Stage 2: per-cluster narrative (sequential to respect 30K tok/min) ----
    cached_n: dict[int, ClusterNarrative] = {}
    if "2" in cache:
        for n_dict in cache["2"]:
            cn = ClusterNarrative(**n_dict)
            cached_n[cn.cluster_index] = cn
        print(f"Stage 2: {len(cached_n)} narratives already cached",
              file=sys.stderr)

    narratives: list[ClusterNarrative] = []
    for i, c in enumerate(cluster_set.clusters):
        if i in cached_n:
            narratives.append(cached_n[i])
            continue

        # Skip off-topic cluster — no narrative worth synthesizing for noise.
        if "off-topic" in c.name.lower() or "out of scope" in c.name.lower():
            print(f"Stage 2 [{i}] skipped (off-topic): {c.name}", file=sys.stderr)
            narratives.append(ClusterNarrative(
                cluster_index=i,
                seminal_works_md="_(skipped — out-of-scope cluster, mostly citation-expansion noise)_",
                evolution_md="_(skipped)_",
                recent_thrust_md="_(skipped)_",
                open_problems_md="_(skipped)_",
            ))
            cache["2"] = [n.model_dump() for n in narratives]
            save_cache()
            continue

        await pace_before_call()
        print(f"Stage 2 [{i}] generating: {c.name} "
              f"({len(c.paper_indices)} papers)...", file=sys.stderr)
        n = await stage2_narrative(client, i, c, papers, log_usage)
        last_api_activity = time.monotonic()
        narratives.append(n)
        # Save after every call so partial progress survives a Ctrl-C or 429.
        cache["2"] = [nn.model_dump() for nn in narratives]
        save_cache()
    print("Stage 2 complete", file=sys.stderr)

    # ---- Stage 3: cross-cutting ----
    if "3" in cache:
        cross = CrossCutting(**cache["3"])
        print("Stage 3 cached", file=sys.stderr)
    else:
        await pace_before_call()
        print("Stage 3: cross-cutting synthesis...", file=sys.stderr)
        cross = await stage3_cross_cutting(client, cluster_set.clusters,
                                            narratives, log_usage)
        last_api_activity = time.monotonic()
        cache["3"] = cross.model_dump()
        save_cache()
        print("Stage 3 complete", file=sys.stderr)

    # ---- Stage 4: assemble ----
    print("Assembling synthesis.md...", file=sys.stderr)
    md = assemble_markdown(papers, cluster_set.clusters, narratives, cross)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD} ({len(md):,} chars, "
          f"~{len(md.split()):,} words)", file=sys.stderr)

    print(f"\nToken usage  uncached_in={totals['uncached_in']:,}  "
          f"cache_writes={totals['writes']:,}  "
          f"cache_reads={totals['reads']:,}  "
          f"out={totals['out']:,}", file=sys.stderr)
    cost = (totals["uncached_in"] / 1e6 * 3.0
            + totals["writes"] / 1e6 * 3.75      # 1.25× base input
            + totals["reads"] / 1e6 * 0.30       # 0.1× base input
            + totals["out"] / 1e6 * 15.0)
    print(f"Estimated API cost: ${cost:.3f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
