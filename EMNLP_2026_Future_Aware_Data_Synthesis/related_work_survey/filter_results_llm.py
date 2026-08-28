"""LLM-as-judge filter for results.json using Claude Sonnet 4.6.

For each paper (title + truncated abstract), Sonnet 4.6 returns a
`{related: bool, reason: str}` judgment. Removed papers are printed to stdout
with the model's reason. Filtered results are written back to results.json
and results.md.

The topic rubric (TOPIC_RUBRIC in config.py) lives in the system prompt
with a `cache_control` breakpoint, so the rubric is paid for once (cache
write on the first call) and read at ~0.1× cost for every subsequent paper.

Run:
    ANTHROPIC_API_KEY=$(cat ~/.keys/claude) PYTHONPATH=~/.local/lib/python3.9/site-packages python filter_results_llm.py
"""

from __future__ import annotations

import asyncio
import json
import re
import site
import sys
from collections import defaultdict
from pathlib import Path

# `pip install --user` site-packages aren't always on sys.path in this env.
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

from anthropic import AsyncAnthropic  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config import TOPIC_RUBRIC, TOPIC_SHORT, VENUE_ORDER  # noqa: E402


HERE = Path(__file__).parent
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"

MODEL_ID = "claude-sonnet-4-6"
# Org limit observed: 30K input tokens/min on this tier. At ~1K tokens/call,
# concurrency=5 keeps us well under that with the SDK's auto-backoff.
CONCURRENCY = 5
ABSTRACT_CHAR_BUDGET = 1500
SDK_MAX_RETRIES = 5


class Judgment(BaseModel):
    related: bool
    reason: str  # short, ≤ 30 words


async def judge_paper(client: AsyncAnthropic, sem: asyncio.Semaphore,
                     paper: dict, log_usage) -> tuple[dict, Judgment | None]:
    title = paper.get("title") or "(untitled)"
    abstract = (paper.get("abstract") or "")[:ABSTRACT_CHAR_BUDGET]
    user_msg = f"Title: {title}\n\nAbstract: {abstract}".strip()
    async with sem:
        try:
            response = await client.messages.parse(
                model=MODEL_ID,
                max_tokens=300,
                system=[{
                    "type": "text",
                    "text": TOPIC_RUBRIC,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
                output_format=Judgment,
            )
        except Exception as e:
            print(f"  ERROR judging {paper.get('paperId')}: {e}", file=sys.stderr)
            return paper, None
    log_usage(response.usage)
    return paper, response.parsed_output


async def main() -> None:
    if not RESULTS_JSON.exists():
        print(f"ERROR: {RESULTS_JSON} not found", file=sys.stderr)
        sys.exit(1)
    papers = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(papers)} papers from {RESULTS_JSON}", file=sys.stderr)

    client = AsyncAnthropic(max_retries=SDK_MAX_RETRIES)
    sem = asyncio.Semaphore(CONCURRENCY)

    totals = {"writes": 0, "reads": 0, "uncached_in": 0, "out": 0}

    def log_usage(u) -> None:
        totals["writes"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        totals["reads"] += getattr(u, "cache_read_input_tokens", 0) or 0
        totals["uncached_in"] += getattr(u, "input_tokens", 0) or 0
        totals["out"] += getattr(u, "output_tokens", 0) or 0

    # Skip papers that were already judged in a prior run — supports
    # incremental retry of failures without re-paying for completed calls.
    already_judged: list[dict] = []
    to_judge: list[dict] = []
    for p in papers:
        if "_llm_related" in p:
            already_judged.append(p)
        else:
            to_judge.append(p)
    if already_judged:
        print(f"Skipping {len(already_judged)} papers already judged in a prior run",
              file=sys.stderr)

    judged: list[tuple[dict, Judgment | None]] = []
    if to_judge:
        # Warm-up: judge the first paper alone so the cache is written before we
        # fan out. Without this, parallel calls all miss cache (none can read
        # what the others are still writing).
        print(f"Warm-up call to populate prompt cache...", file=sys.stderr)
        warm = await judge_paper(client, sem, to_judge[0], log_usage)
        judged.append(warm)

        rest = to_judge[1:]
        if rest:
            print(f"Fanning out {len(rest)} judgments at concurrency={CONCURRENCY}...",
                  file=sys.stderr)
            judged.extend(await asyncio.gather(
                *(judge_paper(client, sem, p, log_usage) for p in rest)
            ))

    # Re-attach prior judgments unchanged
    for p in already_judged:
        j = Judgment(related=p["_llm_related"], reason=p.get("_llm_reason", ""))
        judged.append((p, j))

    kept: list[dict] = []
    removed: list[dict] = []
    failed: list[dict] = []
    for paper, j in judged:
        if j is None:
            failed.append(paper)
            continue
        paper["_llm_related"] = j.related
        paper["_llm_reason"] = j.reason
        (kept if j.related else removed).append(paper)

    print(f"\nTotal: {len(papers)}  Kept: {len(kept)}  "
          f"Removed: {len(removed)}  Failed: {len(failed)}", file=sys.stderr)
    print(f"Tokens — cache writes: {totals['writes']:,}  "
          f"cache reads: {totals['reads']:,}  "
          f"uncached input: {totals['uncached_in']:,}  "
          f"output: {totals['out']:,}", file=sys.stderr)
    if totals["reads"] + totals["writes"] > 0:
        hit_rate = totals["reads"] / (totals["reads"] + totals["writes"])
        print(f"Cache hit rate (by tokens): {hit_rate:.1%}", file=sys.stderr)

    print("=" * 78)
    print("REMOVED PAPERS (LLM judge)")
    print("=" * 78)
    by_venue: dict[str, list[dict]] = defaultdict(list)
    for p in removed:
        by_venue[p.get("_venue_group", "?")].append(p)
    for venue in sorted(by_venue):
        print(f"\n--- {venue} ({len(by_venue[venue])}) ---")
        for p in by_venue[venue]:
            title = p.get("title") or "(untitled)"
            year = p.get("year") or "?"
            print(f"  [{year}] {title}")
            print(f"        reason: {p.get('_llm_reason')}")

    if failed:
        print("\n" + "=" * 78)
        print("FAILED (kept in results.json unchanged)")
        print("=" * 78)
        for p in failed:
            print(f"  {p.get('title')}")
        # Don't drop failures; leave them in `kept` to avoid silent loss.
        kept.extend(failed)

    RESULTS_JSON.write_text(json.dumps(kept, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nWrote {len(kept)} kept papers back to {RESULTS_JSON}",
          file=sys.stderr)
    write_markdown(kept, RESULTS_MD)
    print(f"Rewrote {RESULTS_MD}", file=sys.stderr)


def write_markdown(papers: list[dict], path: Path) -> None:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        by_group[p.get("_venue_group", "?")].append(p)

    lines: list[str] = []
    lines.append(f"# Literature survey: {TOPIC_SHORT}")
    lines.append("")
    lines.append(f"_LLM-filtered ({MODEL_ID}). Total: {len(papers)}._")
    lines.append("")
    lines.append("## Counts by venue")
    lines.append("")
    lines.append("| Venue | Papers |")
    lines.append("|---|---|")
    for v in VENUE_ORDER:
        if by_group.get(v):
            lines.append(f"| {v} | {len(by_group[v])} |")
    lines.append("")

    for v in VENUE_ORDER:
        group = by_group.get(v) or []
        if not group:
            continue
        group.sort(key=lambda p: (-(p.get("year") or 0),
                                   -(p.get("citationCount") or 0)))
        lines.append(f"## {v} ({len(group)})")
        lines.append("")
        for p in group:
            title = p.get("title") or "(untitled)"
            year = p.get("year") or "?"
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:4])
            if p.get("authors") and len(p["authors"]) > 4:
                authors += ", et al."
            url = p.get("url") or ""
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")
            cites = p.get("citationCount") or 0
            lines.append(f"- **{title}** ({year}) — {authors} — "
                         f"_{p.get('venue', '')}_ — cites: {cites}")
            if url:
                lines.append(f"  - {url}")
            if doi:
                lines.append(f"  - doi:{doi}")
            abstract = (p.get("abstract") or "").strip()
            if abstract:
                snippet = re.sub(r"\s+", " ", abstract)
                if len(snippet) > 400:
                    snippet = snippet[:400].rstrip() + "…"
                lines.append(f"  - {snippet}")
            reason = p.get("_llm_reason")
            if reason:
                lines.append(f"  - LLM judge: {reason}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
