"""Adaptive literature-survey crawler.

Backed by the **Semantic Scholar Graph API**. With an API key (`--api-key` or
the `S2_API_KEY` env var) you get 1 req/s sustained, no daily cap published.
Without a key the free tier is shared — expect aggressive 429s.

The script keeps searching until expansions stop producing in-scope papers.
Two expansion mechanisms run each round:

  (A) Keyword expansion — mine bigrams/trigrams from titles+abstracts of papers
      newly added in the previous round, score them, and turn the top-K into
      new search queries.
  (B) Citation-graph expansion — for the top-N highest-cited new papers, pull
      `/paper/{id}/references` and `/paper/{id}/citations`.

After each round we count `new_in_scope_added`. If two consecutive rounds
fall below `--stop-threshold`, the loop terminates ("no hope" condition).

All topic-specific values (seed queries, venues, topic regex, …) live in
`config.py` in the same directory as this script.

Outputs:
  - results.json   — full S2 metadata for every accepted paper.
  - results.md     — venue-grouped, year-sorted browsable summary.
  - survey.log     — round-by-round telemetry.
  - .s2_cache.json — persistent URL→response cache; re-runs are nearly free.

Usage:
    S2_API_KEY=$(cat ~/.keys/semantic_scholar) python survey.py \\
        --max-rounds 8 --max-requests 2000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import urllib.request
import urllib.error

from config import (
    ANCHOR_TERMS,
    QUERY_TASK_SUFFIXES,
    S2_VENUE_FILTER,
    SEED_QUERIES,
    TASK_RE,
    TOPIC_RE,
    TOPIC_SHORT,
    VENUE_GROUPS,
    VENUE_ORDER,
    YEAR_FROM,
)


# ---------------------------------------------------------------------------
# Constants (generic; topic-specific values come from config)
# ---------------------------------------------------------------------------

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL = f"{S2_BASE}/paper/search"

FIELDS = ("paperId,title,abstract,authors,year,venue,publicationVenue,url,"
          "externalIds,citationCount,referenceCount")

S2_VENUE_FILTER_STR = ",".join(S2_VENUE_FILTER)

STOPWORDS: set[str] = set("""
a an the of for and or but with without on in to from by at as is are was were be been being
this that these those it its their our your we they he she him her them his hers
which who whom whose what when where why how
about into onto upon over under again further
not no nor very can could should would may might must shall will just only also same other another
than then so such both each few more most some any all every
do does did done doing have has had having get got make made making used use using uses
new novel based using approach approaches method methods system systems model models
paper papers work works study studies result results approach show shows showed propose proposed
proposes proposing introduce introduces presented present presents experiments experiment
performance improve improving improved improvement neural network networks deep
dataset datasets benchmark benchmarks training train trained training-free
state-of-the-art sota across via towards toward
""".split())


# ---------------------------------------------------------------------------
# HTTP / API helpers
# ---------------------------------------------------------------------------


class Cache:
    """Persistent URL→response cache so re-runs don't re-hit the API.

    Single JSON file at `path`. Atomic write via .tmp+rename. Only successful
    (200) responses are cached."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.data: dict[str, dict] = {}
        self.dirty_count = 0
        if path is not None and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, url: str) -> dict | None:
        return self.data.get(url)

    def put(self, url: str, value: dict) -> None:
        if self.path is None:
            return
        self.data[url] = value
        self.dirty_count += 1
        if self.dirty_count >= 25:
            self.flush()

    def flush(self) -> None:
        if self.path is None or self.dirty_count == 0:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)
        self.dirty_count = 0


class S2Client:
    """Semantic Scholar Graph API client with caching, budget enforcement,
    Retry-After honoring, and adaptive base-pace bumping on sustained 429s."""

    def __init__(self, api_key: str | None, log, cache: Cache,
                 max_requests: int) -> None:
        self.api_key = api_key
        self.log = log
        self.cache = cache
        self.max_requests = max_requests
        self.requests_made = 0
        self.headers = {"User-Agent": "literature-survey-script/3.0"}
        if api_key:
            self.headers["x-api-key"] = api_key
        # S2 with key: 1 req/sec is the documented sustained rate.
        # Without key: free tier shares ~100 req per 5 min across the IP.
        self.base_pace = 1.0 if api_key else 3.5
        self.pace = self.base_pace
        self.consecutive_429s = 0

    def _fetch(self, url: str) -> dict:
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        if self.requests_made >= self.max_requests:
            raise RuntimeError(
                f"hit --max-requests cap ({self.max_requests}); aborting to "
                f"protect S2. Re-run with a larger cap if you need more.")

        for attempt in range(6):
            self.requests_made += 1
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.consecutive_429s = 0
                self.cache.put(url, data)
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {}
                if e.code == 429:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    try:
                        wait = float(ra) if ra else self.pace * (attempt + 2)
                    except ValueError:
                        wait = self.pace * (attempt + 2)
                    wait = max(wait, 5.0)
                    self.consecutive_429s += 1
                    if self.consecutive_429s >= 3 and self.base_pace < 15.0:
                        self.base_pace = min(self.base_pace * 1.5, 15.0)
                        self.pace = self.base_pace
                        self.log(f"    sustained 429s — base pace bumped to "
                                 f"{self.base_pace:.1f}s")
                        self.consecutive_429s = 0
                    self.log(f"    HTTP 429; sleeping {wait:.1f}s "
                             f"(Retry-After={ra})")
                    time.sleep(wait)
                    continue
                if e.code in (500, 502, 503, 504):
                    wait = self.pace * (attempt + 2)
                    self.log(f"    HTTP {e.code}; sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
            except urllib.error.URLError as e:
                wait = self.pace * (attempt + 2)
                self.log(f"    URLError: {e}; sleeping {wait:.1f}s")
                time.sleep(wait)
        self.log(f"    giving up: {url}")
        return {}

    def search(self, query: str, year_from: int | None,
               limit: int) -> list[dict]:
        params: dict[str, str] = {
            "query": query,
            "fields": FIELDS,
            "venue": S2_VENUE_FILTER_STR,
            "limit": str(min(100, limit)),
        }
        if year_from is not None:
            params["year"] = f"{year_from}-"
        out: list[dict] = []
        offset = 0
        while True:
            params["offset"] = str(offset)
            data = self._fetch(f"{S2_SEARCH_URL}?{urlencode(params)}")
            page = data.get("data") or []
            out.extend(page)
            if len(page) < int(params["limit"]) or len(out) >= limit:
                break
            offset += len(page)
            if offset >= 1000:  # S2 caps offset+limit at 1000 on search
                break
            time.sleep(self.pace)
        return out

    def references(self, paper: dict, limit: int = 200) -> list[dict]:
        pid = paper.get("paperId") if isinstance(paper, dict) else paper
        if not pid:
            return []
        url = (f"{S2_BASE}/paper/{pid}/references"
               f"?fields={FIELDS}&limit={limit}")
        data = self._fetch(url)
        return [item.get("citedPaper") or {} for item in (data.get("data") or [])]

    def citations(self, paper: dict, limit: int = 200) -> list[dict]:
        pid = paper.get("paperId") if isinstance(paper, dict) else paper
        if not pid:
            return []
        url = (f"{S2_BASE}/paper/{pid}/citations"
               f"?fields={FIELDS}&limit={limit}")
        data = self._fetch(url)
        return [item.get("citingPaper") or {} for item in (data.get("data") or [])]


# ---------------------------------------------------------------------------
# Classification & filtering
# ---------------------------------------------------------------------------


def classify_venue(paper: dict) -> str | None:
    venue = (paper.get("venue") or "").lower()
    pv = paper.get("publicationVenue") or {}
    if isinstance(pv, dict):
        venue = venue or (pv.get("name") or "").lower()
    if not venue:
        return None
    for group, needles in VENUE_GROUPS.items():
        for needle in needles:
            if needle in venue:
                return group
    return None


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (title or "").lower())).strip()


def is_topical(paper: dict) -> bool:
    """Topic gate: a paper must match BOTH TOPIC_RE and TASK_RE on its
    title+abstract. Without this, citation-graph expansion drags in unrelated
    work that the seed papers happened to cite."""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).strip()
    if not text:
        return False
    return bool(TOPIC_RE.search(text) and TASK_RE.search(text))


# ---------------------------------------------------------------------------
# Keyword expansion (n-gram mining)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9-]+", (text or "").lower())


def mine_phrases(papers: list[dict], existing_query_text: str, top_k: int) -> list[str]:
    """Return up to top_k bigrams/trigrams worth turning into new queries."""
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    anchor_hits: Counter = Counter()

    for p in papers:
        text = ((p.get("title") or "") + " . " + (p.get("abstract") or "")).lower()
        toks = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
        seen_in_doc: set[str] = set()
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            if bg not in seen_in_doc:
                bigrams[bg] += 1
                seen_in_doc.add(bg)
        for i in range(len(toks) - 2):
            tg = f"{toks[i]} {toks[i+1]} {toks[i+2]}"
            if tg not in seen_in_doc:
                trigrams[tg] += 1
                seen_in_doc.add(tg)
        if any(a in text for a in ANCHOR_TERMS):
            for phrase in seen_in_doc:
                anchor_hits[phrase] += 1

    candidates: list[tuple[float, str]] = []
    existing_lc = existing_query_text.lower()
    for phrase, count in (bigrams + trigrams).items():
        if count < 2:
            continue
        if phrase in existing_lc:
            continue
        words = phrase.split()
        if any(w in STOPWORDS for w in words):
            continue
        if not any(w.isalpha() and len(w) > 3 for w in words):
            continue
        score = count + 0.5 * anchor_hits.get(phrase, 0)
        if len(words) == 3:
            score *= 1.2
        candidates.append((score, phrase))

    candidates.sort(reverse=True)
    return [phrase for _, phrase in candidates[:top_k]]


def queries_from_phrases(phrases: list[str]) -> list[str]:
    out: list[str] = []
    suffixes = QUERY_TASK_SUFFIXES or [""]
    for ph in phrases:
        for suf in suffixes:
            out.append(f"{ph} {suf}".strip())
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year-from", type=int, default=YEAR_FROM)
    ap.add_argument("--api-key", default=os.environ.get("S2_API_KEY"),
                    help="Semantic Scholar API key. Falls back to S2_API_KEY env var.")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--stop-threshold", type=int, default=3,
                    help="Two consecutive rounds with fewer than this many new "
                         "in-scope papers triggers stop.")
    ap.add_argument("--keyword-topk", type=int, default=8)
    ap.add_argument("--citation-topn", type=int, default=30)
    ap.add_argument("--per-query-limit", type=int, default=200)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    ap.add_argument("--max-requests", type=int, default=2000,
                    help="Hard cap on live S2 requests per run (cache hits "
                         "don't count). Aborts if exceeded.")
    ap.add_argument("--cache-file", default=None,
                    help="Path to JSON cache of S2 responses. Default: "
                         "<out-dir>/.s2_cache.json. Pass an empty string "
                         "to disable caching.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "survey.log"
    log_fh = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)
        log_fh.write(msg + "\n")
        log_fh.flush()

    if args.cache_file is None:
        cache_path: Path | None = out_dir / ".s2_cache.json"
    elif args.cache_file == "":
        cache_path = None
    else:
        cache_path = Path(args.cache_file)
    cache = Cache(cache_path)
    log(f"Cache: {cache_path or 'disabled'} "
        f"(loaded {len(cache.data)} cached responses)")
    if not args.api_key:
        log("WARN: no --api-key supplied; using S2 free tier (shared, slow). "
            "Pass --api-key or set S2_API_KEY env var for 1 req/s.")

    client = S2Client(args.api_key, log=log, cache=cache,
                      max_requests=args.max_requests)

    seen: dict[str, dict] = {}
    seen_titles: set[str] = set()
    queries_used: set[str] = set()
    round_papers: dict[int, list[dict]] = {}
    consecutive_low = 0

    def absorb(paper: dict, round_idx: int, source: str) -> bool:
        if not paper:
            return False
        pid = paper.get("paperId")
        if not pid or pid in seen:
            return False
        t_key = normalize_title(paper.get("title") or "")
        if t_key and t_key in seen_titles:
            return False
        if (paper.get("year") or 0) < args.year_from:
            return False
        group = classify_venue(paper)
        if not group:
            return False
        if not is_topical(paper):
            return False
        paper["_venue_group"] = group
        paper["_added_via"] = source
        paper["_added_in_round"] = round_idx
        seen[pid] = paper
        if t_key:
            seen_titles.add(t_key)
        return True

    try:
        # ----- round 0: seed queries -----
        log(f"=== ROUND 0: seed queries ({len(SEED_QUERIES)}) ===")
        round_papers[0] = []
        for i, q in enumerate(SEED_QUERIES, 1):
            log(f"[seed {i}/{len(SEED_QUERIES)}] {q}")
            queries_used.add(q.lower())
            try:
                results = client.search(q, args.year_from, args.per_query_limit)
            except urllib.error.URLError as e:
                log(f"  network ERROR: {e}")
                results = []
            added = 0
            for paper in results:
                if absorb(paper, 0, f"query:{q}"):
                    round_papers[0].append(paper)
                    added += 1
            log(f"  -> returned {len(results)}, added {added}")
            time.sleep(client.pace)
        log(f"=== ROUND 0 complete: {len(round_papers[0])} new in-scope papers ===\n")

        # ----- adaptive rounds -----
        for r in range(1, args.max_rounds + 1):
            prev = round_papers.get(r - 1, [])
            if not prev:
                log(f"=== ROUND {r}: no seeds from round {r-1}, stopping ===")
                break

            log(f"=== ROUND {r}: keyword + citation expansion ===")
            round_papers[r] = []

            # (A) Keyword expansion from prev round's accepted papers
            existing_text = " | ".join(queries_used)
            phrases = mine_phrases(prev, existing_text, args.keyword_topk)
            log(f"  mined {len(phrases)} candidate phrases: {phrases}")
            new_queries = [q for q in queries_from_phrases(phrases)
                           if q.lower() not in queries_used]
            log(f"  -> {len(new_queries)} new queries to run")
            for i, q in enumerate(new_queries, 1):
                log(f"  [kw {i}/{len(new_queries)}] {q}")
                queries_used.add(q.lower())
                try:
                    results = client.search(q, args.year_from, args.per_query_limit)
                except urllib.error.URLError as e:
                    log(f"    network ERROR: {e}")
                    results = []
                added = 0
                for paper in results:
                    if absorb(paper, r, f"query:{q}"):
                        round_papers[r].append(paper)
                        added += 1
                log(f"    -> returned {len(results)}, added {added}")
                time.sleep(client.pace)

            # (B) Citation-graph expansion from top prev-round papers
            prev_sorted = sorted(prev, key=lambda p: -(p.get("citationCount") or 0))
            seeds = prev_sorted[:args.citation_topn]
            log(f"  citation-expanding {len(seeds)} top papers from round {r-1}")
            for j, seed in enumerate(seeds, 1):
                pid = seed.get("paperId")
                title = (seed.get("title") or "")[:80]
                log(f"  [cite {j}/{len(seeds)}] {title!r} ({pid})")
                try:
                    refs = client.references(seed)
                    cites = client.citations(seed)
                except urllib.error.URLError as e:
                    log(f"    network ERROR: {e}")
                    refs, cites = [], []
                added_r = added_c = 0
                for paper in refs:
                    if absorb(paper, r, f"ref_of:{pid}"):
                        round_papers[r].append(paper)
                        added_r += 1
                for paper in cites:
                    if absorb(paper, r, f"cite_of:{pid}"):
                        round_papers[r].append(paper)
                        added_c += 1
                log(f"    -> refs:{len(refs)} (+{added_r}), cites:{len(cites)} (+{added_c})")
                time.sleep(client.pace)

            added_this_round = len(round_papers[r])
            log(f"=== ROUND {r} complete: {added_this_round} new in-scope papers "
                f"(total: {len(seen)}) ===\n")

            if added_this_round < args.stop_threshold:
                consecutive_low += 1
            else:
                consecutive_low = 0
            if consecutive_low >= 2:
                log(f"STOP: 2 consecutive rounds < threshold={args.stop_threshold}")
                break
    except RuntimeError as e:
        log(f"!! ABORTED: {e}")
    finally:
        cache.flush()
        log(f"S2 live requests this run: {client.requests_made} / "
            f"{args.max_requests} budget. Cached responses on disk: "
            f"{len(cache.data)}.")

        papers = list(seen.values())
        papers.sort(key=lambda p: (p.get("_venue_group", ""),
                                    -(p.get("year") or 0),
                                    -(p.get("citationCount") or 0)))
        json_path = out_dir / "results.json"
        json_path.write_text(json.dumps(papers, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        log(f"Wrote {len(papers)} papers to {json_path}")

        md_path = out_dir / "results.md"
        write_markdown(papers, md_path, args.year_from, round_papers)
        log(f"Wrote summary to {md_path}")
        log_fh.close()


def write_markdown(papers: list[dict], path: Path, year_from: int,
                   round_papers: dict[int, list[dict]]) -> None:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        by_group[p["_venue_group"]].append(p)

    lines: list[str] = []
    lines.append(f"# Literature survey: {TOPIC_SHORT}")
    lines.append("")
    lines.append(f"_Source: Semantic Scholar Graph API. Year ≥ {year_from}. "
                 f"Total papers: {len(papers)}._")
    lines.append("")

    lines.append("## Round-by-round adds")
    lines.append("")
    lines.append("| Round | New in-scope papers |")
    lines.append("|---|---|")
    for r in sorted(round_papers):
        lines.append(f"| {r} | {len(round_papers[r])} |")
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
            arxiv = ext.get("ArXiv")
            doi = ext.get("DOI")
            cites = p.get("citationCount") or 0
            ident_bits: list[str] = []
            if arxiv:
                ident_bits.append(f"arXiv:{arxiv}")
            if doi:
                ident_bits.append(f"doi:{doi}")
            ident = " · ".join(ident_bits)
            lines.append(f"- **{title}** ({year}) — {authors} — _{p.get('venue', '')}_ "
                         f"— cites: {cites}")
            if url:
                lines.append(f"  - {url}")
            if ident:
                lines.append(f"  - {ident}")
            abstract = (p.get("abstract") or "").strip()
            if abstract:
                snippet = re.sub(r"\s+", " ", abstract)
                if len(snippet) > 400:
                    snippet = snippet[:400].rstrip() + "…"
                lines.append(f"  - {snippet}")
            lines.append(f"  - via: `{p.get('_added_via', '')}` "
                         f"(round {p.get('_added_in_round', '?')})")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
