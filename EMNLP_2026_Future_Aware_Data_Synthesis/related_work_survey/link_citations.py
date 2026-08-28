"""Post-process synthesis.md to make inline citations clickable.

Parses common academic citation patterns:
  - "Author et al. YYYY"      (e.g. "Wu et al. 2025")
  - "Author et al. (YYYY)"    (e.g. "Bijwadia et al. (2023)")
  - "Author & Author YYYY"    (e.g. "Cherry & Suzuki 2009")
  - "Author, Author & Author YYYY" (e.g. "Li, Liu & Niehues 2024")

For each match, look up the corpus by (first-author-lastname, year). If
*exactly one* paper matches, rewrite to `[Author et al. YYYY](#p_<idx>)`. On
zero or ambiguous matches, leave the text unchanged — better to have the
reader ctrl-F than to mis-link.

The appendix table is rewritten so each row's index column carries an HTML
anchor `<a id="p_<idx>"></a><idx>` for those links to target.

Usage:
    python link_citations.py            # in-place rewrite (synthesis.md)
    python link_citations.py --in foo.md --out bar.md
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_SYNTH = HERE / "synthesis.md"


def fold_to_ascii(s: str) -> str:
    """Strip diacritics: 'Łukasz' -> 'Lukasz', 'Niehues' -> 'Niehues'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def normalize_lastname(name: str) -> str:
    """Lowercase + ASCII-fold + strip non-letters. Last-name comparison key."""
    return re.sub(r"[^a-z]", "", fold_to_ascii(name).lower())


def first_author_lastname(paper: dict) -> str | None:
    authors = paper.get("authors") or []
    if not authors:
        return None
    name = (authors[0].get("name") or "").strip()
    if not name:
        return None
    parts = name.split()
    if not parts:
        return None
    return parts[-1]  # last whitespace token; works for most names


# Patterns ordered specific → general.
# Anchored on a left word boundary; trailing close-paren if a "(YYYY)" form.
PATTERNS = [
    # "Author et al. YYYY"  /  "Author et al., YYYY"  /  "Author et al. (YYYY)"
    re.compile(
        r"\b([A-Z][a-zA-ZÀ-ſ'\-]+)\s+et\s+al\.?(?:,)?\s*(\(?\d{4}\)?)"
    ),
    # "Author1, Author2 & Author3 YYYY" — capture first lastname
    re.compile(
        r"\b([A-Z][a-zA-ZÀ-ſ'\-]+)"
        r"(?:,\s*[A-Z][a-zA-ZÀ-ſ'\-]+)+"
        r"\s*&\s*[A-Z][a-zA-ZÀ-ſ'\-]+"
        r"\s*(\(?\d{4}\)?)"
    ),
    # "Author1 & Author2 YYYY"
    re.compile(
        r"\b([A-Z][a-zA-ZÀ-ſ'\-]+)\s*&\s*[A-Z][a-zA-ZÀ-ſ'\-]+"
        r"\s*(\(?\d{4}\)?)"
    ),
]


def build_index(papers: list[dict]) -> dict[tuple[str, int], list[int]]:
    by_first_year: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, p in enumerate(papers):
        last = first_author_lastname(p)
        year = p.get("year")
        if not last or not year:
            continue
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue
        key = (normalize_lastname(last), year_int)
        by_first_year[key].append(idx)
    return dict(by_first_year)


YEAR_RE = re.compile(r"\d{4}")


def link_citations(text: str, by_first_year: dict[tuple[str, int], list[int]],
                   stats: dict) -> str:
    """Apply all citation patterns. Tracks stats: matched/linked/ambiguous."""
    # Skip the appendix table — its lines look like citations but ARE the index
    # we're linking to. Process body up to the appendix header only.
    appendix_marker = "## Appendix: cluster assignments"
    head, sep, tail = text.partition(appendix_marker)
    if not sep:
        # Fallback: process entire doc (will only no-op on appendix rows that
        # don't look like our patterns anyway).
        head, sep, tail = text, "", ""

    def replace(m: re.Match) -> str:
        full = m.group(0)
        author = m.group(1)
        year_str = m.group(2)
        year_match = YEAR_RE.search(year_str)
        if not year_match:
            return full
        year = int(year_match.group(0))
        if year < 1980 or year > 2035:
            return full
        key = (normalize_lastname(author), year)
        candidates = by_first_year.get(key, [])
        stats["scanned"] = stats.get("scanned", 0) + 1
        if len(candidates) == 1:
            stats["linked"] = stats.get("linked", 0) + 1
            idx = candidates[0]
            return f"[{full}](#p_{idx})"
        if len(candidates) == 0:
            stats["unmatched"] = stats.get("unmatched", 0) + 1
        else:
            stats["ambiguous"] = stats.get("ambiguous", 0) + 1
        return full

    out = head
    for pat in PATTERNS:
        out = pat.sub(replace, out)
    return out + sep + tail


# Match a row of the appendix table:
#  | 0 | Cluster Name | 2018 | Interspeech | Title here |
APPENDIX_ROW_RE = re.compile(
    r"^(\|\s*)(\d+)(\s*\|.*\|)\s*$",
    re.MULTILINE,
)


def anchor_appendix_rows(text: str, paper_count: int) -> str:
    """Add `<a id="p_<idx>"></a>` to each appendix table row's index cell."""
    appendix_marker = "## Appendix: cluster assignments"
    head, sep, tail = text.partition(appendix_marker)
    if not sep:
        return text  # no appendix to anchor

    def repl(m: re.Match) -> str:
        idx = int(m.group(2))
        if idx < 0 or idx >= paper_count:
            return m.group(0)
        return f'{m.group(1)}<a id="p_{idx}"></a>{idx}{m.group(3)}'

    tail_anchored = APPENDIX_ROW_RE.sub(repl, tail)
    return head + sep + tail_anchored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="Path to results.json")
    ap.add_argument("--in", dest="src", default=str(DEFAULT_SYNTH),
                    help="Input markdown")
    ap.add_argument("--out", dest="dst", default=None,
                    help="Output markdown (default: in-place; backup to .bak)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src

    papers = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(f"Loaded {len(papers)} papers from {args.results}", file=sys.stderr)
    text = src.read_text(encoding="utf-8")
    print(f"Read {len(text):,} chars from {src}", file=sys.stderr)

    by_first_year = build_index(papers)
    n_keys = len(by_first_year)
    n_singletons = sum(1 for v in by_first_year.values() if len(v) == 1)
    n_collisions = n_keys - n_singletons
    print(f"Citation index: {n_keys} (lastname, year) keys "
          f"({n_singletons} singletons, {n_collisions} have ≥2 papers)",
          file=sys.stderr)

    stats: dict = {}
    text = link_citations(text, by_first_year, stats)
    print(f"Citation rewrite: {stats.get('scanned', 0)} candidates, "
          f"{stats.get('linked', 0)} linked, "
          f"{stats.get('unmatched', 0)} unmatched, "
          f"{stats.get('ambiguous', 0)} ambiguous", file=sys.stderr)

    text = anchor_appendix_rows(text, len(papers))
    n_anchors = text.count('<a id="p_')
    print(f"Anchored {n_anchors} appendix rows", file=sys.stderr)

    if dst == src:
        bak = src.with_suffix(src.suffix + ".pre-link.bak")
        shutil.copy2(src, bak)
        print(f"Backup → {bak}", file=sys.stderr)

    dst.write_text(text, encoding="utf-8")
    print(f"Wrote {dst} ({len(text):,} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
