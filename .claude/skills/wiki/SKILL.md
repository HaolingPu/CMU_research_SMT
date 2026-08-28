---
name: wiki
description: Operate the data_synthesis LLM Wiki (Karpathy pattern). Use when the user says ingest/process a source, asks a question to "the wiki", wants to file/save findings as wiki pages, or asks to lint/health-check the wiki. The wiki lives at data_synthesis/wiki/ and is governed by data_synthesis/wiki/WIKI.md.
---

# Wiki Skill

You maintain the LLM Wiki at `data_synthesis/wiki/`. ALWAYS read `data_synthesis/wiki/WIKI.md`
first — it is the authoritative style guide for filenames, page format, and link conventions.

Dates: there is no system clock available to you in scripts. When a step needs `created`/`updated`
or a log date, use the current date the harness provides in context; if unsure, ask the user.

## Choose the operation

- User drops/points at a source and says process/ingest → **Ingest**.
- User asks a question about the project/domain → **Query**.
- User says lint / health-check / find stale-or-orphan pages → **Lint**.

## Ingest (7 steps)

1. Read the raw source (a file in `raw/`, or code under `../codes/` the user points to).
2. Extract key concepts, entities, and data points. Discuss the takeaways with the user briefly.
3. Write a summary page in the correct dir (`concepts/`, `entities/`, `experiments/`, or
   `comparisons/`) with full frontmatter (title, type, tags, sources, created, updated).
4. Update 10–15 relevant existing pages, adding `[[wikilinks]]` both ways. If fewer than 15
   relevant pages exist, update all that are relevant and say so.
5. Flag contradictions where the new source conflicts with an existing claim. Surface them to
   the user; do not silently overwrite — note both claims and which source is newer.
6. Update `index.md`: add the new page under its category with a one-line summary.
7. Append an entry to `log.md` using the `## [DATE] ingest | <title>` format.

## Query

1. Search the wiki pages (grep titles, tags, bodies) for relevant material.
2. Synthesize an answer that **cites the pages** it came from (`[[page]]`).
3. If the wiki cannot answer: do a web search, answer, then **backfill** — save the findings as
   a new permanent page and update `index.md` + `log.md` (so research compounds).
4. If the answer is broadly useful, offer to file it back as a new page even when the wiki could
   partially answer.

## Lint

1. **Contradictions:** scan for pages making conflicting claims; report pairs.
2. **Stale claims:** find claims superseded by a newer source (compare `updated`/`sources`).
3. **Orphans:** find pages with no inbound `[[link]]` from any other page or from `index.md`.
4. Produce a report and propose concrete fixes (links to add, pages to merge/retire). Apply
   fixes only after the user confirms. Append a `## [DATE] lint | N issues` entry to `log.md`.

## Invariants

- Never edit anything under `raw/`.
- Every new/edited page keeps valid frontmatter and at least one `[[link]]` (no orphans).
- Keep `index.md` in sync with the actual page files.
