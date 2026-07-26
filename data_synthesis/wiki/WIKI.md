# WIKI.md — Schema & Conventions for this LLM Wiki

This file is the **style guide**. Any agent operating on `data_synthesis/wiki/` MUST follow it.
Pattern: Karpathy "LLM Wiki" — the LLM owns and maintains these pages; this is NOT RAG.

## Layers

1. `raw/` — immutable sources, read-only. Never edit. `../codes/` is also a raw source, referenced by path.
2. `concepts/ entities/ experiments/ comparisons/` — LLM-owned pages, cross-linked with `[[wikilinks]]`.
3. `WIKI.md` (this file), `index.md` (catalog), `log.md` (event log).

## Page types & where they go

- `concepts/<name>.md` — a method/idea (e.g. `future-sampling`, `consensus-decoding`, `min-p`).
- `entities/<name>.md` — a dataset/tool/model (e.g. `gigaspeech`, `metricx`, `mfa`, `east`).
- `experiments/<YYYY-MM-name>.md` — a concrete run + its findings.
- `comparisons/<a>-vs-<b>.md` — head-to-head of two approaches/metrics.

Filenames are kebab-case, NO type prefix (the dir is the type).

## Page format

Every page starts with YAML frontmatter, then a body using `[[wikilinks]]`:

```markdown
---
title: Human Readable Title
type: concept            # concept | entity | experiment | comparison
tags: [synthesis, policy]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_core.py
  - raw/papers/some-paper.md
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Human Readable Title

Body. Link other pages with [[consensus-decoding]]. Cite code as `path:line`.

## Sources
- [[gigaspeech]]
```

- Wikilink target = filename without extension, e.g. `[[future-sampling]]`. If a name collides
  across dirs, disambiguate as `[[concepts/future-sampling]]`.
- `created`/`updated` are dates passed in by the user (no system clock in scripts); update `updated` on every edit.

## Operations (see the `wiki` skill for the full workflow)

- **Ingest** (7 steps): read raw → extract concepts/entities/data → write summary page →
  update 10–15 related pages with cross-links → flag contradictions → update `index.md` →
  append `log.md`.
- **Query**: search pages → cited answer → optionally file answer back as a new page;
  backfill via web search when the wiki can't answer, saving findings as permanent pages.
- **Lint**: find contradictions, stale claims superseded by newer sources, and orphan pages
  (no inbound `[[links]]`). Report + propose fixes.

## index.md format

Content-oriented catalog, grouped by category. Each line: `- [[name]] — one-line summary`.
Sections: `## Concepts`, `## Entities`, `## Experiments`, `## Comparisons`.

## log.md format

Append-only. One entry per operation, parseable header:

```
## [YYYY-MM-DD] ingest | <source title>
- added: [[page]]
- updated: [[page]], [[page]]
- contradictions: <none | description>
```
Prefixes: `ingest`, `query`, `lint`.

## Relationship to the memory system

`~/.claude/projects/.../memory/` (also `[[wikilink]]`-style) holds cross-session "how I should
work" facts. This wiki holds the **project's knowledge itself**. Do not duplicate: if a fact is
"a working preference," it belongs in memory; if it's "knowledge about the project/domain," it
belongs here. Cross-reference by name where useful.
