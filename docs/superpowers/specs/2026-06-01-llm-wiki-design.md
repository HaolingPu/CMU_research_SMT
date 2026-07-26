# LLM Wiki for the data_synthesis Project — Design Spec

**Date:** 2026-06-01
**Status:** Approved design, pending implementation
**Pattern:** Karpathy "LLM Wiki" (LLM owns & maintains a structured, interlinked markdown knowledge base — not RAG)

## Goal

Build a persistent, LLM-owned knowledge base for the `data_synthesis/` research project
(future-aware data synthesis for speech translation, EMNLP 2026). The LLM ingests sources,
maintains interlinked markdown pages, answers queries against them, and lints for health.
Viewed by the human via VS Code **Foam** over Remote-SSH (no sync/mount needed).

## Three-Layer Architecture

1. **Raw sources** (`wiki/raw/`, read-only — LLM never edits): papers, dropped experiment
   snapshots. The codebase `data_synthesis/codes/` is also a raw source but is **referenced
   by path, not copied** (single source of truth).
2. **The Wiki** (`wiki/concepts|entities|experiments|comparisons/`): markdown pages the LLM
   fully owns, cross-linked with `[[wikilinks]]`.
3. **The Schema** (`wiki/WIKI.md`): style guide telling the LLM the conventions and the
   ingest/query/lint workflows.

## Directory Layout

```
data_synthesis/wiki/
├── WIKI.md            # Schema: conventions, page format, ingest/query/lint rules
├── index.md           # Master catalog, by category, one-line summary per page
├── log.md             # Append-only timestamped event log
├── raw/               # Immutable sources, read-only
│   ├── papers/        #   papers (PDF or extracted text/notes)
│   ├── experiments/   #   dropped experiment snapshots (config + results)
│   └── README.md      #   note: ../codes is also a raw source, referenced not copied
├── concepts/          # concept pages (future-sampling, consensus-decoding, min-p, ...)
├── entities/          # entity pages (gigaspeech, yodas, metricx, MFA, EAST, ...)
├── experiments/       # experiment finding pages (2026-05-minp-sweep, ...)
└── comparisons/       # comparison pages (la-n-vs-wait-k, comet-vs-bleu, ...)
```

## Page Format

Every wiki page carries YAML frontmatter and uses `[[wikilinks]]` (Foam/Obsidian compatible):

```markdown
---
title: Future Sampling
type: concept            # concept | entity | experiment | comparison
tags: [synthesis, policy]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_core.py
  - raw/papers/some-paper.md
created: 2026-06-01
updated: 2026-06-01
---

# Future Sampling
Body, cross-linked via [[consensus-decoding]] ...

## Sources
- [[gigaspeech]]
```

With subdirs, filenames carry **no type prefix** (the dir is the type): `concepts/future-sampling.md`,
`entities/gigaspeech.md`. Wikilink target = filename without extension, e.g. `[[future-sampling]]`
(or disambiguated `[[concepts/future-sampling]]` if a name collides across dirs). Filenames are
kebab-case within each category dir.

## Core Operations (packaged as a Claude Code skill)

Skill location: `/data/user_data/haolingp/.claude/skills/wiki/SKILL.md` (alongside `grill-me`,
discoverable from the project working dir). The skill references `WIKI.md` for conventions.

### Ingest (7 steps)
1. Read the raw source.
2. Extract key concepts, entities, data points.
3. Write a summary page (with metadata + tags) in the right category dir.
4. Update 10–15 relevant existing pages, adding cross-links.
5. Flag contradictions where new data conflicts with previous claims.
6. Update `index.md` (master catalog).
7. Append a timestamped entry to `log.md`.

### Query
- Search relevant wiki pages, synthesize a cited answer.
- Optionally file a good answer back as a new page (research compounds).
- **Backfill:** if the wiki can't answer, web-search and save findings as permanent pages.

### Lint
- Health check: find contradictions between pages, stale claims superseded by newer sources,
  orphan pages with no inbound links. Report and propose fixes.

## index.md / log.md Conventions

- **index.md**: content-oriented catalog. Each page = link + one-line summary + optional
  metadata (date, source count). Organized by category (Concepts / Entities / Experiments /
  Comparisons).
- **log.md**: append-only, parseable prefixes, e.g. `## [2026-06-01] ingest | <source title>`,
  `## [2026-06-01] lint | N issues`, `## [2026-06-01] query | <question>`.

## Relationship to Existing Tooling

- **Memory system** (`~/.claude/.../memory/`, also `[[wikilinks]]`-style) stays separate:
  memory = cross-session "how I should work" facts; wiki = the project's growing knowledge
  itself. `WIKI.md` documents this split to avoid duplication.
- **Seed pages for v1**: several existing memories are ready-made comparison/concept seeds —
  e.g. LA-N vs Wait-k, COMET-vs-BLEU ranking, EAST infer-prompt, consensus ref-free vs
  ref-based. Use these as the first pages to validate the structure.

## v1 Scope

Full system: directory skeleton + `WIKI.md` schema + `index.md`/`log.md` + the `wiki` skill
implementing ingest/query/lint + a handful of seed pages drawn from existing memories.

## Out of Scope (YAGNI)

- No RAG/embeddings/vector DB — the whole point is LLM-owned markdown.
- No automated daemon/cron ingestion — ingest is user-triggered.
- No raw slurm-log scraping in v1 (deferred; sources are papers, experiment snapshots, code).
- No separate git repo or Obsidian-Sync — Foam over Remote-SSH covers viewing.
