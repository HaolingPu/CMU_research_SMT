---
name: literature-survey
description: Bootstrap an adaptive literature-survey pipeline (Semantic Scholar crawler → Sonnet 4.6 filter → Opus 4.7 cluster+synthesis → citation linker) for a topic the user specifies. Generates a self-contained directory with parameterized scripts. Use when the user wants to start a new literature survey, "extend"/"clone"/"template" an existing survey, or set up a survey for another research field.
---

# Literature survey bootstrapper

Generates a new literature-survey project in a directory the user picks. The pipeline is four scripts; everything topic-specific is centralized in a single `config.py` you write from the user's inputs.

The template files live in `~/.claude/skills/literature-survey/template/`. Read them when you need to know the exact structure of `config.py` or to copy the generic scripts into the user's directory.

## Workflow

### 1. Gather inputs

Ask the user — in ONE consolidated message, not a back-and-forth interview — for:

- **Target directory** (absolute path; should not exist yet, or be empty)
- **Topic** — 2-4 sentences. What tasks, what kinds of papers count as on-topic, what kinds explicitly DON'T?
- **Venues** — conferences/journals to crawl, ideally grouped by family (e.g. "speech: ICASSP, Interspeech; NLP: ACL, EMNLP; ML: NeurIPS, ICLR")
- **Year range** — earliest year to include

Skip whichever the user already gave in their initial message.

### 2. Compose `config.py`

Read `~/.claude/skills/literature-survey/template/config.py` first. It has the structure with example values from a sample survey on rare-words-in-ASR/MT — use it as a reference for what each constant looks like, then write your own values for the user's topic. The constants you need to fill:

- **`TOPIC_SHORT`** — markdown-title-friendly short form of the topic
- **`YEAR_FROM`** — from input
- **`S2_VENUE_FILTER`** — list of venue strings for S2's server-side prefilter (loose union; aliases OK)
- **`VENUE_GROUPS`** — dict mapping display group → list of lowercased substring matchers. Add liberal aliases (full names, abbreviations, "proc." prefixes, IEEE prefixes) so off-spelling venues from S2 aren't dropped.
- **`VENUE_DESCRIPTION`** — one human-readable sentence like "speech (ICASSP, Interspeech), NLP (ACL, EMNLP)…"
- **`SEED_QUERIES`** — 15-30 search queries spanning the topic. Include alternative phrasings, sub-areas, and adjacent terminology. The crawler mines new queries from each round so don't worry about exhaustiveness; worry about breadth.
- **`QUERY_TASK_SUFFIXES`** — 1-3 short task tags appended to mined phrases when generating new queries. Typically the dominant task names (e.g. ["speech recognition", "machine translation"]).
- **`ANCHOR_TERMS`** — 5-15 short lowercased substrings; the "screams on-topic" vocabulary
- **`TOPIC_RE`** and **`TASK_RE`** — TOPIC matches the *problem* keywords; TASK matches the *task surface*. BOTH must match for a paper to pass the citation-graph filter. A generic NER paper should fail TASK_RE; a generic NMT paper should fail TOPIC_RE.
- **`TOPIC_RUBRIC`** — Sonnet 4.6 judge prompt. Structure: role priming → topic in **bold** → two-condition rubric → 5-10 RELATED examples → 5-10 NOT RELATED examples (focus on adjacent-but-out-of-scope kinds the regex would mistakenly admit) → "Be strict" line → output format reminder.
- **`TOPIC_HEADER`** — Opus 4.7 synthesis system prompt. Include topic statement, year range + venue families, and a bullet list of 8-12 expected sub-areas (helps clustering quality).

### 3. Sign-off

Show the user the proposed `config.py` content (or at minimum the regexes, rubric, header, and seed queries — those are the highest-leverage pieces). Ask for adjustments before writing files. The TOPIC_RE and TOPIC_RUBRIC are load-bearing for filter quality — wrong values silently skew the corpus.

### 4. Write files

Once signed off, populate the target directory:

1. Read each template file from `~/.claude/skills/literature-survey/template/`:
   - `survey.py`
   - `filter_results_llm.py`
   - `synthesize_field.py`
   - `link_citations.py`
   - `smoke_test_citations.py`
   - `CLAUDE.md`
2. Write each one verbatim into the target directory (Read then Write — don't shell out to `cp`).
3. Write the `config.py` you composed in step 2 into the same directory.

### 5. Tell the user how to run

Print the four-stage flow with absolute commands:

```
S2_API_KEY=$(cat ~/.keys/semantic_scholar) python survey.py
ANTHROPIC_API_KEY=$(cat ~/.keys/claude) PYTHONPATH=~/.local/lib/python3.9/site-packages python filter_results_llm.py
ANTHROPIC_API_KEY=$(cat ~/.keys/claude) PYTHONPATH=~/.local/lib/python3.9/site-packages python synthesize_field.py
python link_citations.py
```

Mention that `python smoke_test_citations.py` (after Stage 1 of `synthesize_field.py` populates `synthesis.cache.json`) is a cheap (~$0.10) sanity check before the full Stage 2/3 finish.

## Don't

- **Don't run the pipeline scripts during bootstrap.** They cost real money (S2 budget, Anthropic API) and `synthesize_field.py` alone takes ~15 minutes plus dollars in API calls.
- **Don't invent venues outside the user's list.** Conference-by-field expertise belongs to them.
- **Don't skip sign-off on the regexes and rubric.** They're the highest-leverage knobs and the easiest to get subtly wrong.
- **Don't hardcode the user's API key paths into the templates.** The templates ship with `~/.keys/...` placeholders for portability.
