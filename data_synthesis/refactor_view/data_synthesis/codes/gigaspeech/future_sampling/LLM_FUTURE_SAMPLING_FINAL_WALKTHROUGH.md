# `llm_future_sampling_final.py` Detailed Walkthrough

This document explains the current implementation of:

- `data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_final.py`

It is written to help you map:

- what the pipeline is trying to do
- what each major function actually does
- how a verbose log line maps to code behavior
- why certain chunks become `READ` vs `WRITE`

The line references below point to the current file version after the recent patches.

## 1. High-Level Goal

The script builds a **simultaneous translation training target** (chunk-level `READ/WRITE` actions) from:

- an English source trajectory (`src_trajectory`, chunked observations)
- a base LLM (future source sampling)
- an instruct LLM (translation + judging)
- an alignment model (truncate translation to observed-only safe prefix)

At each chunk, the pipeline decides:

- `READ`: do not emit target text yet
- `WRITE`: emit an incremental Chinese delta

## 2. Current Architecture (What Runs Where)

The system is dual-model:

- Base model (Qwen base) runs in-process via `vllm.LLM.generate(...)`
- Instruct model runs as a separate `vllm serve` process and is called through OpenAI-compatible HTTP API
- Alignment model (`awesome-align`) runs on CPU

Important distinction:

- The instruct side now uses **manual chat-template prompt text + completion generation style** (mentor-style continuation prompt construction)
- The HTTP call is still `client.completions.create(...)` because the instruct model lives in a separate server process, not as a local `LLM` object in this Python process

## 3. Main Data Flow Per Chunk

The main loop is in `process_one_utterance(...)` at `llm_future_sampling_final.py:858`.

Per chunk:

1. Update `accumulated_source`
2. If empty chunk and not last: `READ`
3. If final chunk: force final completion (`translate_final`) and `WRITE` remainder if any
4. If too few observed words: `READ`
5. Else run the future-sampling pipeline:
   - Step 1: sample future source continuations
   - Step 2: translate each `(observed + future_i)`
   - Step 3: align + truncate each translation to observed-safe prefix
   - Step 4: LLM judge scores each valid candidate
   - Step 5: consensus + direction check + choose best delta
   - Final gate: `word_head_guard`

## 4. Key State Variables You See in Logs

These appear in verbose logs and are central to understanding behavior.

- `accumulated_source`: observed English prefix so far (`process_one_utterance`, `:883-:891`)
- `committed_norm`: already committed Chinese prefix (`:882`)
- `futures`: sampled English continuations (`:946-:956`)
- `all_translations`: instruct-model translations of `(observed + future_i)` (`:964-:973`)
- `safe_prefix`: alignment-truncated Chinese prefix (observed-only safe part) (`:986-:1005`)
- `delta`: `safe_prefix[len(committed_norm):]` if monotonic (`:991-:995`)
- `valid_candidates`: candidates whose `delta` length passes `min_commit_chars` (`:1018`)
- `qualified`: valid candidates with judge score >= `score_threshold` (`:1060-:1063`)
- `qualified_ratio`: `len(qualified) / len(valid_candidates)` (`:1064`)

## 5. Text Utilities and Why They Matter

See `llm_future_sampling_final.py:125-:246`.

### `normalize_zh(...)` (`:125`)

- Removes whitespace and normalizes Unicode
- Used to keep committed text stable when comparing prefixes and appending deltas

### `clean_llm_output(...)` (`:131`)

- Removes `<think>...</think>` and trims quotes
- Prevents judge/translation outputs from containing control-style wrappers

### `_ends_on_word_head(...)` (`:164`)

- Returns `True` if the output ends with a character likely to start a Chinese word
- Used by the `word_head_guard` to avoid committing risky half-phrases (e.g., ending on `"这"`)

### `_semantic_normalize(...)` (`:188`)

- Normalizes `delta` prefixes for direction comparison
- Strips punctuation and noise prefixes like `"为"`, `"位"`, `"名"` so direction clustering is more semantic

### `check_direction(...)` (`:206`)

- Takes the `qualified` deltas and checks whether they roughly point to the same semantic direction
- Uses `_semantic_normalize` first, then falls back to raw prefix slicing
- Returns:
  - `bool` for direction consistency
  - a debug dict (`counts`, `top_key`, `ratio`) that is logged and saved in `details`

## 6. Future Sampling (Base Model)

`sample_source_futures(...)` at `llm_future_sampling_final.py:551`.

What it does:

- Calls base model `llm.generate([observed_source], SamplingParams(...))`
- Samples `M` continuations (`n=num_candidates`)
- Cleans continuations and caps to `max_words`

Why it matters:

- This stage controls diversity and uncertainty
- At unstable positions (e.g., sentence unfinished like `he is`), futures can split in many directions and make later translation/judging harder

## 7. Translation (Instruct Model) – Mentor-Style Continuation

### Prompt construction for translation

`_build_translation_prompt_text(...)` at `llm_future_sampling_final.py:605`.

This is the mentor-style part:

1. Build a `user` message with:
   - `[TASK] Translate ...`
   - `[INPUT] observed_source`
2. Apply tokenizer chat template with `add_generation_prompt=False`
3. Manually append:
   - `"<|im_start|>assistant\n"`
4. Append `committed` directly into the assistant turn

Effect:

- The model naturally continues from the committed Chinese prefix
- This is stronger than only saying "Do not reproduce committed text" in instructions

### Batch translation path

`_translate_batch_async(...)` at `llm_future_sampling_final.py:670`.

For each source candidate:

- Build mentor-style prompt text
- Call instruct server via `client.completions.create(prompt=...)`
- Clean output
- Filter non-Chinese explanation-like outputs via `_is_chinese_output(...)`

### Why `create(...)` is still used (not local `generate(...)`)

- `create(...)` is the HTTP SDK method name
- The actual generation still happens in the instruct vLLM server
- If the instruct model were loaded in-process as `LLM`/`AsyncLLMEngine`, then you would call `generate(...)` directly

## 8. Alignment and Safe Prefix

### Alignment model

`get_word_alignments(...)` and `truncate_by_alignment(...)` are upstream of chunk decisions.

The important behavior for logs:

- Full candidate translation is aligned against `(observed + future_i)`
- Then truncated to the part believed to cover only `observed`
- Result is `safe_prefix`

### Why this can create "truncation artifact" deltas

If:

- `committed_norm = "而且这"`
- `safe_prefix = "而且这些介绍不可避免"`

then:

- `delta = safe_prefix[len(committed_norm):] = "些介绍不可避免"`

This `delta` starts mid-phrase. It is not necessarily a bad translation. It is a slicing artifact.

This is exactly why the scoring prompt now includes both:

- `full_prefix_so_far`
- `new_delta_only`

and explicitly instructs the judge not to penalize truncation artifacts by default.

## 9. Judge Scoring – What Was Changed and Why

`build_score_prompt(...)` is at `llm_future_sampling_final.py:455`.

The current version fixes two major judge failure modes:

1. Over-translation miscalibration
2. Truncation-artifact mislabeling as mistranslation

### Over-translation definition (important)

The prompt now explicitly defines:

- Over-translation = translating content not yet present in `observed_source`
- If `observed_source` already contains a word/phrase, translating it is correct (not a penalty)

This prevents the judge from incorrectly penalizing cases like:

- observed includes `unavailing`
- delta translates `毫无成效`

### Truncation-aware scoring

The prompt now tells the judge:

- `new_delta_only` may start mid-word/mid-phrase
- use `full_prefix_so_far` for context
- do not treat truncation artifact alone as mistranslation

### Few-shot examples

The function includes 6 examples:

- faithful complete translation
- partial translation of observed content
- true over-translation
- truncation artifact but good prefix
- truncation artifact + real over-translation
- direct mistranslation (`introductions -> 演讲`)

This makes judge scoring much more stable and better calibrated for your streaming setup.

## 10. Chunk Decision Logic (Step 3–Step 5) with Log Mapping

This section maps directly to the verbose logs you posted.

### Step 3 in logs: `alignment-truncated candidates`

Code:

- `llm_future_sampling_final.py:975-:1017`

For each candidate:

- Build `full_src_for_candidate`
- Compute alignments
- Compute `safe_prefix`
- Check monotonicity vs committed
- Compute `delta`
- Check `length_ok`

This produces the log lines like:

- `monotonic=True`
- `len_ok=True`
- `delta_len=...`
- `safe_prefix="..."`

### Step 4 in logs: `judge scores`

Code:

- `llm_future_sampling_final.py:1034-:1058`

Flow:

1. Build `judge_items` containing:
   - `safe_prefix`
   - `delta`
2. Score via `score_candidate_prefixes(...)`
3. Attach `judge_score` and `judge_tags` back onto `valid_candidates`

This is where your log shows lines like:

- `idx=0 score=92 tags=['ok'] delta="..."`

### Step 5 in logs: `consensus + direction`

Code:

- `llm_future_sampling_final.py:1060-:1095`

Flow:

1. `qualified = score >= threshold`
2. `qualified_ratio` computed
3. `check_direction([...delta...])`
4. If direction is consistent:
   - choose highest-score candidate
   - `new_chars = best["delta"]`

This is the key change from old LCP behavior:

- no more `LCP` over all qualified candidates
- write content is from the best qualified candidate
- consensus is enforced by score ratio + direction consistency

## 11. Final Write Gate: `word_head_guard`

Code:

- `llm_future_sampling_final.py:1096-:1110`

Logic:

- `risky = _ends_on_word_head(new_chars) and qualified_ratio < 0.85`
- If risky, force `READ`
- Else if `len(new_chars) >= min_commit_chars`, `WRITE`

Why this exists:

- To reduce committing prefixes that likely end at a Chinese word head (`这`, `那`, `一`, etc.)
- This helps prevent the next chunk from seeing a semantically broken `delta` start

This specifically targets the behavior you observed such as `committed` getting stuck at:

- `"而且这"`

instead of waiting for a safer boundary.

## 12. Why You Sometimes See Many READs in a Row

This is usually not one bug. It is the combination of:

1. Candidate uncertainty is high (future sampling diverges a lot)
2. Alignment truncation yields weak or chopped deltas
3. Judge correctly rejects many deltas
4. Direction check finds no consensus
5. `word_head_guard` blocks risky short outputs

A long READ streak is often a sign the system is being conservative at an uncertain point. The key question is whether it eventually resumes writing with good quality.

## 13. How to Read Your Verbose Log Quickly (Practical Recipe)

When a chunk looks wrong, inspect in this order:

1. `Chunk X/Y`, `accumulated`, `committed`
   - Did we stop at a syntactically incomplete English position?

2. `[Step 1] futures`
   - Are sampled futures semantically clustered or wildly divergent?

3. `[Step 2] translations`
   - Are translations following the same topic, or are they all over the place?

4. `[Step 3] safe_prefix`
   - Are deltas chopped due to committed boundary (e.g., starts with `"些"`)?

5. `[Step 4] judge scores`
   - Are low scores due to real over-translation/mistranslation, or truncation artifact?

6. `[Step 5] consensus + direction`
   - `qualified_ratio` enough?
   - `direction_ok` true?
   - `direction_info["counts"]` split?

7. Final decision line
   - `READ (consensus ...)`
   - `READ (word_head_guard ...)`
   - `WRITE "..."`

This sequence usually tells you exactly which stage is causing the failure.

## 14. Output JSON: What to Inspect

The final per-utterance JSON stores:

- `actions`
- `target_future_sampling`
- `config`
- `details` (if `--save-details`)

Important `config` fields:

- `selection_mode`: current decision aggregation mode
- `patches`: active patch list for experiment tracking

See result assembly at `llm_future_sampling_final.py:1131-:1162`.

## 15. Current Patch Set (What It Means)

The script now records:

- `judge_fewshot_calibration_v2`
- `semantic_direction_check`
- `word_head_guard`

Meaning:

- Judge is calibrated against over-translation confusion and truncation artifacts
- Direction consensus is semanticized (less sensitive to prefix noise)
- Final write is guarded against ending on risky Chinese word-head characters under weak consensus

## 16. Known Remaining Limitations (Important)

These are still real issues in the current implementation.

1. Final chunk completion (`translate_final`) can still append a rewritten tail if model output does not cleanly continue `committed`
2. Alignment truncation is character-based on the target side and can still create awkward boundaries
3. Future sampling quality dominates everything at high-uncertainty points (`he is`, `of`, `that`, etc.)
4. The instruct judge and instruct translator are the same model family, so they can share biases

## 17. Suggested Next Debugging Targets (If You Continue Iterating)

If the next run still has quality issues, I would inspect in this order:

1. `translate_final` merge logic (duplicate/rewrite tail risk)
2. Alignment truncation boundary smoothing (safer target-side cut)
3. Dynamic policy for high-uncertainty observed suffixes (e.g., function-word endings)
4. Direction-check thresholds (`n`, `min_ratio`) as CLI flags

