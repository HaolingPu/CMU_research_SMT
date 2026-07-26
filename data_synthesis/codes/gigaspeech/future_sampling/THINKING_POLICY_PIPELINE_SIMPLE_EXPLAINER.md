# Thinking-Policy Pipeline Simple Explainer

This note explains the current `thinking_policy` pipeline in a way that is easy
to present to someone else.

Relevant code:

- `run_thinking_policy_pool_10_array.sh`
- `llm_future_sampling_thinking_policy.py`

## 1. One-Sentence Idea

We do simultaneous translation by asking:

1. "What English might come next?"
2. "Given all those possible futures, what Chinese is already safe to say now?"
3. "Did we accidentally translate something that belongs to the future rather than the observed prefix?"

If the answer is safe, we `WRITE`. Otherwise we `READ`.

## 2. Pipeline Diagram

```text
Manifest TSV
  |
  |  each row contains:
  |  - src_text_full
  |  - src_trajectory
  v
Controller Python
  |
  |  for each trajectory chunk:
  v
Accumulate observed English prefix
  |
  v
Base model samples N future continuations
  |
  v
Thinking model proposes one safe Chinese delta
  |
  v
Simalign post-check trims/rejects future-only content
  |
  +--> delta empty      -> READ
  |
  +--> delta non-empty  -> WRITE
                          append delta to committed Chinese
  |
  v
Last chunk:
force-complete the remaining translation
  |
  v
Write one JSON output per utterance
```

## 3. What Runs Where

```text
Base model
  - local vLLM `LLM(...)`
  - job: sample possible English futures

Thinking model
  - OpenAI-compatible vLLM servers
  - job: decide the next safe Chinese delta

Simalign
  - alignment post-check
  - job: cut off translation that crosses into future-only English words
```

## 4. Input / Output

Input row:

- `src_text_full`: the full English utterance, usually sentence list
- `src_trajectory`: incremental chunks revealed over time

Output JSON:

- `source_future_sampling`: the chunk sequence
- `target_future_sampling`: Chinese deltas emitted at each step
- `actions`: `READ` or `WRITE`
- `system_output_text`: final concatenated Chinese output

## 5. Walkthrough Example

Toy example:

- full source: `he is the editor, and not the author.`
- trajectory:
  - `he is`
  - `the editor,`
  - `and not`
  - `the author.`

### Step A

Observed prefix:

```text
he is
```

Base model futures might be:

```text
1. a worker at the local school.
2. a teacher who later became a principal.
3. the editor, and not the author.
```

Common safe meaning across all futures:

```text
他是
```

Thinking model proposes:

```text
他是
```

Simalign says this is still within observed English.

Decision:

```text
WRITE "他是"
```

Committed Chinese becomes:

```text
他是
```

### Step B

Observed prefix:

```text
he is the editor,
```

Base model futures might be:

```text
1. and not the author.
2. and not the author of the preface.
3. but also a critic.
```

Now the safe shared meaning is:

```text
编辑
```

Thinking model proposes a delta like:

```text
编辑
```

Decision:

```text
WRITE "编辑"
```

Committed Chinese becomes:

```text
他是编辑
```

### Step C

Observed prefix:

```text
he is the editor, and not
```

Possible futures:

```text
1. the author.
2. the author of the preface.
```

The phrase `author` is now stable across futures, so the thinking model can
safely emit:

```text
，而不是作者
```

After simalign, if the prefix is still supported by the observed source, we:

```text
WRITE "，而不是作者"
```

Committed Chinese becomes:

```text
他是编辑，而不是作者
```

### Step D

Observed prefix:

```text
he is the editor, and not the author.
```

This is the last chunk, so we do not rely only on incremental delta. We call
the final completion step to finish whatever remains and fix punctuation if
needed.

Decision:

```text
WRITE "。"
```

Final output:

```text
他是编辑，而不是作者。
```

## 6. Why This Pipeline Is Useful

Compared with direct translation from the observed prefix only, this pipeline is
more conservative and more stable because:

- the base model explicitly models uncertainty about the future
- the thinking model only emits content consistent with all sampled futures
- simalign provides a hard post-check against future-only over-translation
- the final completion step guarantees we do not end with a half-finished utterance

## 7. Very Short Verbal Version

If you need a 15-second explanation, use this:

```text
At every chunk, we first imagine several ways the English could continue.
Then we ask a thinking model: across all those futures, what Chinese is already safe to say now?
Then we use alignment to verify we did not accidentally translate future-only content.
If something is safe, we WRITE it; otherwise we READ and wait for more source.
At the end, we finish the translation.
```
