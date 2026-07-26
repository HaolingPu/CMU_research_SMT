# Thinking Structured Output JSON Draft

This note proposes a structured-output variant for the thinking-model stage in
`llm_future_sampling_thinking_policy.py`.

The goal is not to magically improve translation quality by itself. The main
benefits are:

- cleaner state presentation to the model
- fixed output format
- easier parsing
- less dependence on `<think>` / `</think>` splitting

## Why try this

The current thinking path mixes three concerns in plain text:

- observed English seen so far
- committed Chinese translation so far
- sampled future English continuations

That works, but the model can still blur the boundary between these fields.
Using JSON makes the state explicit. A fixed JSON output schema also removes a
lot of brittle parsing logic around reasoning/content leakage.

## Proposed input format

The request body sent as `messages[0]["content"]` can be a JSON string like:

```json
{
  "task": "safe_incremental_simultaneous_interpretation",
  "version": "thinking_json_v1",
  "source_language": "English",
  "target_language": "Chinese",
  "output_contract": {
    "format": "json_object",
    "schema": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["WRITE", "READ"]
        },
        "delta": {
          "type": "string"
        }
      },
      "required": ["action", "delta"],
      "additionalProperties": false
    }
  },
  "rules": [
    "Return a JSON object only. No prose, no markdown, no code fences.",
    "The delta must append directly after committed_translation.",
    "The output may complete older observed meaning that is still missing or unfinished in committed_translation.",
    "Before translating newer material, first complete any unfinished tail already supported by observed_source.",
    "Do not revise, replace, or paraphrase already committed_translation. Only append new Chinese characters.",
    "The entire delta must remain valid under all future_continuations.",
    "If no safe new Chinese characters can be appended now, set action to READ and delta to an empty string."
  ],
  "examples": [
    {
      "observed_source": "he is",
      "future_continuations": [
        "a worker at the local school.",
        "a teacher who later became a principal."
      ],
      "committed_translation": "",
      "output": {
        "action": "WRITE",
        "delta": "他是"
      }
    },
    {
      "observed_source": "I went to the",
      "future_continuations": [
        "bank to deposit some cash.",
        "beach to watch the sunset."
      ],
      "committed_translation": "我去了",
      "output": {
        "action": "READ",
        "delta": ""
      }
    }
  ],
  "state": {
    "observed_source": "he is the editor, and not",
    "committed_translation": "他",
    "future_continuations": [
      "the author.",
      "the author of the preface."
    ]
  }
}
```

## Proposed output format

Keep the output schema minimal:

```json
{
  "action": "WRITE",
  "delta": "是编辑，而不是作者"
}
```

or

```json
{
  "action": "READ",
  "delta": ""
}
```

Why keep it minimal:

- fewer decoding errors
- less temptation for the model to over-explain
- easier post-processing
- easier A/B comparison against the current plain-text prompt

## Suggested integration path

### Stage 1: prompt-only experiment

Use the JSON request as the user content, but still parse the output as plain
text JSON with `json.loads`.

Pros:

- very easy to test
- no dependency on server-side structured outputs

Cons:

- the model can still output invalid JSON sometimes

### Stage 2: vLLM structured output

If the server version supports guided/structured outputs, enforce the schema on
the server side.

Your existing reference pattern in
`codes/gigaspeech/llm_output_gigaspeech_trajectory.py` is:

```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    repetition_penalty=1.1,
    guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
)
```

For the thinking-model API path, the equivalent request style depends on the
vLLM/OpenAI-compatible server version. The important idea is the same:

- define one JSON schema
- force the model to emit only that schema

## What was added in code

In `llm_future_sampling_thinking_policy.py`, three experimental helpers were
added without changing the default execution path:

- `THINKING_JSON_OUTPUT_SCHEMA`
- `build_thinking_json_request(...)`
- `build_thinking_json_prompt(...)`
- `parse_thinking_json_response(...)`

This means you can inspect and test the structured-output path first, without
affecting your current runs.

## Recommendation

Run this as an A/B experiment, not a full replacement yet:

- A: current plain-text prompt
- B: JSON state + minimal JSON output

Measure separately:

- parse stability
- over-translation rate
- final translation quality

The most likely win is cleaner parsing and more stable control flow. The true
translation quality may improve a little, but should be treated as an empirical
question.
