---
title: Thinking Policy
type: concept
tags: [synthesis, policy, reasoning]
sources:
  - ../codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py
  - ../codes/gigaspeech/future_sampling/THINKING_POLICY_PIPELINE_SIMPLE_EXPLAINER.md
created: 2026-06-01
updated: 2026-06-01
---

# Thinking Policy

A [[future-sampling]] variant where a **reasoning model** makes the per-chunk READ/WRITE decision
by explicitly reasoning through future uncertainty, instead of code-based consensus. The thinking
chains are saved in the output for interpretability.

Main framework: `../codes/gigaspeech/future_sampling/llm_future_sampling_thinking_policy.py`
(supports OpenAI and Gemini APIs). Backend variants:
- Gemini: `gemini/llm_future_sampling_thinking_policy_gemini_json_flash*.py` (Flash, Pro fallback,
  future-distribution modeling).
- OpenAI o1: `llm_future_sampling_thinking_policy_openai.py`.
- DeepSeek R1: `deepseek/llm_future_sampling_thinking_policy_deepseek_reasoner.py` and a local
  R1-distill-Qwen32B variant.

Output: `../outputs/gigaspeech/train_xl_future_sampling_thinking*`.

## Related
- [[future-sampling]], [[consensus-decoding]], [[synthesis-pipeline]].

## Sources
- code: `../codes/gigaspeech/future_sampling/` (thinking_policy*, gemini/, deepseek/)
