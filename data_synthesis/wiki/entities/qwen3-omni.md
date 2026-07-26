---
title: Qwen3-Omni-30B-A3B
type: entity
tags: [model, base]
sources:
  - scripts/train/train_EAST_s.sh
created: 2026-06-01
updated: 2026-06-01
---

# Qwen3-Omni-30B-A3B

The multimodal base model fine-tuned for all simultaneous-translation checkpoints. Path:
`ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/`. Takes audio chunks (960ms) + text,
trained via [[megatron-swift]] with LoRA to produce the [[infinisst-omni]] checkpoint family.

Separate from the LLMs used inside synthesis (Qwen3-4B/30B base, Gemma, Gemini, DeepSeek for
[[future-sampling]] / [[thinking-policy]]) — those are inference-only, not fine-tuned here.

## Related
- [[megatron-swift]], [[infinisst-omni]].

## Sources
- model: `ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/`
