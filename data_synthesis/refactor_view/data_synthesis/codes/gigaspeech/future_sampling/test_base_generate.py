#!/usr/bin/env python3
"""
Quick test: Qwen3-30B-A3B-Base with llm.generate() for source continuation.

Usage (on a GPU node):
  conda activate vllm
  CUDA_VISIBLE_DEVICES=0 python test_base_generate.py

This loads the Base model and tests pure text continuation (no chat template).
"""

import os
import time

os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/user_data/haolingp/models/Qwen3-30B-A3B-FP8"

TEST_SOURCES = [
    "The president announced today that the government will invest heavily in",
    "Scientists have discovered a new species of deep-sea fish that can",
    "In the heart of the city, a small café serves the best coffee you",
    "The annual technology conference brought together experts from around the world to discuss",
    "After years of research, the team finally published their findings on",
]

NUM_CANDIDATES = 10
FUTURE_TOKENS = 30


def main():
    print("=" * 60)
    print("Test: Qwen3-30B-A3B-Base with llm.generate()")
    print("=" * 60)

    print(f"\nLoading model from {MODEL_PATH} ...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="auto",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Sampling params for diverse continuation
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.6,
        max_tokens=FUTURE_TOKENS,
        n=NUM_CANDIDATES,
    )

    # For each test source, generate N candidates
    for i, src in enumerate(TEST_SOURCES):
        print(f"\n{'─' * 60}")
        print(f"Source {i+1}: \"{src}\"")
        print(f"{'─' * 60}")

        t1 = time.time()
        outputs = llm.generate([src], sampling_params)
        elapsed = time.time() - t1

        for j, out in enumerate(outputs[0].outputs):
            print(f"  Candidate {j+1:2d}: {out.text.strip()}")

        print(f"  (generated {len(outputs[0].outputs)} candidates in {elapsed:.2f}s)")

    # Also test: batch of prompts (simulating real pipeline)
    print(f"\n{'=' * 60}")
    print("Batch test: 5 prompts × 10 candidates each")
    print(f"{'=' * 60}")
    t2 = time.time()
    batch_outputs = llm.generate(TEST_SOURCES, sampling_params)
    print(f"Batch completed in {time.time() - t2:.2f}s")
    for i, out in enumerate(batch_outputs):
        print(f"  Source {i+1}: {len(out.outputs)} candidates generated")

    print("\n✓ Base model test PASSED")


if __name__ == "__main__":
    main()
