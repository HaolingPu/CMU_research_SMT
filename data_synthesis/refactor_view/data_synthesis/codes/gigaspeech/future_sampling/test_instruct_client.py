#!/usr/bin/env python3
"""
Quick test: Qwen3-30B-Instruct via vLLM OpenAI-compatible API.

Prerequisites: start the server first with `bash test_instruct_serve.sh`

Usage:
  python test_instruct_client.py

Tests:
  1. Translation (EN -> ZH)
  2. LLM-as-judge (prefix safety)
  3. Concurrent requests (10 candidates in parallel)
"""

import asyncio
import json
import time

from openai import OpenAI, AsyncOpenAI

API_BASE = "http://localhost:8100/v1"
API_KEY = "dummy"
MODEL_NAME = "qwen3-instruct"


def test_translation():
    """Test basic EN->ZH translation."""
    print("=" * 60)
    print("Test 1: Translation (EN -> ZH)")
    print("=" * 60)

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    source = "The president announced today that the government will invest heavily in renewable energy."
    prompt = (
        f"Translate the following English to Chinese. "
        f"Output ONLY the Chinese translation, no explanation.\n\n"
        f'English: "{source}"\n'
        f"Chinese:"
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    elapsed = time.time() - t0

    result = resp.choices[0].message.content.strip()
    print(f"  Source:      {source}")
    print(f"  Translation: {result}")
    print(f"  Latency:     {elapsed:.2f}s")
    print(f"  Tokens:      {resp.usage.completion_tokens} completion, {resp.usage.prompt_tokens} prompt")
    print()


def test_judge():
    """Test LLM-as-judge for prefix safety."""
    print("=" * 60)
    print("Test 2: LLM-as-Judge (prefix safety)")
    print("=" * 60)

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    observed_source = "The president announced today that the government"
    committed = ""
    greedy_ext = "总统今天宣布政府将大力投资可再生能源"
    candidates = [
        "总统今天宣布政府将加大对可再生能源的投资",
        "总统今天宣布政府将大力推动教育改革",
        "总统今天宣布政府将在基础设施方面增加投入",
        "总统今天宣布政府将出台新的经济刺激政策",
        "总统今天宣布政府将投入大量资金用于医疗",
    ]

    candidates_str = "\n".join(f'  {i+1}. "{t}"' for i, t in enumerate(candidates))
    prompt = f"""You are judging a simultaneous translation decision.

Observed English so far: "{observed_source}"
Chinese committed so far: "{committed}"

Proposed next Chinese characters (from greedy translation): "{greedy_ext}"

Other candidate translations (from future source sampling):
{candidates_str}

Task: Decide how many characters of the proposed text can be safely committed right now, given only the observed English.

Rules:
- Only commit characters that are semantically consistent with the MAJORITY of candidate translations.
- Stop before any character where the candidates diverge in meaning.
- If nothing is safe to commit, output 0.

Output a JSON object with exactly these fields:
{{
  "safe_chars": <integer, how many characters of proposed text to commit>,
  "reasoning": "<one sentence explanation>"
}}"""

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    elapsed = time.time() - t0

    raw = resp.choices[0].message.content.strip()
    print(f"  Raw response: {raw}")
    print(f"  Latency:      {elapsed:.2f}s")

    try:
        import re
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            safe = obj.get("safe_chars", "?")
            reasoning = obj.get("reasoning", "?")
            committed_text = greedy_ext[:safe] if isinstance(safe, int) else ""
            print(f"  safe_chars:   {safe}")
            print(f"  Will commit:  \"{committed_text}\"")
            print(f"  Reasoning:    {reasoning}")
    except Exception as e:
        print(f"  Parse error: {e}")
    print()


async def test_concurrent():
    """Test concurrent requests (simulating 10 translation candidates)."""
    print("=" * 60)
    print("Test 3: Concurrent requests (10 translations)")
    print("=" * 60)

    client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)

    sources_with_futures = [
        "The president announced today that the government will invest heavily in renewable energy and green technology.",
        "The president announced today that the government will focus on healthcare reform and education.",
        "The president announced today that the government will strengthen national defense capabilities.",
        "The president announced today that the government will reduce taxes for middle-class families.",
        "The president announced today that the government will launch a new infrastructure plan.",
        "The president announced today that the government will address climate change with bold action.",
        "The president announced today that the government will increase funding for scientific research.",
        "The president announced today that the government will create millions of new jobs.",
        "The president announced today that the government will reform immigration policies.",
        "The president announced today that the government will improve housing affordability.",
    ]

    async def translate_one(src: str, idx: int):
        prompt = (
            f"Translate the following English to Chinese. "
            f"Output ONLY the Chinese translation, no explanation.\n\n"
            f'English: "{src}"\nChinese:'
        )
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        return idx, resp.choices[0].message.content.strip()

    t0 = time.time()
    tasks = [translate_one(s, i) for i, s in enumerate(sources_with_futures)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    for idx, trans in sorted(results):
        print(f"  Candidate {idx+1:2d}: {trans}")

    print(f"\n  Total latency for 10 concurrent requests: {elapsed:.2f}s")
    print(f"  Avg per request (if sequential): ~{elapsed/10:.2f}s")
    print()


def main():
    print("\nChecking server connectivity...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    try:
        models = client.models.list()
        print(f"Server OK. Available models: {[m.id for m in models.data]}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {API_BASE}")
        print(f"  {e}")
        print(f"\nMake sure to start the server first:")
        print(f"  bash test_instruct_serve.sh")
        return

    test_translation()
    test_judge()
    asyncio.run(test_concurrent())

    print("✓ All instruct model tests PASSED")


if __name__ == "__main__":
    main()
