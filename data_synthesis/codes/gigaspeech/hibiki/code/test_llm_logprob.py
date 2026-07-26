from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


model_name = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
tp_size = 1
topk_logprobs = 5

system_prompt = (
    "You are a professional translator. Translate the English source into "
    "Chinese. Output only the Chinese translation."
)
source_text = "These forewords are inevitably both monotonous and useless."
target_text = "这些前言不可避免地既单调又无益。"


def get_logprob_value(logprob_obj):
    if hasattr(logprob_obj, "logprob"):
        return float(logprob_obj.logprob)
    return float(logprob_obj)


def format_candidate(tokenizer, token_id, logprob_obj):
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    decoded = tokenizer.decode([token_id])
    logprob = get_logprob_value(logprob_obj)
    return {
        "token_id": int(token_id),
        "raw_token": raw_token,
        "decoded": decoded,
        "logprob": logprob,
    }


tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    dtype="auto",
    tensor_parallel_size=tp_size,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)

messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": "English source:\n"
        f"{source_text}\n\nChinese translation:",
    },
]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
prompt_text = tokenizer.decode(prompt_ids)

target_ids = tokenizer.encode(target_text, add_special_tokens=False)
target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

print("Prompt text:")
print(repr(prompt_text))
print("\nTarget text:", target_text)
print("Target IDs:", target_ids)
print("Target tokens:", target_tokens)

print("\nHow vLLM is called:")
print("full_ids = prompt_ids + target_ids[:i] + [target_ids[i]]")
print("outputs = llm.generate([TokensPrompt(prompt_token_ids=full_ids)], params)")
print("actual token logprob = outputs[0].prompt_logprobs[-1][target_ids[i]]")

params = SamplingParams(
    max_tokens=1,
    temperature=0.0,
    prompt_logprobs=topk_logprobs,
)

print("\nPer-token teacher-forced logprob:")
for i, target_token_id in enumerate(target_ids):
    prefix_ids = target_ids[:i]
    full_ids = list(prompt_ids) + list(prefix_ids) + [target_token_id]

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=full_ids)],
        params,
    )
    output = outputs[0]
    prompt_logprobs = output.prompt_logprobs
    last_entry = None if prompt_logprobs is None else prompt_logprobs[-1]

    raw_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]
    decoded_token = tokenizer.decode([target_token_id])
    decoded_prefix = tokenizer.decode(target_ids[: i + 1])

    print(
        f"\n{i:02d} | id={target_token_id} | raw={repr(raw_token)} "
        f"| decoded={repr(decoded_token)} | prefix={repr(decoded_prefix)}"
    )

    if last_entry is None:
        print("  prompt_logprobs[-1] is None")
        continue

    if target_token_id not in last_entry:
        print(
            "  target token not found in prompt_logprobs[-1], "
            f"keys={list(last_entry.keys())[:10]}"
        )
        continue

    actual_logprob = get_logprob_value(last_entry[target_token_id])
    print(f"  actual logprob: {actual_logprob:.6f}")

    sorted_candidates = sorted(
        last_entry.items(),
        key=lambda kv: get_logprob_value(kv[1]),
        reverse=True,
    )
    print(f"  top {min(topk_logprobs, len(sorted_candidates))} candidates:")
    for rank, (cand_id, cand_obj) in enumerate(sorted_candidates[:topk_logprobs], start=1):
        item = format_candidate(tokenizer, cand_id, cand_obj)
        print(
            f"    {rank:02d} | id={item['token_id']} | raw={repr(item['raw_token'])} "
            f"| decoded={repr(item['decoded'])} | logprob={item['logprob']:.6f}"
        )
