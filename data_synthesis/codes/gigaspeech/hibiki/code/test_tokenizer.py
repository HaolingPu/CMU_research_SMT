from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"   # 改成你实际在用的 tokenizer 名字
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "这些前言不可避免地既单调又无益。"

token_ids = tokenizer.encode(text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
decoded = tokenizer.decode(token_ids)

print("Original text:", text)
print("Token IDs:", token_ids)
print("Tokens:", tokens)
print("Decoded full:", decoded)
print("Num chars:", len(text))
print("Num tokens:", len(token_ids))

print("\nPer-token inspection:")
for i, tid in enumerate(token_ids):
    raw_tok = tokenizer.convert_ids_to_tokens([tid])[0]
    decoded_tok = tokenizer.decode([tid])
    print(f"{i:02d} | id={tid} | raw={repr(raw_tok)} | decoded={repr(decoded_tok)}")

print("\nCumulative prefix decoding:")
for i in range(1, len(token_ids) + 1):
    print(f"{i:02d} | {repr(tokenizer.decode(token_ids[:i]))}")