import os
import re
import soundfile as sf

CORPUS = "/data/user_data/haolingp/outputs/mfa_corpus/en000"

bad_dirs = []

def is_bad_token(tok):
    # MFA 英文 + arpa 最常见不接受的 token
    return (
        tok.isdigit()
        or tok.startswith("<")
        or tok.startswith("[")
        or re.search(r"[^a-zA-Z' ]", tok)
    )

for sub in sorted(os.listdir(CORPUS)):
    d = os.path.join(CORPUS, sub)
    if not os.path.isdir(d):
        continue

    wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
    labs = sorted(f for f in os.listdir(d) if f.endswith(".lab"))

    problems = []

    if len(wavs) != len(labs):
        problems.append(f"wav/lab count mismatch: {len(wavs)} vs {len(labs)}")

    wav_set = {w.replace(".wav", "") for w in wavs}
    lab_set = {l.replace(".lab", "") for l in labs}

    if wav_set != lab_set:
        problems.append("wav/lab basename mismatch")

    for l in labs[:3]:  # 抽查 3 个
        with open(os.path.join(d, l)) as f:
            text = f.read().strip()
        if not text:
            problems.append(f"{l} is empty")
            break

        toks = text.split()
        bad = [t for t in toks if is_bad_token(t)]
        if bad:
            problems.append(f"{l} has bad tokens: {bad[:3]}")
            break

    for w in wavs[:1]:
        try:
            audio, sr = sf.read(os.path.join(d, w))
            if len(audio) < sr * 0.2:
                problems.append(f"{w} too short (<0.2s)")
        except Exception as e:
            problems.append(f"{w} unreadable: {e}")

    if problems:
        bad_dirs.append((sub, problems))

print(f"Checked {len(os.listdir(CORPUS))} dirs")
print(f"Bad dirs: {len(bad_dirs)}")

for sub, probs in bad_dirs[:10]:
    print(sub)
    for p in probs:
        print("  -", p)
