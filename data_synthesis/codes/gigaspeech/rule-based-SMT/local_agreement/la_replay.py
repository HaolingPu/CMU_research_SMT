#!/usr/bin/env python3
"""Verbose step-by-step replay of LA-N=2 random-segment synthesis traces."""
import json

CASES = [
    ('seg=1 (commit-every-chunk)',
     '/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_seg13/task_0/AUD0000000003_1011.json'),
    ('seg=2',
     '/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_seg13/task_0/AUD0000000003_100.json'),
    ('seg=3',
     '/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/rule_based_SMT/local_agreement/la_rand40k_seg13/task_0/AUD0000000003_0.json'),
]


def replay(label, path):
    r = json.load(open(path))
    seg = r['decoder_impl']['segment_size']
    la_n = r['decoder_impl']['la_n']
    chunks = r['src_trajectory']
    actions = r['actions']
    targets = r['target_trajectory']
    n = len(chunks)
    n_steps = (n + seg - 1) // seg

    print('=' * 100)
    print(f'CASE: {label}   utt={r["utt_id"]}')
    print(f'  segment_size={seg}  la_n={la_n}  n_chunks={n}  n_steps={n_steps}')
    print(f'  reference: {r["reference_text"]}')
    print(f'  prediction: {r["prediction"]}')
    print(f'  metrics: {r.get("metrics")}')
    print('-' * 100)
    print(f'{"step":<5}{"chunk":<7}{"role":<16}{"action":<7}{"src_word":<24}{"delta":<28}{"committed_so_far"}')
    print('-' * 100)

    committed = ''
    step_start = 0
    step_idx = 0
    while step_start < n:
        step_end = min(step_start + seg, n)
        is_last_step = (step_end == n)
        for i in range(step_start, step_end):
            if i == step_end - 1:
                role = 'force_complete' if is_last_step else 'LCP-decision'
            else:
                role = 'pre-read'
            src = chunks[i] or '<silence>'
            act = actions[i]
            tgt = targets[i] or ''
            delta = tgt[len(committed):] if tgt.startswith(committed) else f'<RESET>{tgt}'
            committed = tgt
            print(f'{step_idx:<5}{i:<7}{role:<16}{act:<7}{src[:22]:<24}{repr(delta)[:26]:<28}{committed[-40:]}')
        step_idx += 1
        step_start = step_end
    print()


for label, path in CASES:
    replay(label, path)
