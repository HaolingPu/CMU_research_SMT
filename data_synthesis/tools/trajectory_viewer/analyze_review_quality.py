#!/usr/bin/env python3
"""Create a case-by-case quality audit for the packaged 100-case review set."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


ASSESSMENTS = {
    9: (
        "EARLY_COMMIT_FAIL",
        "At step 3, `They were so huge that the` commits `它们如此巨大，以至于`. "
        "The 36 selected futures mix people, animals, and inanimate objects; the actual continuation "
        "(`spoons they used`) and prior giant context require human `他们`. The wrong pronoun is irreversible.",
    ),
    14: (
        "MT_SEMANTIC_ERROR",
        "`his captor` becomes `他的俘虏` (his captive), reversing the semantic relation. "
        "The cue was already observed, so this is translator lexical quality rather than early commitment.",
    ),
    20: (
        "MT_SEMANTIC_ERROR",
        "The utterance-final command `Say at once` becomes narrative `说道：`, changing speech act. "
        "The complete phrase was available before the final commit.",
    ),
    29: (
        "WATCH",
        "The final object attachment is malformed: sword, slipper, and handkerchief should all modify "
        "the person being sought. Meaning remains recoverable, but the Chinese coordination is poor.",
    ),
    33: (
        "WATCH",
        "Very literal clause structure (`啊，确实，你父亲的真正儿子`) is awkward, but no specific "
        "future-dependent fact was committed incorrectly.",
    ),
    42: (
        "WATCH",
        "`stick to your needle` is translated literally. This may be an intentional tailor pun, while the "
        "reference paraphrases it differently; not enough evidence for an early-commit failure.",
    ),
    52: (
        "MT_SEMANTIC_ERROR",
        "The opening attachment becomes `对她母亲来说，她是个寡妇`, making the daughter appear to be "
        "the widow. The translator had already seen `her mother, who was a widow`.",
    ),
    54: (
        "MT_SEMANTIC_ERROR",
        "`cats ... inspect the new servant` becomes cats inspecting `她新雇的仆人`, changing who the "
        "new servant is. The decoder waited through the relevant phrase, so this is parsing/MT quality.",
    ),
    60: (
        "WATCH",
        "The source itself ends at `when Lizina,` but the reference contains an unstated continuation. "
        "The model correctly leaves the translation incomplete; this is a source/reference boundary issue.",
    ),
    61: (
        "WATCH",
        "Singular `her sister` becomes plural `她的姐姐们`. This is a local number error, not an "
        "ambiguity-sensitive early commit.",
    ),
    64: (
        "BOUNDARY_AMBIGUITY_FAIL",
        "Missing punctuation hides the speaker turn in `wife then go to your death cried the king`. "
        "The futures assume one continuing speaker and the output attaches `go to your death` to the wrong "
        "speaker. The method does not cover this dialogue-boundary alternative.",
    ),
    67: (
        "MT_SEMANTIC_ERROR",
        "The comparative-negation construction `no more modest than she was good and kind` is rendered as "
        "positive `既不自矜，又善良仁慈`. The entire construction was observed before commit.",
    ),
    71: (
        "EARLY_COMMIT_FAIL",
        "The decoder READs while the candidates contain both `he` and `her`, but after the prefix resets at "
        "the next clause it commits `在他额头...`. `their adored Lizina` then resolves the referent as female; "
        "the correct `她` can no longer be recovered.",
    ),
    72: (
        "WATCH",
        "`Fair Lizina` is translated as `美丽的莉齐娜`, while the supplied reference uses `公平的`. "
        "The model reading is plausible in the fairy-tale context, so this is reference/lexical ambiguity.",
    ),
    83: (
        "EARLY_COMMIT_FAIL",
        "At `Here I must remain no`, the decoder correctly READs and futures predict `longer`. The next "
        "normalized prefix resets to only `and`; it then commits `我必须留在这里了`, reversing `must remain "
        "no longer`. This is an irreversible negation error caused by lost prefix context.",
    ),
    84: (
        "MT_SEMANTIC_ERROR",
        "`the king's son allowed himself to be persuaded` becomes `王子说服了自己`, reversing passive "
        "persuasion into self-persuasion. The full construction was visible before commit.",
    ),
    99: (
        "WATCH",
        "No early semantic error is visible, but LAAL is 23.23 and the first WRITE occurs at step 21. "
        "This is an extreme latency failure even though the final content is mostly faithful.",
    ),
}


SUCCESS_NOTES = {
    1: "Strong ambiguity handling: READs through `introductions are inevitably`, then commits only stable `而这些介绍` before the unresolved adjectives.",
    25: "Aggressive but safe: commits the Chinese alternative frame after `frozen to death or`, then waits for `buried in the snow`.",
    35: "Safe partial commitment: `No, I won't` yields only `不`, postponing the complement until it is observed.",
    58: "Safe underspecification: `Before her stood` commits only `在她面前`, leaving the object unresolved.",
    90: "Safe underspecification: `This so enraged the` commits only `这使`, waiting for `king`.",
    97: "Safe underspecification: `When he heard a` commits only `当他听到`, waiting for the sound type.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def first_write(case: dict) -> dict | None:
    return next(
        (
            step
            for step in case["steps"]
            if step["action"] == "WRITE" and step["translation_delta"]
        ),
        None,
    )


def compact(text: str, limit: int = 70) -> str:
    value = " ".join(text.split()).replace("|", "\\|")
    return value if len(value) <= limit else f"{value[: limit - 1]}..."


def default_note(case: dict) -> str:
    step = first_write(case)
    if not step:
        return "No non-empty WRITE was produced."
    prefix = compact(step["future_source_prefix"], 45)
    delta = compact(step["translation_delta"], 28)
    return f"No material early-commit error found; first WRITE at step {step['step']}: `{prefix}` -> `{delta}`."


def aggregate(cases: list[dict]) -> dict:
    steps = [step for case in cases for step in case["steps"]]
    future_steps = [step for step in steps if step["selected_futures"]]
    selected_counts = [
        sum(len(group["candidates"]) for group in step["selected_futures"])
        for step in future_steps
    ]
    raw_groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for step in future_steps:
        for group in step["raw_stats"]:
            key = (group["label"], group["mode"])
            raw_groups[key][0] += group["requested"]
            raw_groups[key][1] += group["kept"]

    statuses = defaultdict(int)
    for order in range(1, len(cases) + 1):
        statuses[ASSESSMENTS.get(order, ("PASS", ""))[0]] += 1

    bleus = [case["metrics"]["bleu_char"] for case in cases]
    laals = [case["metrics"]["laal_text"] for case in cases]
    return {
        "case_count": len(cases),
        "trajectory_steps": len(steps),
        "read_steps": sum(step["action"] == "READ" for step in steps),
        "write_steps": sum(step["action"] == "WRITE" for step in steps),
        "future_steps": len(future_steps),
        "selected_future_mean": statistics.mean(selected_counts),
        "selected_future_median": statistics.median(selected_counts),
        "steps_with_fewer_than_10_futures": sum(count < 10 for count in selected_counts),
        "bleu_mean": statistics.mean(bleus),
        "bleu_median": statistics.median(bleus),
        "bleu_min": min(bleus),
        "bleu_max": max(bleus),
        "laal_mean": statistics.mean(laals),
        "laal_median": statistics.median(laals),
        "status_counts": dict(statuses),
        "future_keep_rates": {
            f"{label} / {mode}": kept / requested
            for (label, mode), (requested, kept) in raw_groups.items()
        },
    }


def render_markdown(cases: list[dict], stats: dict) -> str:
    lines = [
        "# Future Consensus 100-Case Quality Audit",
        "",
        "## Review criterion",
        "",
        "An **early-commit failure** is counted only when the system irreversibly WRITEs target content "
        "whose correct form depends on source information that has not yet arrived, or when a prefix reset "
        "drops already observed disambiguating context before the commit. Errors made after the complete "
        "relevant source phrase is available are reported separately as MT semantic errors.",
        "",
        "This is a structured model-assisted audit of all stored trajectories, not a replacement for "
        "independent bilingual human annotation.",
        "",
        "## Aggregate result",
        "",
        f"- Cases: **{stats['case_count']}**; trajectory decisions: **{stats['trajectory_steps']}** "
        f"({stats['read_steps']} READ / {stats['write_steps']} WRITE).",
        f"- Confirmed early-commit failures: **{stats['status_counts'].get('EARLY_COMMIT_FAIL', 0)}/100** "
        "(cases 9, 71, and 83).",
        f"- Boundary-ambiguity failure: **{stats['status_counts'].get('BOUNDARY_AMBIGUITY_FAIL', 0)}/100** "
        "(case 64).",
        f"- Separate MT semantic errors: **{stats['status_counts'].get('MT_SEMANTIC_ERROR', 0)}/100**; "
        f"watch/data/latency cases: **{stats['status_counts'].get('WATCH', 0)}/100**.",
        f"- Character BLEU mean/median: **{stats['bleu_mean']:.2f}/{stats['bleu_median']:.2f}** "
        f"(range {stats['bleu_min']:.2f}-{stats['bleu_max']:.2f}).",
        f"- LAAL mean/median: **{stats['laal_mean']:.2f}/{stats['laal_median']:.2f}**.",
        f"- Mean selected futures per sampled step: **{stats['selected_future_mean']:.1f}** "
        f"(median {stats['selected_future_median']:.0f}); only "
        f"{stats['steps_with_fewer_than_10_futures']} sampled steps had fewer than 10.",
        "",
        "Future keep rates:",
        "",
    ]
    for label, rate in stats["future_keep_rates"].items():
        lines.append(f"- {label}: **{rate:.1%}**")

    lines.extend(
        [
            "",
            "## What the evidence says",
            "",
            "The method is doing useful work: most cases delay content-bearing decisions and commit only "
            "stable Chinese scaffolding. Cases 1, 25, 35, 58, 90, and 97 are clear examples. However, the "
            "100-case set does **not** support the claim that early commitment is solved. Three confirmed "
            "failures remain, and all three had 36 selected futures at the failing step. More candidates "
            "alone will not fix them.",
            "",
            "The dominant weaknesses are: (1) sampler semantic bias, as in case 9; (2) loss of antecedent or "
            "negation context when the normalized prefix resets, as in cases 71 and 83; and (3) failure to "
            "represent dialogue-boundary alternatives when punctuation is missing, as in case 64.",
            "",
            "Low BLEU is not equivalent to bad translation in this sample. Cases 2, 16, and 49 have BLEU "
            "below 34 but are semantically strong paraphrases. Conversely, cases 71 and 83 contain material "
            "semantic errors despite otherwise plausible output.",
            "",
            "## Case-by-case audit",
            "",
            "| # | Utterance | BLEU | LAAL | Status | Assessment |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    records = []
    for order, case in enumerate(cases, 1):
        status, note = ASSESSMENTS.get(order, ("PASS", SUCCESS_NOTES.get(order, default_note(case))))
        record = {
            "order": order,
            "utt_id": case["utt_id"],
            "bleu_char": case["metrics"]["bleu_char"],
            "laal_text": case["metrics"]["laal_text"],
            "status": status,
            "assessment": note,
        }
        records.append(record)
        lines.append(
            f"| {order} | `{case['utt_id']}` | {record['bleu_char']:.1f} | "
            f"{record['laal_text']:.2f} | **{status}** | {note} |"
        )

    lines.extend(
        [
            "",
            "## Recommended fixes and next evaluation",
            "",
            "1. Carry the full observed source context and unresolved target state across prefix-normalization "
            "boundaries; never evaluate a new `and`/sentence fragment without the clause that contains an "
            "unresolved negation or antecedent.",
            "2. Add explicit uncertainty constraints for gendered Chinese pronouns. If futures contain both "
            "male and female realizations, allow neutral `其`/name repetition or force READ.",
            "3. Add dialogue-boundary futures for punctuation-poor ASR: same speaker, speaker switch, and "
            "reported-speech alternatives.",
            "4. Evaluate early commitment directly with an oracle full-source translation alignment or "
            "human annotation. BLEU/COMET alone cannot identify when an irreversible decision was made.",
            "5. Build a stratified review set across recordings and ambiguity types. These 100 cases are "
            "contiguous TSV-order segments from one GigaSpeech recording and are not a representative test set.",
            "",
        ]
    )
    return "\n".join(lines), records


def main() -> None:
    args = parse_args()
    payload = json.loads(args.review_json.read_text())
    cases = sorted(payload["cases"], key=lambda case: case["row_index"])
    stats = aggregate(cases)
    markdown, records = render_markdown(cases, stats)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown)
    args.output_json.write_text(
        json.dumps({"summary": stats, "cases": records}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
