#!/usr/bin/env python3
"""Generate a single HTML visualizer of READ/WRITE trajectories for 3 models
across 8 (speed, seg) conditions on the future-aware test set.

Each instance shows the English source, gold reference, and three stacked
timelines (one per model). Each timeline chunk is clickable to play the
corresponding audio span; a "play full" button plays the whole utterance.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd

CKPT_ROOT = Path("/data/user_data/siqiouya/ckpts/infinisst-omni")
DATA_ROOT = Path("/data/group_data/li_lab/siqiouya/datasets/future_aware_test")

MODELS = [
    ("s_origin", "gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf"),
    ("hibiki", "gigaspeech-zh-hibiki-s-bsz4/v0-20260326-141050-hf"),
    ("consensus-topk5_f200", "gigaspeech-zh-consensus-topk5_f200-s-bsz4/v0-20260502-125501-hf"),
]
MODEL_COLORS = {
    "s_origin": "#1f77b4",
    "hibiki": "#ff7f0e",
    "consensus-topk5_f200": "#2ca02c",
}

SEGS = [960, 1920, 2880, 3840]
SPEEDS = ["1.0", "0.7"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument(
        "--data-tsv",
        type=Path,
        default=DATA_ROOT / "future_aware_testset_v2.tsv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rw_trajectories.html",
    )
    p.add_argument(
        "--portable",
        action="store_true",
        help="Emit relative audio paths (wav_spd_<speed>/FA_NNNN.wav) so the HTML "
        "is portable when bundled with the wav dirs as siblings.",
    )
    return p.parse_args()


def load_instances(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            out[inst["index"]] = inst
    return out


def group_writes(prediction: str, delays: list[float]) -> list[tuple[float, str]]:
    """Group chars by their delay (source-time of emission).
    Returns list of (delay_ms, chars) in emission order.
    """
    groups: list[list] = []
    pos = 0
    for d in delays:
        if not groups or groups[-1][0] != d:
            groups.append([d, ""])
        groups[-1][1] += prediction[pos]
        pos += 1
    return [(d, c) for d, c in groups]


def build_chunks(source_length_ms: float, seg_ms: int, groups: list[tuple[float, str]]) -> list[dict]:
    """Bucket the timeline into chunks of `seg_ms`, with a final partial chunk
    ending exactly at `source_length_ms`. WRITE detection is range-based: any
    delay falling in (prev_boundary, this_boundary] counts as a write within
    that chunk (covers end-of-stream finalize writes whose delay equals
    source_length and is not a multiple of seg_ms).
    """
    boundaries: list[float] = []
    n = 1
    while n * seg_ms < source_length_ms:
        boundaries.append(float(n * seg_ms))
        n += 1
    boundaries.append(float(source_length_ms))

    chunks: list[dict] = []
    prev = 0.0
    for b in boundaries:
        chars = ""
        for d, c in groups:
            if prev < d <= b:
                chars += c
        chunks.append(
            {
                "start_ms": prev,
                "end_ms": b,
                "chars": chars,
                "is_write": bool(chars),
            }
        )
        prev = b
    return chunks


def render_chunk_html(chunk: dict, audio_id: str) -> str:
    klass = "chunk write" if chunk["is_write"] else "chunk read"
    label = (
        f'<span class="t">{int(chunk["end_ms"])}</span>'
        f'<span class="ch">{html.escape(chunk["chars"])}</span>'
        if chunk["is_write"]
        else '<span class="t">&nbsp;</span><span class="ch">&nbsp;</span>'
    )
    return (
        f'<button class="{klass}" '
        f'data-audio="{audio_id}" '
        f'data-start="{chunk["start_ms"]:.1f}" '
        f'data-end="{chunk["end_ms"]:.1f}" '
        f'onclick="playSegment(this)">{label}</button>'
    )


def render_instance(
    idx: int,
    row: pd.Series,
    speed: str,
    seg: int,
    per_model: dict[str, dict],
    tab_id: str,
    portable: bool,
) -> str:
    utt_id = row["id"]
    src_text = html.escape(str(row["source"]))
    ref_text = html.escape(str(row["reference"]))
    utt_type = html.escape(str(row["type"]))

    if portable:
        audio_src = f"wav_spd_{speed}/{utt_id}.wav"
    else:
        audio_src = f"file://{DATA_ROOT / f'wav_spd_{speed}' / f'{utt_id}.wav'}"
    audio_id = f"aud-{tab_id}-{idx}"
    source_length_ms = next(iter(per_model.values()))["source_length"]

    chunk_lists: dict[str, list[dict]] = {}
    for label, _ in MODELS:
        inst = per_model[label]
        groups = group_writes(inst["prediction"], inst["delays"])
        chunk_lists[label] = build_chunks(source_length_ms, seg, groups)

    rows_html_parts: list[str] = []
    n_chunks = len(next(iter(chunk_lists.values())))
    for label, _ in MODELS:
        chunks = chunk_lists[label]
        chunk_html = "".join(render_chunk_html(c, audio_id) for c in chunks)
        color = MODEL_COLORS[label]
        pred_escaped = html.escape(per_model[label]["prediction"])
        rows_html_parts.append(
            f'<div class="row">'
            f'<span class="label" style="border-left-color:{color}">{html.escape(label)}</span>'
            f'<div class="bar">'
            f'<div class="playhead" data-playhead-for="{audio_id}"></div>'
            f"{chunk_html}"
            f'</div>'
            f'<span class="pred" title="{pred_escaped}">{pred_escaped}</span>'
            f"</div>"
        )

    ticks_html_parts = []
    if n_chunks > 0:
        chunks0 = chunk_lists[MODELS[0][0]]
        for c in chunks0:
            ticks_html_parts.append(f'<div class="tick">{int(c["end_ms"])}</div>')
    ticks_html = (
        '<div class="row tick-row">'
        '<span class="label">&nbsp;</span>'
        f'<div class="bar">{"".join(ticks_html_parts)}</div>'
        '<span class="pred">&nbsp;</span>'
        "</div>"
    )

    return f"""
<section class="inst" id="{tab_id}-inst-{idx}">
  <h3>
    <a class="anchor" href="#{tab_id}-toc-{idx}">{html.escape(utt_id)}</a>
    <span class="tag">{utt_type}</span>
    <span class="dur">{source_length_ms/1000:.2f}s</span>
    <button class="full-play" onclick="playFull('{audio_id}')">▶ full</button>
    <span class="elapsed" data-elapsed-for="{audio_id}">0 / {int(source_length_ms)} ms</span>
  </h3>
  <div class="src"><b>EN:</b> {src_text}</div>
  <div class="ref"><b>ZH ref:</b> {ref_text}</div>
  <audio id="{audio_id}" src="{audio_src}" preload="none"
         data-source-length-ms="{source_length_ms:.4f}"
         ontimeupdate="updatePlayhead(this)"
         onseeked="updatePlayhead(this)"
         onpause="updatePlayhead(this)"></audio>
  <div class="timelines">
    {''.join(rows_html_parts)}
    {ticks_html}
  </div>
</section>"""


def render_tab_panel(
    tab_id: str,
    speed: str,
    seg: int,
    df: pd.DataFrame,
    per_inst: dict[int, dict[str, dict]],
    is_active: bool,
    portable: bool,
) -> str:
    toc_links = "".join(
        f'<a id="{tab_id}-toc-{idx}" href="#{tab_id}-inst-{idx}">'
        f'{html.escape(row["id"])}</a>'
        for idx, row in df.iterrows()
    )

    instances_html = "".join(
        render_instance(idx, row, speed, seg, per_inst[idx], tab_id, portable)
        for idx, row in df.iterrows()
        if idx in per_inst
    )

    return f"""
<div class="tab-panel{' active' if is_active else ''}" id="panel-{tab_id}">
  <nav class="toc"><b>TOC:</b> {toc_links}</nav>
  {instances_html}
</div>"""


def build_html(df: pd.DataFrame, all_data: dict[tuple[str, int], dict[int, dict[str, dict]]], portable: bool) -> str:
    tab_buttons = []
    tab_panels = []
    first = True
    for speed in SPEEDS:
        for seg in SEGS:
            tab_id = f"spd{speed.replace('.', '_')}_seg{seg}"
            label = f"speed {speed}, seg {seg}ms"
            tab_buttons.append(
                f'<button class="tab-btn{" active" if first else ""}" '
                f'onclick="showTab(\'{tab_id}\', this)">{label}</button>'
            )
            tab_panels.append(
                render_tab_panel(tab_id, speed, seg, df, all_data[(speed, seg)], first, portable)
            )
            first = False

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 0; color: #222; background: #fafafa; font-size: 14px; }
header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
         padding: 8px 16px; z-index: 100; }
header h1 { margin: 0 0 6px; font-size: 16px; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; }
.tab-btn { padding: 6px 12px; border: 1px solid #ccc; background: #f0f0f0;
           cursor: pointer; border-radius: 4px; font-size: 13px; }
.tab-btn.active { background: #1f77b4; color: #fff; border-color: #1f77b4; }
.tab-panel { display: none; padding: 12px 16px; }
.tab-panel.active { display: block; }
.toc { background: #fff; border: 1px solid #ddd; padding: 8px;
       border-radius: 4px; margin-bottom: 12px; font-size: 12px;
       max-height: 100px; overflow-y: auto; }
.toc a { display: inline-block; margin: 1px 4px; color: #1f77b4;
         text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.inst { background: #fff; border: 1px solid #e2e2e2; border-radius: 6px;
        padding: 10px 14px; margin-bottom: 10px; }
.inst h3 { margin: 0 0 6px; font-size: 14px; display: flex;
           align-items: center; gap: 8px; }
.inst h3 a.anchor { color: #222; text-decoration: none; }
.inst h3 .tag { background: #ffd966; color: #333; padding: 1px 6px;
                border-radius: 3px; font-size: 11px; }
.inst h3 .dur { color: #888; font-size: 11px; font-weight: normal; }
.full-play { margin-left: auto; padding: 2px 8px; font-size: 12px;
             cursor: pointer; background: #fff; border: 1px solid #999;
             border-radius: 3px; }
.full-play:hover { background: #eee; }
.src, .ref { margin: 2px 0; }
.timelines { margin-top: 6px; font-family: ui-monospace, 'SF Mono', monospace; }
.row { display: flex; align-items: stretch; margin: 1px 0; min-height: 28px; }
.row .label { width: 180px; flex-shrink: 0; padding: 4px 6px; font-weight: 600;
              border-left: 4px solid #ccc; background: #f5f5f5;
              font-size: 12px; align-self: stretch;
              display: flex; align-items: center; }
.row .bar { display: flex; flex: 1; gap: 1px; position: relative; }
.playhead { position: absolute; top: 0; bottom: 0; width: 2px;
            background: #d32f2f; pointer-events: none; z-index: 10;
            left: 0; opacity: 0; transition: left 0.05s linear; }
.playhead.active { opacity: 1; }
.elapsed { font-size: 11px; color: #d32f2f; font-family: ui-monospace, monospace;
           font-weight: 500; }
.row .pred { width: 220px; flex-shrink: 0; padding: 4px 6px;
             font-size: 12px; color: #555; overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap;
             font-family: -apple-system, sans-serif; }
.chunk { flex: 1; border: none; padding: 2px 3px; font-size: 11px;
         cursor: pointer; min-width: 30px; display: flex;
         flex-direction: column; justify-content: center; align-items: center;
         line-height: 1.15; font-family: inherit; }
.chunk:hover { outline: 2px solid #555; outline-offset: -2px; }
.chunk.read { background: #e8e8e8; color: #999; }
.chunk.write { background: #c8e6c9; color: #1b5e20; font-weight: 600; }
.chunk .t { font-size: 9px; color: #777; font-weight: normal; }
.chunk.write .t { color: #2e7d32; }
.chunk .ch { font-size: 13px; }
.row.tick-row { min-height: 14px; }
.row.tick-row .label, .row.tick-row .pred { background: transparent;
              border: none; }
.row.tick-row .bar { gap: 1px; }
.tick { flex: 1; text-align: center; font-size: 9px; color: #888;
        min-width: 30px; }
"""

    js = """
function showTab(tabId, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + tabId).classList.add('active');
  btn.classList.add('active');
}

function playSegment(btn) {
  const audio = document.getElementById(btn.dataset.audio);
  if (!audio) return;
  const startSec = parseFloat(btn.dataset.start) / 1000;
  const endSec = parseFloat(btn.dataset.end) / 1000;
  audio.pause();
  audio.currentTime = startSec;
  const onUpdate = () => {
    if (audio.currentTime >= endSec) {
      audio.pause();
      audio.removeEventListener('timeupdate', onUpdate);
    }
  };
  audio.addEventListener('timeupdate', onUpdate);
  audio.play();
}

function playFull(audioId) {
  const audio = document.getElementById(audioId);
  if (!audio) return;
  audio.pause();
  audio.currentTime = 0;
  audio.play();
}

function updatePlayhead(audio) {
  const sourceLenMs = parseFloat(audio.dataset.sourceLengthMs || '0');
  if (!sourceLenMs) return;
  const curMs = audio.currentTime * 1000;
  const pct = Math.max(0, Math.min(100, (curMs / sourceLenMs) * 100));
  document.querySelectorAll('[data-playhead-for="' + audio.id + '"]').forEach(h => {
    h.style.left = pct + '%';
    h.classList.add('active');
  });
  const el = document.querySelector('[data-elapsed-for="' + audio.id + '"]');
  if (el) el.textContent = Math.round(curMs) + ' / ' + Math.round(sourceLenMs) + ' ms';
}
"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Future-Aware R/W Trajectories</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Future-aware test set: read/write trajectories
    (s_origin / hibiki / consensus-topk5_f200)</h1>
  <div class="tabs">{''.join(tab_buttons)}</div>
</header>
{''.join(tab_panels)}
<script>{js}</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.data_tsv, sep="\t")
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} test rows from {args.data_tsv}")

    all_data: dict[tuple[str, int], dict[int, dict[str, dict]]] = {}
    for speed in SPEEDS:
        for seg in SEGS:
            per_inst: dict[int, dict[str, dict]] = {}
            missing_any = False
            for label, model_path in MODELS:
                inst_log = (
                    args.ckpt_root
                    / model_path
                    / "evaluation/future_aware_test/en-zh"
                    / f"spd_{speed}_seg{seg}"
                    / "instances.log"
                )
                if not inst_log.exists():
                    print(f"  MISSING: {inst_log}")
                    missing_any = True
                    continue
                model_data = load_instances(inst_log)
                for idx, inst in model_data.items():
                    per_inst.setdefault(idx, {})[label] = inst
            all_data[(speed, seg)] = {
                idx: d for idx, d in per_inst.items() if len(d) == len(MODELS)
            }
            n = len(all_data[(speed, seg)])
            tag = " (some models missing)" if missing_any else ""
            print(f"  speed={speed} seg={seg}: {n} instances usable{tag}")

    html_str = build_html(df, all_data, args.portable)
    args.output.write_text(html_str)
    print(f"\nWrote {args.output}  ({args.output.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
