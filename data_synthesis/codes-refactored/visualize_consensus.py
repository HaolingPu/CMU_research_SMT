"""Render consensus future-sampling traces as standalone HTML for inspection.

Reads one (json, verbose log) pair per utterance from a consensus_future task
directory and writes one self-contained .html per utterance. Open in a browser.

Usage:
  python visualize_consensus.py \
      --consensus-dir /data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/consensus_future/sample_200/ \
      --out-dir ./viz_out \
      --utt-id AUD0000000003_0           # one utterance, OR
      --all [--limit N]                  # batch
      [--overwrite]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_CHUNK_HEADER_RE = re.compile(r"^Chunk\s+(\d+)\s*/\s*(\d+)\s*$")
_FUTURE_RE = re.compile(r"^\s*future\[(\d+)\]\s*\((primary|secondary)\):\s*(.+?)\s*$")
_STEP_ACCEPTED_RE = re.compile(r"^\s*step=(\d+)\s+accepted=(['\"])(.*)\2\s+pending=(['\"])(.*)\4\s*$")
_STEP_STOP_RE = re.compile(
    r"^\s*step=(\d+)\s+stop=(\S+)\s+intersection=\[(.*)\]\s+pending=(['\"])(.*)\4\s*$"
)
_CAND_HEADER_RE = re.compile(r"^\s*future\[(\d+)\]\s+candidates=(\d+):\s*\[(.*)\]\s*$")
_CAND_PAIR_RE = re.compile(r"'([^']*)':(-?\d+(?:\.\d+)?)")
_KV_QUOTED_RE = re.compile(r"^([a-z_]+):\s*(['\"])(.*)\2\s*$")


def _strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _parse_intersection(body: str) -> list[str]:
    body = body.strip()
    if not body:
        return []
    # Body is a list of quoted strings: 'a', 'b', 'c'
    return [m.group(1) for m in re.finditer(r"'([^']*)'", body)]


def _parse_candidates(body: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for m in _CAND_PAIR_RE.finditer(body):
        tok, prob = m.group(1), m.group(2)
        try:
            out.append((tok, float(prob)))
        except ValueError:
            continue
    return out


def parse_header_comments(lines: list[str]) -> tuple[dict, int]:
    """Parse the leading `# key: value` block. Returns (header, end_idx)."""
    header: dict[str, Any] = {}
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        line = lines[i].rstrip()
        if line.startswith("############"):
            i += 1
            continue
        body = line.lstrip("#").lstrip()
        if not body:
            i += 1
            continue
        # Bare comma-separated kv pairs, e.g. "num_futures=20, top_k=5, min_p=0.0"
        if ":" not in body and "=" in body:
            for p in (s.strip() for s in body.split(",")):
                if "=" in p:
                    k, _, v = p.partition("=")
                    header[k.strip()] = _coerce(v.strip())
            i += 1
            continue
        if ":" not in body:
            i += 1
            continue
        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()
        # Multi-line list values: collect lines until matching closing bracket
        if value == "[" or (value.startswith("[") and not value.endswith("]")):
            buf = [value]
            i += 1
            while i < len(lines) and lines[i].startswith("#"):
                inner = lines[i].lstrip("#").rstrip()
                inner_strip = inner.lstrip()
                buf.append(inner_strip)
                if inner_strip.startswith("]"):
                    i += 1
                    break
                i += 1
            joined = " ".join(buf)
            try:
                header[key] = json.loads(joined)
            except json.JSONDecodeError:
                header[key] = joined
            continue
        # base_model[primary]: api model=... base=... num_futures=...
        if key.startswith("base_model"):
            tokens = value.split()
            d: dict[str, Any] = {}
            d["mode"] = tokens[0] if tokens else ""
            for tok in tokens[1:]:
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    d[k] = _coerce(v)
            header[key] = d
            i += 1
            continue
        header[key] = _coerce(value)
        i += 1
    return header, i


def _coerce(v: str) -> Any:
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_consensus_log(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header, idx = parse_header_comments(lines)

    chunks: list[dict] = []
    cur: dict | None = None
    section: str | None = None  # "futures" | "consensus" | None
    cur_step: dict | None = None

    def flush_chunk():
        nonlocal cur
        if cur is not None:
            chunks.append(cur)
            cur = None

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("===") and idx + 1 < len(lines):
            m = _CHUNK_HEADER_RE.match(lines[idx + 1].strip())
            if m:
                flush_chunk()
                cur = {
                    "chunk_idx": int(m.group(1)),
                    "total_chunks": int(m.group(2)),
                    "source_observed": "",
                    "source_observed_full": "",
                    "future_source_prefix": "",
                    "committed_before": "",
                    "futures": [],
                    "consensus_steps": [],
                    "pending_before_trim": "",
                    "commit_after_trim": "",
                    "action": "",
                    "delta": "",
                    "committed_after": "",
                    "is_final": False,
                }
                section = None
                cur_step = None
                idx += 2
                continue

        if cur is None:
            idx += 1
            continue

        # Key/value lines at top of chunk block
        if section is None or section in {"futures", "consensus"}:
            kv = _KV_QUOTED_RE.match(line)
            if kv:
                k = kv.group(1)
                v = kv.group(3)
                if k in cur:
                    cur[k] = v
                idx += 1
                continue

        # [Step 1-2] future_sampling total=N
        if stripped.startswith("[Step 1-2]"):
            section = "futures"
            idx += 1
            continue

        # [Step 4-5] consensus summary:
        if stripped.startswith("[Step 4-5]"):
            section = "consensus"
            cur_step = None
            idx += 1
            continue

        # [Step 6-7] pending_before_trim='...'
        if stripped.startswith("[Step 6-7]"):
            section = None
            cur_step = None
            body = line.split("]", 1)[1].strip()
            kv = _KV_QUOTED_RE.match(body)
            if kv and kv.group(1) in cur:
                cur[kv.group(1)] = kv.group(3)
            idx += 1
            continue

        # -> READ delta='...' / -> WRITE delta='...' / -> READ (too few futures)
        if stripped.startswith("-> "):
            body = stripped[3:]
            if body.startswith("READ"):
                cur["action"] = "READ"
            elif body.startswith("WRITE"):
                cur["action"] = "WRITE"
            m = re.search(r"delta=(['\"])(.*)\1", body)
            if m:
                cur["delta"] = m.group(2)
            idx += 1
            continue

        # Final chunk: "  [Final] delta='...'"
        if stripped.startswith("[Final]"):
            cur["is_final"] = True
            m = re.search(r"delta=(['\"])(.*)\1", stripped)
            delta = m.group(2) if m else ""
            cur["delta"] = delta
            cur["pending_before_trim"] = delta
            cur["commit_after_trim"] = delta
            cur["action"] = "WRITE" if delta else "READ"
            cur["committed_after"] = (cur.get("committed_before", "") or "") + delta
            idx += 1
            continue

        # Within sections
        if section == "futures":
            m = _FUTURE_RE.match(line)
            if m:
                cur["futures"].append({
                    "idx": int(m.group(1)),
                    "kind": m.group(2),
                    "text": _strip_outer_quotes(m.group(3)),
                })
            idx += 1
            continue

        if section == "consensus":
            sa = _STEP_ACCEPTED_RE.match(line)
            if sa:
                cur_step = {
                    "step": int(sa.group(1)),
                    "accepted": sa.group(3),
                    "stop_reason": None,
                    "intersection": [],
                    "pending": sa.group(5),
                    "future_candidates": [],
                }
                cur["consensus_steps"].append(cur_step)
                idx += 1
                continue
            ss = _STEP_STOP_RE.match(line)
            if ss:
                cur_step = {
                    "step": int(ss.group(1)),
                    "accepted": None,
                    "stop_reason": ss.group(2),
                    "intersection": _parse_intersection(ss.group(3)),
                    "pending": ss.group(5),
                    "future_candidates": [],
                }
                cur["consensus_steps"].append(cur_step)
                idx += 1
                continue
            ch = _CAND_HEADER_RE.match(line)
            if ch and cur_step is not None:
                cur_step["future_candidates"].append(_parse_candidates(ch.group(3)))
                idx += 1
                continue

        idx += 1

    flush_chunk()
    return {"header": header, "chunks": chunks}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", "Noto Sans CJK SC", "Helvetica Neue",
               Arial, sans-serif;
}
body {
  margin: 0;
  padding: 24px 32px 64px;
  background: #fafafa;
  color: #222;
  font-size: 14px;
  line-height: 1.5;
  max-width: 1280px;
  margin-left: auto;
  margin-right: auto;
}
h1 { font-size: 20px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 24px 0 8px; }
.muted { color: #888; }
.summary {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 16px 20px;
  margin-bottom: 16px;
}
.summary .label { font-weight: 600; color: #555; }
.summary .text  { white-space: pre-wrap; }
.summary .text.zh { font-family: "PingFang SC", "Noto Sans CJK SC",
                                 "Microsoft YaHei", monospace; }
.metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 24px;
  margin-top: 8px;
  font-size: 13px;
}
.metrics span code { background: #f1f1f1; padding: 1px 6px; border-radius: 3px; }
.chunk {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-left-width: 6px;
  border-radius: 6px;
  margin: 12px 0;
  padding: 12px 16px;
}
.chunk.write { border-left-color: #2e7d32; }
.chunk.read  { border-left-color: #9e9e9e; }
.chunk-header {
  font-weight: 600;
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: baseline;
  margin-bottom: 8px;
}
.chunk-header .badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.5px;
}
.chunk.write .chunk-header .badge { background: #2e7d32; }
.chunk.read  .chunk-header .badge { background: #9e9e9e; }
.chunk-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 8px;
}
.chunk-pane {
  background: #f7f7f7;
  border-radius: 4px;
  padding: 8px 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.chunk-pane .label {
  display: block;
  font-size: 11px;
  color: #888;
  margin-bottom: 4px;
  letter-spacing: 0.5px;
}
.chunk-pane .delta { color: #2e7d32; font-weight: 700; }
.chunk-pane .delta.src { background: #e8f5e9; padding: 0 2px; border-radius: 2px; }
.chunk-pane .pending { color: #888; font-style: italic; }
details { margin: 6px 0; }
details > summary {
  cursor: pointer;
  padding: 4px 8px;
  background: #f0f0f0;
  border-radius: 4px;
  font-size: 13px;
  user-select: none;
}
details > summary:hover { background: #e8e8e8; }
details > div { padding: 8px 4px; }
.future {
  font-family: ui-monospace, SFMono-Regular, "Menlo", monospace;
  font-size: 12px;
  margin: 2px 0;
  padding-left: 60px;
  text-indent: -60px;
  word-break: break-word;
}
.future .tag {
  display: inline-block;
  width: 52px;
  font-weight: 600;
  font-size: 11px;
}
.future.primary  .tag { color: #1565c0; }
.future.secondary .tag { color: #6a1b9a; }
.consensus-step {
  margin: 10px 0;
  padding: 8px 10px;
  background: #fafbfc;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
}
.consensus-step.stop  { background: #fff7f7; border-color: #f5cccc; }
.consensus-step .step-header { font-weight: 600; margin-bottom: 6px; }
.consensus-step .accepted { color: #2e7d32; }
.consensus-step .stop-reason { color: #b71c1c; }
table.cands {
  border-collapse: collapse;
  font-family: ui-monospace, SFMono-Regular, "Menlo", monospace;
  font-size: 11px;
  width: 100%;
}
table.cands td, table.cands th {
  border: 1px solid #e0e0e0;
  padding: 3px 6px;
  text-align: center;
  vertical-align: top;
}
table.cands th { background: #f5f5f5; font-weight: 600; }
table.cands td.tok-accepted { background: #e8f5e9; font-weight: 700; }
table.cands td.tok-empty { color: #ccc; }
table.cands .row-label { font-weight: 600; color: #555; background: #f5f5f5; }
.candidate { display: flex; flex-direction: column; align-items: center; }
.candidate .tok { font-weight: 600; }
.candidate .prob { color: #888; font-size: 10px; }
table.idx { width: 100%; border-collapse: collapse; }
table.idx th, table.idx td {
  border: 1px solid #e0e0e0;
  padding: 6px 10px;
  font-size: 13px;
  text-align: left;
}
table.idx th { background: #f5f5f5; cursor: pointer; user-select: none; }
table.idx tr:hover td { background: #fafafa; }
table.idx td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.idx a { color: #1565c0; text-decoration: none; }
table.idx a:hover { text-decoration: underline; }
"""


def _esc(s: str) -> str:
    return html.escape(s if s is not None else "", quote=False)


def render_top_summary(header: dict, json_data: dict) -> str:
    utt_id = _esc(json_data.get("utt_id", header.get("utt_id", "")))
    src = _esc(json_data.get("source_full_text", header.get("source_full_text", "")))
    pred = _esc(json_data.get("prediction", ""))
    ref = _esc(json_data.get("reference_text", ""))
    primary = _esc(str(header.get("base_model[primary]", "")))
    secondary = _esc(str(header.get("base_model[secondary]", "")))
    metrics = json_data.get("metrics", {}) or {}
    decoder = json_data.get("decoder_impl", {}) or {}

    metric_items = []
    for k in ("bleu_char", "laal_text", "length_ratio_ref", "length_ratio_src",
              "pred_chars", "ref_chars", "src_words"):
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                v = f"{v:.3f}" if abs(v) < 100 else f"{v:.2f}"
            metric_items.append(f"<span><b>{k}</b> <code>{_esc(str(v))}</code></span>")
    metric_html = "".join(metric_items)

    decoder_str = ", ".join(f"{k}={v}" for k, v in decoder.items())

    return f"""
<div class="summary">
  <h1>{utt_id}</h1>
  <div class="muted" style="font-size:12px;margin-bottom:8px;">
    decoder: {_esc(decoder_str)}<br>
    primary: {primary}<br>
    secondary: {secondary}
  </div>
  <div><span class="label">Source:</span> <span class="text">{src}</span></div>
  <div style="margin-top:8px;"><span class="label">Prediction:</span>
       <span class="text zh">{pred}</span></div>
  <div style="margin-top:4px;"><span class="label">Reference:</span>
       <span class="text zh">{ref}</span></div>
  <div class="metrics">{metric_html}</div>
</div>
"""


def render_consensus_step(step: dict, futures: list[dict]) -> str:
    is_stop = step.get("stop_reason") is not None
    klass = "consensus-step stop" if is_stop else "consensus-step"
    if is_stop:
        intersection = step.get("intersection") or []
        inter_repr = ", ".join(f"'{_esc(t)}'" for t in intersection) or "—"
        head = (f"step={step['step']} → "
                f"<span class='stop-reason'>STOP "
                f"({_esc(step.get('stop_reason', ''))}, intersection=[{inter_repr}])"
                f"</span> · pending=<code>{_esc(step.get('pending', ''))}</code>")
    else:
        head = (f"step={step['step']} → accepted="
                f"<span class='accepted'>'{_esc(step.get('accepted', ''))}'</span>"
                f" · pending=<code>{_esc(step.get('pending', ''))}</code>")

    cands_per_future = step.get("future_candidates") or []
    if not cands_per_future:
        body = "<div class='muted'>(no candidate distributions logged)</div>"
    else:
        max_k = max((len(c) for c in cands_per_future), default=0)
        accepted = step.get("accepted")
        rows = []
        for fi, cands in enumerate(cands_per_future):
            kind = futures[fi]["kind"] if fi < len(futures) else ""
            row = [f"<td class='row-label'>f{fi}<br><span style='font-size:10px;color:#999'>{kind[:3]}</span></td>"]
            for j in range(max_k):
                if j < len(cands):
                    tok, prob = cands[j]
                    klass2 = "tok-accepted" if (accepted is not None and tok == accepted) else ""
                    row.append(
                        f"<td class='{klass2}'>"
                        f"<div class='candidate'>"
                        f"<span class='tok'>{_esc(tok)}</span>"
                        f"<span class='prob'>{prob:.3f}</span>"
                        f"</div></td>"
                    )
                else:
                    row.append("<td class='tok-empty'>—</td>")
            rows.append("<tr>" + "".join(row) + "</tr>")
        header_row = "<tr><th></th>" + "".join(
            f"<th>top {j+1}</th>" for j in range(max_k)
        ) + "</tr>"
        body = (f"<table class='cands'>{header_row}{''.join(rows)}</table>")

    return f"<div class='{klass}'><div class='step-header'>{head}</div>{body}</div>"


def render_chunk(chunk: dict) -> str:
    action = chunk.get("action") or "READ"
    klass = "chunk write" if action == "WRITE" else "chunk read"

    src_obs = chunk.get("source_observed", "")
    delta = chunk.get("delta", "")
    fsp = chunk.get("future_source_prefix", "")
    committed_before = chunk.get("committed_before", "")
    pending = chunk.get("pending_before_trim", "")

    # Source pane: show committed-source-so-far + new chunk
    if fsp.endswith(src_obs):
        src_prev = fsp[: len(fsp) - len(src_obs)].rstrip()
    else:
        src_prev = fsp
    src_prev_html = _esc(src_prev)
    src_new_html = _esc(src_obs)

    # Target pane
    if action == "WRITE":
        tgt_html = (f"{_esc(committed_before)}"
                    f"<span class='delta'>{_esc(delta)}</span>")
    else:
        if pending and pending != committed_before:
            extra = (f"<span class='pending'>(no commit — pending="
                     f"'{_esc(pending)}')</span>")
        else:
            extra = "<span class='pending'>(no commit)</span>"
        tgt_html = f"{_esc(committed_before)} {extra}"

    futures = chunk.get("futures") or []
    futures_html = "".join(
        f"<div class='future {f['kind']}'>"
        f"<span class='tag'>[{f['kind'][:3]}{f['idx']:02d}]</span>"
        f"{_esc(f['text'])}</div>"
        for f in futures
    ) or "<div class='muted'>(no futures sampled — too few candidates)</div>"

    consensus_steps = chunk.get("consensus_steps") or []
    consensus_html = "".join(
        render_consensus_step(s, futures) for s in consensus_steps
    ) or "<div class='muted'>(no consensus steps logged)</div>"

    header = (
        f"<span>Chunk {chunk['chunk_idx']}/{chunk['total_chunks']}</span>"
        f"<span class='badge'>{action}</span>"
        f"<span class='muted'>source_observed=<code>'{_esc(src_obs)}'</code></span>"
        f"<span class='muted'>delta=<code>'{_esc(delta)}'</code></span>"
    )

    return f"""
<div class="{klass}">
  <div class="chunk-header">{header}</div>
  <div class="chunk-body">
    <div class="chunk-pane">
      <span class="label">SOURCE so far</span>
      {src_prev_html}<span class="delta src">{src_new_html}</span>
    </div>
    <div class="chunk-pane">
      <span class="label">TARGET so far</span>
      <span class="text zh">{tgt_html}</span>
    </div>
  </div>
  <details>
    <summary>Futures ({len(futures)})</summary>
    <div>{futures_html}</div>
  </details>
  <details>
    <summary>Consensus steps ({len(consensus_steps)})</summary>
    <div>{consensus_html}</div>
  </details>
</div>
"""


def render_html(json_data: dict, parsed_log: dict) -> str:
    header = parsed_log.get("header", {}) or {}
    chunks = parsed_log.get("chunks", []) or []
    title = json_data.get("utt_id", header.get("utt_id", "consensus trace"))
    parts = [
        "<!doctype html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        render_top_summary(header, json_data),
        "<h2>Per-chunk decisions</h2>",
    ]
    parts.extend(render_chunk(c) for c in chunks)
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

_INDEX_JS = """
function sortTable(idx) {
  const tbl = document.querySelector('table.idx');
  const tbody = tbl.tBodies[0];
  const rows = Array.from(tbody.rows);
  const numeric = rows.every(r => !isNaN(parseFloat(r.cells[idx].dataset.sort
                                                    ?? r.cells[idx].textContent)));
  const dir = tbl.dataset.sortDir === 'asc' && tbl.dataset.sortCol === String(idx) ? -1 : 1;
  rows.sort((a, b) => {
    const av = a.cells[idx].dataset.sort ?? a.cells[idx].textContent;
    const bv = b.cells[idx].dataset.sort ?? b.cells[idx].textContent;
    if (numeric) return (parseFloat(av) - parseFloat(bv)) * dir;
    return av.localeCompare(bv) * dir;
  });
  rows.forEach(r => tbody.appendChild(r));
  tbl.dataset.sortDir = dir === 1 ? 'asc' : 'desc';
  tbl.dataset.sortCol = String(idx);
}
"""


def render_index(entries: list[dict]) -> str:
    rows = []
    for e in entries:
        m = e.get("metrics") or {}

        def cell_num(v, fmt="{:.3f}"):
            if v is None or (isinstance(v, float) and (v != v)):
                return "<td class='num'>—</td>"
            try:
                return f"<td class='num' data-sort='{float(v)}'>{fmt.format(float(v))}</td>"
            except (TypeError, ValueError):
                return f"<td class='num'>{_esc(str(v))}</td>"

        rows.append(
            "<tr>"
            f"<td><a href='{_esc(e['html'])}'>{_esc(e['utt_id'])}</a></td>"
            f"{cell_num(m.get('bleu_char'), '{:.2f}')}"
            f"{cell_num(m.get('laal_text'), '{:.2f}')}"
            f"{cell_num(m.get('length_ratio_ref'), '{:.3f}')}"
            f"{cell_num(m.get('length_ratio_src'), '{:.3f}')}"
            f"{cell_num(m.get('pred_chars'), '{:.0f}')}"
            f"{cell_num(m.get('ref_chars'), '{:.0f}')}"
            f"{cell_num(m.get('src_words'), '{:.0f}')}"
            f"{cell_num(e.get('num_chunks'), '{:.0f}')}"
            f"{cell_num(e.get('num_writes'), '{:.0f}')}"
            f"<td>{_esc(e.get('preview', '')[:60])}</td>"
            "</tr>"
        )
    headers = ["utt_id", "bleu_char", "laal_text", "len/ref", "len/src",
               "pred_chars", "ref_chars", "src_words", "chunks", "writes",
               "prediction preview"]
    th = "".join(
        f"<th onclick='sortTable({i})'>{_esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>consensus trace index</title>
<style>{_CSS}</style>
<script>{_INDEX_JS}</script>
</head><body>
<h1>Consensus traces · {len(entries)} utterances</h1>
<p class='muted'>Click column headers to sort.</p>
<table class='idx'>
  <thead><tr>{th}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_one(consensus_dir: str, utt_id: str, out_dir: str,
                 overwrite: bool) -> dict | None:
    json_path = os.path.join(consensus_dir, f"{utt_id}.json")
    log_path = os.path.join(consensus_dir, "verbose", f"verbose_{utt_id}.log")
    out_path = os.path.join(out_dir, f"{utt_id}.html")

    if not os.path.exists(json_path):
        print(f"skip {utt_id}: missing {json_path}", file=sys.stderr)
        return None
    if not os.path.exists(log_path):
        print(f"skip {utt_id}: missing {log_path}", file=sys.stderr)
        return None
    if os.path.exists(out_path) and not overwrite:
        print(f"skip {utt_id}: exists (use --overwrite)")
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        parsed = parse_consensus_log(log_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_html(json_data, parsed))
        print(f"wrote {out_path}")

    # Even if skipped, read JSON for index entry
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    actions = json_data.get("actions") or []
    return {
        "utt_id": utt_id,
        "html": f"{utt_id}.html",
        "metrics": json_data.get("metrics") or {},
        "num_chunks": len(actions),
        "num_writes": sum(1 for a in actions if a == "WRITE"),
        "preview": json_data.get("prediction") or "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--consensus-dir", required=True,
                    help="task directory containing <utt>.json and verbose/verbose_<utt>.log")
    ap.add_argument("--out-dir", required=True, help="output directory for HTML files")
    ap.add_argument("--utt-id", help="single utterance id to render")
    ap.add_argument("--all", action="store_true", help="render every *.json in --consensus-dir")
    ap.add_argument("--limit", type=int, default=None, help="cap number of utterances when --all")
    ap.add_argument("--overwrite", action="store_true", help="re-render even if output exists")
    args = ap.parse_args(argv)

    if not args.utt_id and not args.all:
        ap.error("specify --utt-id or --all")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.utt_id:
        utt_ids = [args.utt_id]
    else:
        utt_ids = sorted(
            os.path.splitext(p)[0]
            for p in os.listdir(args.consensus_dir)
            if p.endswith(".json")
        )
        if args.limit is not None:
            utt_ids = utt_ids[: args.limit]

    entries: list[dict] = []
    for utt_id in utt_ids:
        try:
            entry = _process_one(args.consensus_dir, utt_id, args.out_dir,
                                 args.overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {utt_id}: {exc!r}", file=sys.stderr)
            continue
        if entry is not None:
            entries.append(entry)

    if args.all and entries:
        idx_path = os.path.join(args.out_dir, "index.html")
        with open(idx_path, "w", encoding="utf-8") as f:
            f.write(render_index(entries))
        print(f"wrote {idx_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
