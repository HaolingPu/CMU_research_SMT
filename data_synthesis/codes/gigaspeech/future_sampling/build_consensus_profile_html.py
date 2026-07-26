#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isnan(result):
            return result
    return None


def _load_cases(experiment_dir: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    task_paths = sorted(experiment_dir.glob("task_*/*.json"))
    root_paths = sorted(path for path in experiment_dir.glob("*.json") if path.is_file())
    json_paths = task_paths if task_paths else root_paths

    for path in json_paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        actions = list(obj.get("actions") or [])
        source_chunks = list(obj.get("src_trajectory") or obj.get("source_chunks") or [])
        target_chunks = list(obj.get("target_trajectory") or obj.get("target_future_sampling") or [])
        metrics = obj.get("metrics") or {}
        bleu = _safe_float(metrics.get("bleu_char"))
        laal = _safe_float(metrics.get("laal_text"))

        write_indices = [i for i, action in enumerate(actions) if action == "WRITE"]
        read_indices = [i for i, action in enumerate(actions) if action == "READ"]

        total_emitted_chars = sum(len(chunk) for chunk in target_chunks if isinstance(chunk, str))
        last_chunk_chars = len(target_chunks[-1]) if target_chunks and isinstance(target_chunks[-1], str) else 0
        last_chunk_share = (last_chunk_chars / total_emitted_chars) if total_emitted_chars else 0.0

        chunk_rows = []
        cumulative = []
        running_text = ""
        for idx, (src, tgt, action) in enumerate(zip(source_chunks, target_chunks, actions), start=1):
            tgt = tgt or ""
            running_text += tgt
            cumulative.append(running_text)
            chunk_rows.append(
                {
                    "idx": idx,
                    "action": action,
                    "source": src or "",
                    "target": tgt,
                    "cumulative": running_text,
                }
            )

        task_name = path.parent.name if path.parent != experiment_dir else "root"
        cases.append(
            {
                "utt_id": obj.get("utt_id") or path.stem,
                "task": task_name,
                "path": str(path),
                "source_full_text": obj.get("source_full_text") or "",
                "prediction": obj.get("prediction") or "",
                "reference_text": obj.get("reference_text") or "",
                "bleu": bleu,
                "laal": laal,
                "chunk_count": len(actions),
                "write_count": len(write_indices),
                "read_count": len(read_indices),
                "first_write_chunk": (write_indices[0] + 1) if write_indices else None,
                "last_write_chunk": (write_indices[-1] + 1) if write_indices else None,
                "last_chunk_share": last_chunk_share,
                "chunk_rows": chunk_rows,
            }
        )
    return cases


def _summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    bleu_vals = [case["bleu"] for case in cases if case["bleu"] is not None]
    laal_vals = [case["laal"] for case in cases if case["laal"] is not None]
    return {
        "count": len(cases),
        "avg_bleu": mean(bleu_vals) if bleu_vals else None,
        "avg_laal": mean(laal_vals) if laal_vals else None,
        "min_bleu": min(bleu_vals) if bleu_vals else None,
        "max_bleu": max(bleu_vals) if bleu_vals else None,
        "min_laal": min(laal_vals) if laal_vals else None,
        "max_laal": max(laal_vals) if laal_vals else None,
    }


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def build_html(title: str, experiment_dir: Path, cases: List[Dict[str, Any]], initial_sort: str) -> str:
    summary = _summary(cases)
    payload = {
        "title": title,
        "experiment_dir": str(experiment_dir),
        "summary": summary,
        "cases": cases,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --bg-accent: #efe2cf;
      --panel: rgba(255, 251, 245, 0.86);
      --panel-strong: rgba(255, 248, 239, 0.98);
      --ink: #2f2419;
      --muted: #7a6755;
      --line: rgba(104, 75, 42, 0.16);
      --brand: #ab5f2c;
      --brand-2: #2f7a78;
      --good: #2f7a4f;
      --warn: #a55b19;
      --bad: #9e2f2f;
      --read: rgba(69, 105, 144, 0.10);
      --write: rgba(51, 124, 68, 0.10);
      --shadow: 0 14px 34px rgba(76, 43, 9, 0.10);
      --radius: 18px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.65), transparent 28%),
        radial-gradient(circle at top right, rgba(171,95,44,0.10), transparent 24%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 45%, #efe6d8 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }}

    .app {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }}

    .sidebar {{
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(248,240,229,0.95));
      backdrop-filter: blur(10px);
      padding: 24px 18px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }}

    .main {{
      padding: 26px;
    }}

    .hero {{
      padding: 18px 18px 20px;
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 6px);
      background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(246,233,212,0.88));
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}

    .hero h1 {{
      margin: 0 0 10px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}

    .summary-card, .panel {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    .summary-card {{
      padding: 14px 16px;
    }}

    .summary-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .summary-value {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      font-weight: 700;
    }}

    .toolbar {{
      display: grid;
      gap: 10px;
      margin: 20px 0 16px;
    }}

    .control {{
      display: grid;
      gap: 6px;
    }}

    label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.84);
      color: var(--ink);
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
      outline: none;
    }}

    input:focus, select:focus {{
      border-color: rgba(171,95,44,0.55);
      box-shadow: 0 0 0 3px rgba(171,95,44,0.12);
    }}

    .list-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}

    .case-list {{
      display: grid;
      gap: 10px;
      padding-bottom: 24px;
    }}

    .case-item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 14px 13px;
      background: rgba(255,255,255,0.78);
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }}

    .case-item:hover {{
      transform: translateY(-1px);
      border-color: rgba(171,95,44,0.35);
      box-shadow: 0 10px 20px rgba(76, 43, 9, 0.08);
    }}

    .case-item.active {{
      border-color: rgba(171,95,44,0.55);
      background: linear-gradient(135deg, rgba(255,252,248,1), rgba(245,233,215,1));
      box-shadow: 0 14px 26px rgba(76, 43, 9, 0.10);
    }}

    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
      margin-bottom: 8px;
    }}

    .case-id {{
      font-size: 15px;
      font-weight: 700;
      word-break: break-all;
    }}

    .case-task {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
    }}

    .metric-row, .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .metric-pill, .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 9px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}

    .metric-pill strong {{
      font-size: 12px;
    }}

    .page-top {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 18px;
    }}

    .nav {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
    }}

    button {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }}

    button:hover {{
      transform: translateY(-1px);
      border-color: rgba(171,95,44,0.45);
      box-shadow: 0 8px 18px rgba(76, 43, 9, 0.08);
    }}

    .counter {{
      color: var(--muted);
      font-size: 14px;
    }}

    .detail-grid {{
      display: grid;
      gap: 18px;
    }}

    .panel {{
      padding: 18px;
    }}

    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      font-weight: 700;
    }}

    .text-block {{
      display: grid;
      gap: 14px;
    }}

    .text-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.68);
    }}

    .text-card h3 {{
      margin: 0 0 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .text-card pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 15px/1.7 "IBM Plex Sans", "Avenir Next", sans-serif;
    }}

    .pred {{
      border-color: rgba(47, 122, 120, 0.20);
      background: rgba(237, 250, 249, 0.72);
    }}

    .ref {{
      border-color: rgba(171, 95, 44, 0.18);
      background: rgba(255, 247, 241, 0.72);
    }}

    .chunk-table-wrap {{
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 11px 12px;
      border-bottom: 1px solid rgba(104, 75, 42, 0.10);
      font-size: 13px;
      line-height: 1.5;
    }}

    th {{
      position: sticky;
      top: 0;
      background: rgba(248, 241, 232, 0.98);
      z-index: 1;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    tr.read-row {{
      background: var(--read);
    }}

    tr.write-row {{
      background: var(--write);
    }}

    .action-badge {{
      display: inline-flex;
      min-width: 62px;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}

    .action-read {{
      color: #325a7e;
      background: rgba(69, 105, 144, 0.12);
    }}

    .action-write {{
      color: #285837;
      background: rgba(51, 124, 68, 0.12);
    }}

    .muted {{
      color: var(--muted);
    }}

    .empty-state {{
      padding: 18px;
      border-radius: 16px;
      border: 1px dashed var(--line);
      background: rgba(255,255,255,0.5);
      color: var(--muted);
    }}

    .footer-note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }}

    @media (max-width: 1120px) {{
      .app {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <section class="hero">
        <h1>{title}</h1>
        <p>Manual review page for consensus-decoding outputs. Search, sort, and inspect one case at a time with chunk-level READ/WRITE traces.</p>
        <div class="summary-grid">
          <div class="summary-card">
            <span class="summary-label">Cases</span>
            <span class="summary-value">{summary["count"]}</span>
          </div>
          <div class="summary-card">
            <span class="summary-label">Avg BLEU</span>
            <span class="summary-value">{_fmt(summary["avg_bleu"])}</span>
          </div>
          <div class="summary-card">
            <span class="summary-label">Avg LAAL</span>
            <span class="summary-value">{_fmt(summary["avg_laal"])}</span>
          </div>
          <div class="summary-card">
            <span class="summary-label">BLEU Range</span>
            <span class="summary-value" style="font-size:18px">{_fmt(summary["min_bleu"])} .. {_fmt(summary["max_bleu"])}</span>
          </div>
        </div>
      </section>

      <div class="toolbar">
        <div class="control">
          <label for="searchInput">Search</label>
          <input id="searchInput" type="text" placeholder="utt_id / source / prediction / reference">
        </div>
        <div class="control">
          <label for="sortSelect">Sort</label>
          <select id="sortSelect">
            <option value="utt_asc">utt_id ↑</option>
            <option value="bleu_asc">BLEU low → high</option>
            <option value="bleu_desc">BLEU high → low</option>
            <option value="laal_desc">LAAL high → low</option>
            <option value="laal_asc">LAAL low → high</option>
            <option value="chunk_desc">Chunk count high → low</option>
            <option value="write_asc">WRITE count low → high</option>
          </select>
        </div>
      </div>

      <div id="listMeta" class="list-meta"></div>
      <div id="caseList" class="case-list"></div>
    </aside>

    <main class="main">
      <div class="page-top">
        <div>
          <div class="muted" id="experimentDir">{experiment_dir}</div>
          <div style="margin-top:4px;font-size:13px;color:var(--muted)">Tip: use <strong>J/K</strong> or the buttons to move through cases.</div>
        </div>
        <div class="nav">
          <button id="prevBtn" type="button">Previous</button>
          <div id="counter" class="counter"></div>
          <button id="nextBtn" type="button">Next</button>
        </div>
      </div>

      <div id="detailRoot" class="detail-grid"></div>
    </main>
  </div>

  <script>
    const APP_DATA = {data_json};

    const state = {{
      query: "",
      sortKey: {json.dumps(initial_sort, ensure_ascii=False)},
      filtered: APP_DATA.cases.slice(),
      selectedIndex: 0,
    }};

    const caseListEl = document.getElementById("caseList");
    const detailRootEl = document.getElementById("detailRoot");
    const listMetaEl = document.getElementById("listMeta");
    const counterEl = document.getElementById("counter");
    const searchInputEl = document.getElementById("searchInput");
    const sortSelectEl = document.getElementById("sortSelect");
    const prevBtnEl = document.getElementById("prevBtn");
    const nextBtnEl = document.getElementById("nextBtn");

    function fmt(value, digits = 4) {{
      return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "NA";
    }}

    function escapeHTML(value) {{
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function sortCases(cases) {{
      const arr = cases.slice();
      const score = (v, fallback) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);
      arr.sort((a, b) => {{
        switch (state.sortKey) {{
          case "bleu_asc":
            return score(a.bleu, Infinity) - score(b.bleu, Infinity) || a.utt_id.localeCompare(b.utt_id);
          case "bleu_desc":
            return score(b.bleu, -Infinity) - score(a.bleu, -Infinity) || a.utt_id.localeCompare(b.utt_id);
          case "laal_asc":
            return score(a.laal, Infinity) - score(b.laal, Infinity) || a.utt_id.localeCompare(b.utt_id);
          case "laal_desc":
            return score(b.laal, -Infinity) - score(a.laal, -Infinity) || a.utt_id.localeCompare(b.utt_id);
          case "chunk_desc":
            return b.chunk_count - a.chunk_count || a.utt_id.localeCompare(b.utt_id);
          case "write_asc":
            return a.write_count - b.write_count || a.utt_id.localeCompare(b.utt_id);
          case "utt_asc":
          default:
            return a.utt_id.localeCompare(b.utt_id);
        }}
      }});
      return arr;
    }}

    function filterCases() {{
      const q = state.query.trim().toLowerCase();
      let cases = APP_DATA.cases;
      if (q) {{
        cases = cases.filter((item) => {{
          const hay = [
            item.utt_id,
            item.task,
            item.source_full_text,
            item.prediction,
            item.reference_text,
          ].join("\\n").toLowerCase();
          return hay.includes(q);
        }});
      }}
      state.filtered = sortCases(cases);
      if (state.selectedIndex >= state.filtered.length) {{
        state.selectedIndex = Math.max(0, state.filtered.length - 1);
      }}
    }}

    function renderList() {{
      listMetaEl.textContent = `${{state.filtered.length}} / ${{APP_DATA.cases.length}} cases`;
      caseListEl.innerHTML = "";
      if (!state.filtered.length) {{
        caseListEl.innerHTML = '<div class="empty-state">No case matches the current search.</div>';
        return;
      }}

      state.filtered.forEach((item, idx) => {{
        const el = document.createElement("button");
        el.type = "button";
        el.className = "case-item" + (idx === state.selectedIndex ? " active" : "");
        el.innerHTML = `
          <div class="case-head">
            <div class="case-id">${{escapeHTML(item.utt_id)}}</div>
            <div class="case-task">${{escapeHTML(item.task)}}</div>
          </div>
          <div class="metric-row">
            <span class="metric-pill"><strong>BLEU</strong> ${{fmt(item.bleu)}}</span>
            <span class="metric-pill"><strong>LAAL</strong> ${{fmt(item.laal)}}</span>
            <span class="metric-pill"><strong>Chunks</strong> ${{item.chunk_count}}</span>
            <span class="metric-pill"><strong>WRITE</strong> ${{item.write_count}}</span>
          </div>
        `;
        el.addEventListener("click", () => {{
          state.selectedIndex = idx;
          render();
        }});
        caseListEl.appendChild(el);
      }});
    }}

    function renderDetail() {{
      detailRootEl.innerHTML = "";
      if (!state.filtered.length) {{
        detailRootEl.innerHTML = '<div class="empty-state">No case to display.</div>';
        counterEl.textContent = "0 / 0";
        return;
      }}

      const item = state.filtered[state.selectedIndex];
      counterEl.textContent = `${{state.selectedIndex + 1}} / ${{state.filtered.length}}`;

      const topPanel = document.createElement("section");
      topPanel.className = "panel";
      const trajectoryStrip = item.chunk_rows.map((row) => `
        <div title="Chunk ${{row.idx}}: ${{row.action}}" style="
          height: 12px;
          border-radius: 999px;
          flex: 1 1 0;
          min-width: 8px;
          background: ${{row.action === "WRITE" ? "linear-gradient(135deg, rgba(51,124,68,0.92), rgba(87,157,99,0.92))" : "linear-gradient(135deg, rgba(69,105,144,0.55), rgba(113,144,176,0.55))"}};
          border: 1px solid ${{row.action === "WRITE" ? "rgba(51,124,68,0.32)" : "rgba(69,105,144,0.22)"}};
        "></div>
      `).join("");
      topPanel.innerHTML = `
        <h2>${{escapeHTML(item.utt_id)}}</h2>
        <div class="chip-row">
          <span class="chip"><strong>Task</strong> ${{escapeHTML(item.task)}}</span>
          <span class="chip"><strong>BLEU</strong> ${{fmt(item.bleu)}}</span>
          <span class="chip"><strong>LAAL</strong> ${{fmt(item.laal)}}</span>
          <span class="chip"><strong>Chunks</strong> ${{item.chunk_count}}</span>
          <span class="chip"><strong>READ</strong> ${{item.read_count}}</span>
          <span class="chip"><strong>WRITE</strong> ${{item.write_count}}</span>
          <span class="chip"><strong>First WRITE</strong> ${{item.first_write_chunk ?? "NA"}}</span>
          <span class="chip"><strong>Last WRITE</strong> ${{item.last_write_chunk ?? "NA"}}</span>
          <span class="chip"><strong>Last chunk share</strong> ${{fmt(item.last_chunk_share * 100, 1)}}%</span>
        </div>
        <div style="margin-top:14px">
          <div class="muted" style="margin-bottom:8px;font-size:12px;text-transform:uppercase;letter-spacing:0.08em">Trajectory Strip</div>
          <div style="display:flex;gap:6px;align-items:center">${{trajectoryStrip}}</div>
        </div>
        <div class="footer-note">${{escapeHTML(item.path)}}</div>
      `;
      detailRootEl.appendChild(topPanel);

      const textPanel = document.createElement("section");
      textPanel.className = "panel";
      textPanel.innerHTML = `
        <h2>Translation View</h2>
        <div class="text-block">
          <div class="text-card">
            <h3>Source</h3>
            <pre>${{escapeHTML(item.source_full_text)}}</pre>
          </div>
          <div class="text-card pred">
            <h3>Prediction</h3>
            <pre>${{escapeHTML(item.prediction)}}</pre>
          </div>
          <div class="text-card ref">
            <h3>Reference</h3>
            <pre>${{escapeHTML(item.reference_text)}}</pre>
          </div>
        </div>
      `;
      detailRootEl.appendChild(textPanel);

      const chunkPanel = document.createElement("section");
      chunkPanel.className = "panel";
      const rows = item.chunk_rows.map((row) => `
        <tr class="${{row.action === "WRITE" ? "write-row" : "read-row"}}">
          <td>${{row.idx}}</td>
          <td><span class="action-badge ${{row.action === "WRITE" ? "action-write" : "action-read"}}">${{row.action}}</span></td>
          <td>${{escapeHTML(row.source || "")}}</td>
          <td>${{escapeHTML(row.target || "")}}</td>
          <td>${{escapeHTML(row.cumulative || "")}}</td>
        </tr>
      `).join("");
      chunkPanel.innerHTML = `
        <h2>Chunk Timeline</h2>
        <div class="chunk-table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Action</th>
                <th>Source Chunk</th>
                <th>Emitted Text</th>
                <th>Cumulative Prediction</th>
              </tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        </div>
      `;
      detailRootEl.appendChild(chunkPanel);
    }}

    function render() {{
      renderList();
      renderDetail();
      prevBtnEl.disabled = state.selectedIndex <= 0;
      nextBtnEl.disabled = state.selectedIndex >= state.filtered.length - 1;
    }}

    searchInputEl.addEventListener("input", (event) => {{
      state.query = event.target.value;
      filterCases();
      render();
    }});

    sortSelectEl.addEventListener("change", (event) => {{
      state.sortKey = event.target.value;
      filterCases();
      render();
    }});

    prevBtnEl.addEventListener("click", () => {{
      if (state.selectedIndex > 0) {{
        state.selectedIndex -= 1;
        render();
      }}
    }});

    nextBtnEl.addEventListener("click", () => {{
      if (state.selectedIndex < state.filtered.length - 1) {{
        state.selectedIndex += 1;
        render();
      }}
    }});

    document.addEventListener("keydown", (event) => {{
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {{
        return;
      }}
      if (event.key === "j" || event.key === "ArrowDown") {{
        if (state.selectedIndex < state.filtered.length - 1) {{
          state.selectedIndex += 1;
          render();
        }}
      }} else if (event.key === "k" || event.key === "ArrowUp") {{
        if (state.selectedIndex > 0) {{
          state.selectedIndex -= 1;
          render();
        }}
      }}
    }});

    filterCases();
    sortSelectEl.value = state.sortKey;
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML profiler for consensus decoding outputs.")
    parser.add_argument("--experiment-dir", required=True, help="Directory containing task_*/*.json outputs.")
    parser.add_argument("--output-html", default="", help="Path to write the HTML file.")
    parser.add_argument("--title", default="Consensus Profiling", help="Page title.")
    parser.add_argument("--bleu-max", type=float, default=None, help="Keep only cases with BLEU <= this value.")
    parser.add_argument("--bleu-min", type=float, default=None, help="Keep only cases with BLEU >= this value.")
    parser.add_argument("--limit", type=int, default=None, help="Keep only the first N cases after filtering.")
    parser.add_argument(
        "--initial-sort",
        default="utt_asc",
        choices=["utt_asc", "bleu_asc", "bleu_desc", "laal_asc", "laal_desc", "chunk_desc", "write_asc"],
        help="Initial sort key when the page loads.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    output_html = Path(args.output_html).resolve() if args.output_html else experiment_dir / "consensus_profile.html"
    cases = _load_cases(experiment_dir)
    if not cases:
        raise RuntimeError(f"No task_*/*.json files found under: {experiment_dir}")

    if args.bleu_max is not None:
        cases = [case for case in cases if case["bleu"] is not None and case["bleu"] <= args.bleu_max]
    if args.bleu_min is not None:
        cases = [case for case in cases if case["bleu"] is not None and case["bleu"] >= args.bleu_min]
    if args.limit is not None:
        cases = sorted(
            cases,
            key=lambda case: (case["bleu"] if case["bleu"] is not None else float("inf"), case["utt_id"]),
        )[: args.limit]
    if not cases:
        raise RuntimeError("No cases remained after filtering.")

    html = build_html(args.title, experiment_dir, cases, args.initial_sort)
    output_html.write_text(html, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
