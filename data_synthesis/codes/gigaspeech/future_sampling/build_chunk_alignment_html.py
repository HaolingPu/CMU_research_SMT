#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Optional

from compare_metricx_qe import load_metricx_scores


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 251, 245, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --ink: #1e293b;
      --muted: #667085;
      --line: #ddcfbf;
      --line-strong: #c7b39a;
      --accent: #0f766e;
      --accent-2: #9a3412;
      --accent-soft: rgba(15, 118, 110, 0.10);
      --warn-soft: rgba(154, 52, 18, 0.10);
      --shadow: 0 18px 48px rgba(63, 44, 20, 0.10);
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 14px;
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      min-height: 100%;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 244, 214, 0.95), transparent 32%),
        radial-gradient(circle at 85% 10%, rgba(214, 242, 235, 0.92), transparent 26%),
        linear-gradient(180deg, #f7f1e7 0%, #efe7d7 100%);
      font-family: "Avenir Next", "Segoe UI", "Noto Sans", sans-serif;
    }}

    body {{
      padding: 14px;
    }}

    .app {{
      max-width: 1560px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}

    .sidebar,
    .main {{
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(199, 179, 154, 0.7);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }}

    .sidebar {{
      position: sticky;
      top: 24px;
      padding: 16px;
      max-height: calc(100vh - 28px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .main {{
      padding: 16px;
      min-height: calc(100vh - 28px);
    }}

    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 4px 0 8px;
      font-size: clamp(24px, 2.7vw, 34px);
      line-height: 1.02;
      font-weight: 800;
    }}

    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}

    .control-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}

    .control-row {{
      display: grid;
      grid-template-columns: 1fr 132px 96px;
      gap: 8px;
    }}

    label.small {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}

    input,
    select,
    button {{
      font: inherit;
    }}

    input,
    select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink);
    }}

    button {{
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }}

    button:hover {{
      transform: translateY(-1px);
      border-color: var(--line-strong);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}

    .stat-card {{
      padding: 10px 12px;
      border-radius: 14px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}

    .stat-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .stat-value {{
      margin-top: 4px;
      font-size: 18px;
      font-weight: 800;
    }}

    .case-list {{
      overflow: auto;
      padding-right: 4px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .case-item {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid transparent;
      background: rgba(255, 255, 255, 0.80);
      text-align: left;
    }}

    .case-item.active {{
      border-color: rgba(15, 118, 110, 0.35);
      background: linear-gradient(180deg, rgba(224, 248, 244, 0.98), rgba(255, 255, 255, 0.98));
      box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.12);
    }}

    .case-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }}

    .case-id {{
      font-size: 13px;
      font-weight: 800;
      word-break: break-word;
    }}

    .case-rank {{
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}

    .mini-metrics {{
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
    }}

    .pill.metricx {{
      background: var(--warn-soft);
      border-color: rgba(154, 52, 18, 0.22);
      color: var(--accent-2);
    }}

    .pill.good {{
      background: var(--accent-soft);
      border-color: rgba(15, 118, 110, 0.18);
      color: var(--accent);
    }}

    .main-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}

    .main-title h2 {{
      margin: 4px 0 6px;
      font-size: clamp(20px, 2.3vw, 28px);
      line-height: 1.08;
    }}

    .main-title p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      max-width: 880px;
    }}

    .nav-buttons {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}

    .summary-card {{
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}

    .summary-card .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .summary-card .value {{
      margin-top: 4px;
      font-size: 20px;
      font-weight: 800;
    }}

    .tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}

    .tab-btn {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      font-weight: 700;
    }}

    .tab-btn.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}

    .tab-panel {{
      display: none;
    }}

    .tab-panel.active {{
      display: block;
    }}

    .chunk-table-wrap {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: var(--panel-strong);
    }}

    .chunk-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}

    .chunk-table thead th {{
      padding: 8px 10px;
      text-align: left;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      background: linear-gradient(180deg, rgba(253, 248, 239, 0.98), rgba(250, 244, 233, 0.98));
      border-bottom: 1px solid var(--line);
    }}

    .chunk-table tbody td {{
      padding: 8px 10px;
      vertical-align: top;
      border-top: 1px solid rgba(221, 207, 191, 0.72);
      line-height: 1.4;
      font-size: 14px;
      word-break: break-word;
    }}

    .chunk-table tbody tr:nth-child(even) td {{
      background: rgba(248, 252, 252, 0.55);
    }}

    .chunk-num {{
      width: 92px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .chunk-source {{
      width: 42%;
    }}

    .chunk-target {{
      width: 42%;
    }}

    .chunk-action {{
      display: inline-block;
      margin-left: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(15, 118, 110, 0.18);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      vertical-align: middle;
    }}

    .chunk-action.warn {{
      background: var(--warn-soft);
      border-color: rgba(154, 52, 18, 0.18);
      color: var(--accent-2);
    }}

    .chunk-label {{
      display: none;
    }}

    .chunk-text {{
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-size: 14px;
    }}

    .empty {{
      color: #a19587;
      font-style: italic;
    }}

    .text-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}

    .text-panel {{
      padding: 12px;
      border-radius: 14px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}

    .text-panel h3 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}

    .text-panel .body {{
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
      color: var(--ink);
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}

    .meta-box {{
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}

    .meta-box h3 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}

    .keyvals {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px 12px;
      align-items: start;
      line-height: 1.45;
    }}

    .key {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .value {{
      word-break: break-word;
    }}

    .hint {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}

    @media (max-width: 1280px) {{
      .app {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
        max-height: none;
      }}
    }}

    @media (max-width: 860px) {{
      body {{
        padding: 12px;
      }}

      .control-row,
      .summary-strip,
      .meta-grid,
      .chunk-grid,
      .stats {{
        grid-template-columns: 1fr;
      }}

      .main-head {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div>
        <div class="eyebrow">Chunk Review</div>
        <h1>{title}</h1>
        <p class="sub">{subtitle}</p>
      </div>

      <div class="control-grid">
        <div>
          <label class="small" for="searchInput">Search</label>
          <input id="searchInput" placeholder="Search utt_id">
        </div>
        <div class="control-row">
          <div>
            <label class="small" for="sortKey">Sort By</label>
            <select id="sortKey">
              <option value="metricx_qe">MetricX QE</option>
              <option value="bleu_char">BLEU</option>
              <option value="laal_text">LAAL</option>
              <option value="utt_id">utt_id</option>
            </select>
          </div>
          <div>
            <label class="small" for="sortDir">Order</label>
            <select id="sortDir">
              <option value="asc">Asc</option>
              <option value="desc">Desc</option>
            </select>
          </div>
          <div>
            <label class="small" for="jumpInput">Jump</label>
            <input id="jumpInput" placeholder="#">
          </div>
        </div>
      </div>

      <div class="stats">
        <div class="stat-card">
          <div class="stat-label">Cases</div>
          <div class="stat-value" id="statCount">{count}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">QE Ready</div>
          <div class="stat-value" id="statQeReady">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Best QE</div>
          <div class="stat-value" id="statQeBest">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg BLEU</div>
          <div class="stat-value" id="statBleu">-</div>
        </div>
      </div>

      <div class="case-list" id="caseList"></div>
    </aside>

    <main class="main">
      <div class="main-head">
        <div class="main-title">
          <div class="eyebrow">Detailed Review</div>
          <h2 id="detailTitle">-</h2>
          <p id="detailSub">Source chunk on the left, target chunk on the right. Use the sidebar to sort by QE and inspect difficult cases first.</p>
        </div>
        <div class="nav-buttons">
          <button id="prevBtn" type="button">Previous</button>
          <button id="nextBtn" type="button">Next</button>
        </div>
      </div>

      <section class="summary-strip">
        <div class="summary-card">
          <div class="label">MetricX QE</div>
          <div class="value" id="metricQe">-</div>
        </div>
        <div class="summary-card">
          <div class="label">BLEU</div>
          <div class="value" id="metricBleu">-</div>
        </div>
        <div class="summary-card">
          <div class="label">LAAL</div>
          <div class="value" id="metricLaal">-</div>
        </div>
        <div class="summary-card">
          <div class="label">Chunks</div>
          <div class="value" id="metricChunks">-</div>
        </div>
      </section>

      <div class="tabs">
        <button class="tab-btn active" type="button" data-tab="chunks">Chunk Pairs</button>
        <button class="tab-btn" type="button" data-tab="texts">Full Text</button>
        <button class="tab-btn" type="button" data-tab="meta">Meta</button>
      </div>

      <section class="tab-panel active" id="tab-chunks">
        <div class="chunk-table-wrap">
          <table class="chunk-table">
            <thead>
              <tr>
                <th>Chunk</th>
                <th>Source</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody id="chunkStack"></tbody>
          </table>
        </div>
      </section>

      <section class="tab-panel" id="tab-texts">
        <div class="text-grid">
          <div class="text-panel">
            <h3>Full Source</h3>
            <div class="body" id="fullSource"></div>
          </div>
          <div class="text-panel">
            <h3>Prediction</h3>
            <div class="body" id="prediction"></div>
          </div>
          <div class="text-panel">
            <h3>Reference</h3>
            <div class="body" id="reference"></div>
          </div>
        </div>
      </section>

      <section class="tab-panel" id="tab-meta">
        <div class="meta-grid">
          <div class="meta-box">
            <h3>Case Metadata</h3>
            <div class="keyvals" id="metaKeyvals"></div>
          </div>
          <div class="meta-box">
            <h3>Reading Hints</h3>
            <div class="hint">
              Lower MetricX QE is better. BLEU is higher-is-better. LAAL is latency-oriented, so lower values usually mean earlier commitment. Start with low-QE samples if you want to catch semantic errors first.
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const DATA = {data_json};

    const state = {{
      search: "",
      sortKey: "{default_sort_key}",
      sortDir: "{default_sort_dir}",
      selectedId: DATA.length ? DATA[0].utt_id : null,
      activeTab: "chunks",
    }};

    const searchInput = document.getElementById("searchInput");
    const sortKeyEl = document.getElementById("sortKey");
    const sortDirEl = document.getElementById("sortDir");
    const jumpInput = document.getElementById("jumpInput");
    const caseList = document.getElementById("caseList");
    const detailTitle = document.getElementById("detailTitle");
    const detailSub = document.getElementById("detailSub");
    const metricQe = document.getElementById("metricQe");
    const metricBleu = document.getElementById("metricBleu");
    const metricLaal = document.getElementById("metricLaal");
    const metricChunks = document.getElementById("metricChunks");
    const chunkStack = document.getElementById("chunkStack");
    const fullSource = document.getElementById("fullSource");
    const prediction = document.getElementById("prediction");
    const reference = document.getElementById("reference");
    const metaKeyvals = document.getElementById("metaKeyvals");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const statQeReady = document.getElementById("statQeReady");
    const statQeBest = document.getElementById("statQeBest");
    const statBleu = document.getElementById("statBleu");

    function escapeHtml(text) {{
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function numericOrNull(value) {{
      if (value === null || value === undefined || value === "") return null;
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }}

    function fmtNumber(value, digits = 4) {{
      const n = numericOrNull(value);
      return n === null ? "—" : n.toFixed(digits);
    }}

    function getMetric(item, key) {{
      if (key === "metricx_qe") return numericOrNull(item.metricx_qe);
      if (key === "utt_id") return item.utt_id;
      return numericOrNull(item.metrics?.[key]);
    }}

    function compareItems(a, b) {{
      const key = state.sortKey;
      const dir = state.sortDir === "desc" ? -1 : 1;
      const av = getMetric(a, key);
      const bv = getMetric(b, key);

      if (key === "utt_id") {{
        return a.utt_id.localeCompare(b.utt_id) * dir;
      }}

      const aBad = !Number.isFinite(av);
      const bBad = !Number.isFinite(bv);
      if (aBad && bBad) return a.utt_id.localeCompare(b.utt_id);
      if (aBad) return 1;
      if (bBad) return -1;
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return a.utt_id.localeCompare(b.utt_id);
    }}

    function getFiltered() {{
      const q = state.search.trim().toLowerCase();
      return DATA
        .filter(item => !q || item.utt_id.toLowerCase().includes(q))
        .slice()
        .sort(compareItems);
    }}

    function pickSelected(filtered) {{
      if (!filtered.length) return null;
      const found = filtered.find(item => item.utt_id === state.selectedId);
      return found || filtered[0];
    }}

    function renderSidebar(filtered, selected) {{
      caseList.innerHTML = filtered.map((item, idx) => {{
        const active = selected && item.utt_id === selected.utt_id ? " active" : "";
        return `
          <button class="case-item${{active}}" type="button" data-utt="${{escapeHtml(item.utt_id)}}">
            <div class="case-top">
              <div class="case-id">${{escapeHtml(item.utt_id)}}</div>
              <div class="case-rank">#${{idx + 1}}</div>
            </div>
            <div class="mini-metrics">
              <span class="pill metricx">QE ${{fmtNumber(item.metricx_qe, 3)}}</span>
              <span class="pill good">BLEU ${{fmtNumber(item.metrics?.bleu_char, 2)}}</span>
              <span class="pill">LAAL ${{fmtNumber(item.metrics?.laal_text, 2)}}</span>
            </div>
          </button>
        `;
      }}).join("");

      caseList.querySelectorAll(".case-item").forEach(btn => {{
        btn.addEventListener("click", () => {{
          state.selectedId = btn.dataset.utt;
          render();
        }});
      }});
    }}

    function renderChunks(item) {{
      const src = item.src_trajectory || [];
      const tgt = item.target_trajectory || [];
      const actions = item.actions || [];
      const total = Math.max(src.length, tgt.length, actions.length);
      const rows = [];

      for (let i = 0; i < total; i++) {{
        const action = String(actions[i] ?? "").trim();
        const badgeClass = /write|commit/i.test(action) ? "chunk-action warn" : "chunk-action";
        rows.push(`
          <tr>
            <td class="chunk-num">
              ${{i + 1}}
              ${{action ? `<span class="${{badgeClass}}">${{escapeHtml(action)}}</span>` : ""}}
            </td>
            <td class="chunk-source">
              <div class="chunk-text">${{src[i] ? escapeHtml(src[i]) : '<span class="empty">(empty)</span>'}}</div>
            </td>
            <td class="chunk-target">
              <div class="chunk-text">${{tgt[i] ? escapeHtml(tgt[i]) : '<span class="empty">(empty)</span>'}}</div>
            </td>
          </tr>
        `);
      }}
      chunkStack.innerHTML = rows.join("");
    }}

    function renderMeta(item, rank, filteredLength) {{
      metaKeyvals.innerHTML = [
        ["utt_id", item.utt_id],
        ["Rank In Current View", `${{rank + 1}} / ${{filteredLength}}`],
        ["MetricX QE", fmtNumber(item.metricx_qe, 4)],
        ["BLEU", fmtNumber(item.metrics?.bleu_char, 4)],
        ["LAAL", fmtNumber(item.metrics?.laal_text, 4)],
        ["Chunk Count", String(Math.max(item.src_trajectory?.length || 0, item.target_trajectory?.length || 0, item.actions?.length || 0))],
        ["Decoder", item.decoder_impl || "—"],
      ].map(([key, value]) => `
        <div class="key">${{escapeHtml(key)}}</div>
        <div class="value">${{escapeHtml(value)}}</div>
      `).join("");
    }}

    function renderDetail(item, filtered) {{
      const rank = filtered.findIndex(row => row.utt_id === item.utt_id);
      detailTitle.textContent = item.utt_id;
      detailSub.textContent = `Current rank ${{rank + 1}} of ${{filtered.length}} in this filtered view. Lower QE is better; chunk review uses 3 columns: chunk, source, target.`;

      metricQe.textContent = fmtNumber(item.metricx_qe, 4);
      metricBleu.textContent = fmtNumber(item.metrics?.bleu_char, 4);
      metricLaal.textContent = fmtNumber(item.metrics?.laal_text, 4);
      metricChunks.textContent = String(Math.max(item.src_trajectory?.length || 0, item.target_trajectory?.length || 0, item.actions?.length || 0));

      fullSource.textContent = item.source_full_text || "";
      prediction.textContent = item.prediction || "";
      reference.textContent = item.reference_text || "";

      renderChunks(item);
      renderMeta(item, rank, filtered.length);
      window.location.hash = item.utt_id;
    }}

    function renderStats() {{
      const qeVals = DATA.map(item => numericOrNull(item.metricx_qe)).filter(v => v !== null);
      const bleuVals = DATA.map(item => numericOrNull(item.metrics?.bleu_char)).filter(v => v !== null);
      statQeReady.textContent = `${{qeVals.length}} / ${{DATA.length}}`;
      statQeBest.textContent = qeVals.length ? Math.min(...qeVals).toFixed(3) : "—";
      if (bleuVals.length) {{
        const avg = bleuVals.reduce((a, b) => a + b, 0) / bleuVals.length;
        statBleu.textContent = avg.toFixed(2);
      }} else {{
        statBleu.textContent = "—";
      }}
    }}

    function render() {{
      sortKeyEl.value = state.sortKey;
      sortDirEl.value = state.sortDir;
      searchInput.value = state.search;

      const filtered = getFiltered();
      const selected = pickSelected(filtered);
      state.selectedId = selected ? selected.utt_id : null;

      renderSidebar(filtered, selected);
      if (selected) renderDetail(selected, filtered);
      else {{
        detailTitle.textContent = "No matching case";
        detailSub.textContent = "Try a different search or sort order.";
        chunkStack.innerHTML = "";
        fullSource.textContent = "";
        prediction.textContent = "";
        reference.textContent = "";
        metaKeyvals.innerHTML = "";
      }}

      prevBtn.disabled = !selected || filtered.findIndex(row => row.utt_id === selected.utt_id) <= 0;
      nextBtn.disabled = !selected || filtered.findIndex(row => row.utt_id === selected.utt_id) >= filtered.length - 1;
    }}

    function stepSelection(offset) {{
      const filtered = getFiltered();
      const idx = filtered.findIndex(item => item.utt_id === state.selectedId);
      if (idx < 0) return;
      const next = filtered[idx + offset];
      if (!next) return;
      state.selectedId = next.utt_id;
      render();
    }}

    document.querySelectorAll(".tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        state.activeTab = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach(node => node.classList.toggle("active", node === btn));
        document.querySelectorAll(".tab-panel").forEach(panel => {{
          panel.classList.toggle("active", panel.id === `tab-${{state.activeTab}}`);
        }});
      }});
    }});

    searchInput.addEventListener("input", () => {{
      state.search = searchInput.value;
      render();
    }});

    sortKeyEl.addEventListener("change", () => {{
      state.sortKey = sortKeyEl.value;
      if (state.sortKey === "metricx_qe" || state.sortKey === "laal_text") {{
        state.sortDir = "asc";
      }} else if (state.sortKey === "bleu_char") {{
        state.sortDir = "desc";
      }}
      render();
    }});

    sortDirEl.addEventListener("change", () => {{
      state.sortDir = sortDirEl.value;
      render();
    }});

    jumpInput.addEventListener("change", () => {{
      const filtered = getFiltered();
      const n = Number(jumpInput.value);
      if (Number.isInteger(n) && n >= 1 && n <= filtered.length) {{
        state.selectedId = filtered[n - 1].utt_id;
        render();
      }}
    }});

    prevBtn.addEventListener("click", () => stepSelection(-1));
    nextBtn.addEventListener("click", () => stepSelection(1));

    renderStats();
    const initialHash = (window.location.hash || "").replace(/^#/, "");
    if (initialHash && DATA.some(item => item.utt_id === initialHash)) {{
      state.selectedId = initialHash;
    }}
    render();
  </script>
</body>
</html>
"""


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _load_metricx_scores(metricx_dir: Optional[str]) -> Dict[str, float]:
    if not metricx_dir:
        return {}
    scores, _ = load_metricx_scores(metricx_dir, "future_sampling")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for chunk-aligned JSON outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--title", default="Chunk Alignment Review")
    parser.add_argument("--metricx-dir", default="", help="Optional MetricX directory used to inject per-utt QE.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    qe_scores = _load_metricx_scores(args.metricx_dir or None)
    data = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            continue
        utt_id = str(obj.get("utt_id", "")).strip()
        if not utt_id:
            continue
        metrics = obj.get("metrics") or {}
        obj["metrics"] = metrics
        obj["metricx_qe"] = _safe_float(qe_scores.get(utt_id))
        data.append(obj)

    if not data:
        raise SystemExit(f"No utterance JSON files found in {input_dir}")

    qe_ready = sum(1 for item in data if item.get("metricx_qe") is not None)
    subtitle = (
        f"Directory: {input_dir} | Cases: {len(data)} | MetricX QE: "
        f"{qe_ready}/{len(data)} ready"
    )
    default_sort_key = "metricx_qe" if qe_ready else "utt_id"
    default_sort_dir = "asc" if qe_ready else "asc"

    html_text = PAGE_TEMPLATE.format(
        title=html.escape(args.title),
        subtitle=html.escape(subtitle),
        count=len(data),
        data_json=json.dumps(data, ensure_ascii=False),
        default_sort_key=default_sort_key,
        default_sort_dir=default_sort_dir,
    )

    out_path = Path(args.output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
