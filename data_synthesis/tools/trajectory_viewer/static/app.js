const state = {
  payload: null,
  filtered: [],
  caseIndex: 0,
  timer: null,
};

const els = {
  runSummary: document.querySelector("#run-summary"),
  caseList: document.querySelector("#case-list"),
  caseCount: document.querySelector("#case-count"),
  caseSearch: document.querySelector("#case-search"),
  clearSearch: document.querySelector("#clear-search"),
  previousCase: document.querySelector("#previous-case"),
  nextCase: document.querySelector("#next-case"),
  playTrajectory: document.querySelector("#play-trajectory"),
  currentCaseLabel: document.querySelector("#current-case-label"),
  casePosition: document.querySelector("#case-position"),
  caseId: document.querySelector("#case-id"),
  sourceFull: document.querySelector("#source-full"),
  metricGrid: document.querySelector("#metric-grid"),
  audioPlayer: document.querySelector("#audio-player"),
  audioStatus: document.querySelector("#audio-status"),
  trajectory: document.querySelector("#trajectory"),
  finalPrediction: document.querySelector("#final-prediction"),
  referenceText: document.querySelector("#reference-text"),
  showCumulative: document.querySelector("#show-cumulative"),
  stepTemplate: document.querySelector("#step-template"),
};

function formatMetric(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "N/A";
}

function selectedFutureCount(step) {
  return (step.selected_futures || []).reduce(
    (total, group) => total + (group.candidates || []).length,
    0,
  );
}

function metricCard(label, value) {
  const card = document.createElement("div");
  card.className = "metric";
  const name = document.createElement("span");
  name.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  card.append(name, strong);
  return card;
}

function renderMetrics(item) {
  const cards = [
    metricCard("CHAR BLEU", formatMetric(item.metrics.bleu_char)),
    metricCard("LAAL", formatMetric(item.metrics.laal_text, 2)),
    metricCard("WRITE STEPS", String(item.write_steps)),
    metricCard("TOTAL STEPS", String(item.steps.length)),
  ];
  if (Number.isFinite(item.consensus_steps)) {
    cards.push(metricCard("CONSENSUS STEPS", String(item.consensus_steps)));
    cards.push(metricCard("HORIZON DROPS", String(item.horizon_drops ?? 0)));
  }
  els.metricGrid.replaceChildren(...cards);
}

// ---------------------------------------------------------------------------
// Next-token consensus panel (lazy-loaded from data/consensus/<utt>.json)
// ---------------------------------------------------------------------------

const detailCache = new Map();

async function loadDetail(item) {
  if (!item.detail_url) return null;
  if (detailCache.has(item.utt_id)) return detailCache.get(item.utt_id);
  const promise = fetch(item.detail_url)
    .then((response) => (response.ok ? response.json() : null))
    .catch(() => null);
  detailCache.set(item.utt_id, promise);
  return promise;
}

function chip(token, prob, classes) {
  const span = document.createElement("span");
  span.className = `chip ${classes || ""}`.trim();
  const tok = document.createElement("b");
  tok.textContent = token === "" ? "∅" : token.replace(/\n/g, "⏎");
  const p = document.createElement("i");
  p.textContent = Number.isFinite(prob) ? prob.toFixed(prob >= 0.01 ? 2 : 3) : "";
  span.append(tok, p);
  span.title = `${JSON.stringify(token)} p=${prob}`;
  return span;
}

function futureRows(consStep, futures, highlight, intersection) {
  const wrap = document.createElement("div");
  wrap.className = "fut-rows";
  const inter = new Set(intersection || []);
  for (const f of consStep.futures || []) {
    const meta = futures[f.i] || {};
    const row = document.createElement("div");
    row.className = "fut-row";
    const head = document.createElement("div");
    head.className = "fut-meta";
    const idx = document.createElement("span");
    idx.className = "fut-idx";
    idx.textContent = `f${f.i}`;
    const tag = document.createElement("span");
    tag.className = `fut-tag ${meta.mode === "contrastive" ? "fut-contrastive" : "fut-plausible"}`;
    tag.textContent = `${meta.label || "?"} · ${meta.mode || "?"}`;
    const text = document.createElement("span");
    text.className = "fut-text";
    text.textContent = meta.text || "(future text unavailable)";
    text.title = meta.text || "";
    head.append(idx, tag, text);
    const chips = document.createElement("div");
    chips.className = "chips";
    const top1 = f.top && f.top.length ? f.top[0][0] : null;
    for (const [token, prob] of f.top || []) {
      const classes = [];
      if (highlight !== null && token === highlight) classes.push("hit");
      if (inter.has(token)) classes.push("inter");
      if (token === top1) classes.push("top1");
      chips.append(chip(token, prob, classes.join(" ")));
    }
    if (highlight !== null && !(f.top || []).some(([token]) => token === highlight)) {
      const miss = document.createElement("span");
      miss.className = "chip miss";
      miss.textContent = `${highlight} not in top-${(f.top || []).length}`;
      chips.append(miss);
    }
    row.append(head, chips);
    wrap.append(row);
  }
  return wrap;
}

function consensusStepElement(consStep, futures) {
  const details = document.createElement("details");
  details.className = `cons-step ${consStep.kind === "accepted" ? "cons-accepted" : "cons-stop"}`;
  const summary = document.createElement("summary");
  const k = document.createElement("span");
  k.className = "cons-k";
  k.textContent = consStep.k === "filter" ? "filter" : `step ${consStep.k}`;
  summary.append(k);
  let highlight = null;
  let intersection = [];
  if (consStep.kind === "accepted") {
    highlight = consStep.token;
    const tok = document.createElement("span");
    tok.className = "cons-token";
    tok.textContent = consStep.token;
    const stat = document.createElement("span");
    stat.className = "cons-stat";
    const n = consStep.n_futures || (consStep.futures || []).length;
    stat.textContent = n
      ? `top-1 in ${consStep.top1_agree}/${n} futures · mean p ${formatMetric(consStep.mean_p, 2)} · min p ${formatMetric(consStep.min_p, 3)}`
      : "";
    summary.append(document.createTextNode(" accepted "), tok, stat);
  } else {
    intersection = consStep.intersection || [];
    const stop = document.createElement("span");
    stop.className = "cons-stopreason";
    stop.textContent = `STOP · ${consStep.stop_reason || "?"}`;
    const stat = document.createElement("span");
    stat.className = "cons-stat";
    if (consStep.k === "filter") {
      stat.textContent = `pending ${JSON.stringify(consStep.pending || "")}`;
    } else {
      stat.textContent = `${consStep.intersection_size ?? intersection.length} candidates in the union, none in every future's top-k`;
    }
    summary.append(document.createTextNode(" "), stop, stat);
  }
  details.append(summary);
  const body = document.createElement("div");
  body.className = "cons-body";
  if (intersection.length) {
    const inter = document.createElement("div");
    inter.className = "inter-list";
    const label = document.createElement("span");
    label.className = "inter-label";
    label.textContent = "candidate union:";
    inter.append(label);
    for (const token of intersection) inter.append(chip(token, NaN, "inter"));
    body.append(inter);
  }
  let rendered = false;
  details.addEventListener("toggle", () => {
    if (!details.open || rendered) return;
    rendered = true;
    if ((consStep.futures || []).length) body.append(futureRows(consStep, futures, highlight, intersection));
  });
  details.append(body);
  return details;
}

function consensusPanel(step, detailStep) {
  const summaryData = step.consensus_summary;
  if (!summaryData && !detailStep) return null;
  const panel = document.createElement("details");
  panel.className = "consensus-details";
  const summary = document.createElement("summary");
  const parts = [];
  if (summaryData) {
    if (summaryData.final_completion) parts.push("final chunk: translator completes without consensus");
    else if (summaryData.too_few_futures) parts.push("READ: too few futures to vote");
    else {
      parts.push(`next-token consensus: ${summaryData.accepted.length} accepted`);
      if (summaryData.stop_reason) parts.push(`stop: ${summaryData.stop_reason}`);
      if (summaryData.horizon_dropped) parts.push(`horizon filter dropped "${summaryData.horizon_dropped}"`);
    }
    if (Number.isFinite(summaryData.raw_total)) parts.push(`raw futures ${summaryData.raw_accepted}/${summaryData.raw_total} kept`);
  }
  summary.textContent = parts.join(" · ");
  panel.append(summary);

  const body = document.createElement("div");
  body.className = "consensus-body";
  if (summaryData && summaryData.accepted.length) {
    const line = document.createElement("p");
    line.className = "cons-line";
    line.append(document.createTextNode("accepted sequence: "));
    for (const token of summaryData.accepted) line.append(chip(token, NaN, "hit"));
    body.append(line);
  }
  for (const group of step.raw_stats || []) {
    if (group.dropped === undefined) continue;
    const line = document.createElement("p");
    line.className = "cons-line muted";
    line.textContent = `${group.label} / ${group.mode}: kept ${group.kept}/${group.requested}${group.dropped && group.dropped !== "none" ? ` (dropped ${group.dropped})` : ""}`;
    body.append(line);
  }
  if (detailStep) {
    const futures = detailStep.futures || [];
    for (const consStep of detailStep.consensus || []) body.append(consensusStepElement(consStep, futures));
    if (detailStep.horizon) {
      const line = document.createElement("p");
      line.className = "cons-line warn";
      line.textContent = `horizon filter: pending "${detailStep.horizon.dropped}" has ${detailStep.horizon.len} token(s) < min_horizon ${detailStep.horizon.min_horizon} → discarded → READ`;
      body.append(line);
    }
    if (detailStep.pending_before_trim !== null && detailStep.pending_before_trim !== undefined
        && detailStep.pending_before_trim !== detailStep.commit_after_trim) {
      const line = document.createElement("p");
      line.className = "cons-line warn";
      line.textContent = `byte-boundary trim: pending "${detailStep.pending_before_trim}" → committed "${detailStep.commit_after_trim}"`;
      body.append(line);
    }
  } else {
    const line = document.createElement("p");
    line.className = "cons-line muted";
    line.textContent = "Loading per-future distributions…";
    body.append(line);
  }
  panel.append(body);
  return panel;
}

function attachConsensus(item, detail) {
  const rows = document.querySelectorAll(".trajectory-row");
  rows.forEach((row) => {
    const index = Number(row.dataset.step);
    const step = item.steps[index];
    const detailStep = detail ? detail.steps[index] : null;
    row.querySelector(".consensus-details")?.remove();
    const panel = consensusPanel(step, detailStep);
    if (panel) row.querySelector(".translation-cell").append(panel);
  });
}

function futureGroupElement(group) {
  const section = document.createElement("section");
  section.className = "future-group";
  const title = document.createElement("h4");
  const model = group.label || group.model || "Sampler";
  title.textContent = `${model} / ${group.mode}`;
  const list = document.createElement("ol");
  for (const candidate of group.candidates || []) {
    const item = document.createElement("li");
    item.textContent = candidate;
    list.append(item);
  }
  section.append(title, list);
  return section;
}

function renderStep(step, index) {
  const fragment = els.stepTemplate.content.cloneNode(true);
  const row = fragment.querySelector(".trajectory-row");
  row.dataset.step = String(index);
  row.style.animationDelay = `${Math.min(index * 26, 420)}ms`;
  fragment.querySelector(".step-number").textContent = `STEP ${String(step.step).padStart(2, "0")}`;

  const sourceChunk = fragment.querySelector(".source-chunk");
  sourceChunk.textContent = step.source_chunk.trim() || "No new source audio text";
  if (!step.source_chunk.trim()) sourceChunk.classList.add("empty-chunk");
  fragment.querySelector(".source-cumulative").textContent = step.source_cumulative || "-";

  const action = fragment.querySelector(".action-pill");
  action.textContent = step.action;
  action.classList.add(step.action === "WRITE" ? "action-write" : "action-read");
  const delta = fragment.querySelector(".translation-delta");
  delta.textContent = step.translation_delta || "Wait for more source context";
  if (!step.translation_delta) delta.classList.add("empty-chunk");
  fragment.querySelector(".translation-cumulative").textContent = step.translation_cumulative || "-";

  const details = fragment.querySelector(".future-details");
  const futureCount = selectedFutureCount(step);
  if (!futureCount) {
    details.remove();
  } else {
    const rawCount = (step.raw_stats || []).reduce((sum, group) => sum + (group.requested || 0), 0);
    details.querySelector("summary").textContent = `${futureCount} futures used for consensus${rawCount ? ` / ${rawCount} raw` : ""}`;
    const prefix = details.querySelector(".future-prefix");
    prefix.textContent = `Future source prefix: ${step.future_source_prefix || step.source_cumulative}`;
    const groups = details.querySelector(".future-groups");
    for (const group of step.selected_futures || []) groups.append(futureGroupElement(group));
  }
  return fragment;
}

function renderCase() {
  stopPlayback();
  const item = state.filtered[state.caseIndex];
  if (!item) return;
  const totalCases = state.payload.cases.length;
  els.casePosition.textContent = `CASE ${item.display_order} OF ${totalCases} / ${item.task}`;
  els.currentCaseLabel.textContent = `${String(item.display_order).padStart(3, "0")} / ${item.utt_id}`;
  els.caseId.textContent = item.utt_id;
  els.sourceFull.textContent = item.source_full_text;
  els.finalPrediction.textContent = item.prediction || "No final prediction";
  els.referenceText.textContent = item.reference_text || "Reference unavailable";
  renderMetrics(item);
  document.querySelector(".raw-log-link")?.remove();
  if (item.raw_log_url) {
    const link = document.createElement("a");
    link.className = "raw-log-link";
    link.href = item.raw_log_url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "Open raw verbose log ↗";
    els.sourceFull.insertAdjacentElement("afterend", link);
  }
  els.trajectory.replaceChildren(...item.steps.map(renderStep));
  attachConsensus(item, null);
  loadDetail(item).then((detail) => {
    if (state.filtered[state.caseIndex] !== item) return;
    attachConsensus(item, detail);
  });
  document.querySelectorAll(".case-link").forEach((button) => {
    button.classList.toggle("active", button.dataset.uttId === item.utt_id);
    if (button.dataset.uttId === item.utt_id) button.scrollIntoView({ block: "nearest" });
  });
  const hash = `#case=${encodeURIComponent(item.utt_id)}`;
  if (window.location.hash !== hash) window.history.replaceState(null, "", hash);

  if (item.audio_url) {
    els.audioPlayer.hidden = false;
    els.audioPlayer.src = item.audio_url;
    els.audioStatus.textContent = `${formatMetric(item.audio_duration_seconds, 1)} seconds / mono 16 kHz preview`;
  } else {
    els.audioPlayer.hidden = true;
    els.audioPlayer.removeAttribute("src");
    els.audioStatus.textContent = "Audio was not available while packaging this case";
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function caseLinkElement(item, filteredIndex) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "case-link";
  button.dataset.uttId = item.utt_id;
  button.title = item.source_full_text;

  const order = document.createElement("span");
  order.className = "case-order";
  order.textContent = String(item.display_order).padStart(3, "0");
  const copy = document.createElement("span");
  copy.className = "case-link-copy";
  const id = document.createElement("span");
  id.className = "case-link-id";
  id.textContent = item.utt_id;
  const source = document.createElement("span");
  source.className = "case-link-source";
  source.textContent = item.source_full_text;
  copy.append(id, source);
  button.append(order, copy);
  button.addEventListener("click", () => {
    state.caseIndex = filteredIndex;
    renderCase();
  });
  return button;
}

function rebuildCaseList() {
  els.caseList.replaceChildren();
  if (!state.filtered.length) {
    const empty = document.createElement("p");
    empty.className = "case-list-empty";
    empty.textContent = "No cases match this search.";
    els.caseList.append(empty);
    els.currentCaseLabel.textContent = "No matching case";
    els.caseCount.textContent = `0 / ${state.payload.cases.length}`;
    return;
  }
  state.filtered.forEach((item, index) => els.caseList.append(caseLinkElement(item, index)));
  els.caseCount.textContent = state.filtered.length === state.payload.cases.length
    ? String(state.payload.cases.length)
    : `${state.filtered.length} / ${state.payload.cases.length}`;
  state.caseIndex = Math.min(state.caseIndex, Math.max(0, state.filtered.length - 1));
  renderCase();
}

function filterCases(query) {
  const normalized = query.trim().toLowerCase();
  const allCases = state.payload.cases;
  state.filtered = normalized
    ? allCases.filter((item) => `${item.utt_id} ${item.source_full_text} ${item.prediction}`.toLowerCase().includes(normalized))
    : [...allCases];
  state.caseIndex = 0;
  rebuildCaseList();
}

function moveCase(delta) {
  if (!state.filtered.length) return;
  state.caseIndex = (state.caseIndex + delta + state.filtered.length) % state.filtered.length;
  renderCase();
}

function stopPlayback() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  els.playTrajectory.textContent = "Play trajectory";
  document.querySelectorAll(".trajectory-row.active").forEach((row) => row.classList.remove("active"));
}

function playTrajectory() {
  if (state.timer) {
    stopPlayback();
    els.audioPlayer.pause();
    return;
  }
  const item = state.filtered[state.caseIndex];
  const rows = [...document.querySelectorAll(".trajectory-row")];
  if (!item || !rows.length) return;
  const durationMs = item.audio_duration_seconds
    ? Math.max(450, (item.audio_duration_seconds * 1000) / rows.length)
    : 850;
  let index = -1;
  els.playTrajectory.textContent = "Stop";
  if (item.audio_url) {
    els.audioPlayer.currentTime = 0;
    els.audioPlayer.play().catch(() => {});
  }
  const advance = () => {
    if (index >= 0) rows[index].classList.remove("active");
    index += 1;
    if (index >= rows.length) {
      stopPlayback();
      return;
    }
    rows[index].classList.add("active");
    rows[index].scrollIntoView({ behavior: "smooth", block: "center" });
  };
  advance();
  state.timer = window.setInterval(advance, durationMs);
}

async function load() {
  const response = await fetch("data/review.json");
  if (!response.ok) throw new Error(`Failed to load review data: ${response.status}`);
  state.payload = await response.json();
  state.payload.cases.sort((left, right) =>
    (left.row_index ?? Number.MAX_SAFE_INTEGER) - (right.row_index ?? Number.MAX_SAFE_INTEGER)
      || left.utt_id.localeCompare(right.utt_id),
  );
  state.payload.cases.forEach((item, index) => { item.display_order = index + 1; });
  state.filtered = [...state.payload.cases];
  const { run_name: runName, case_count: caseCount, selection } = state.payload.meta;
  els.runSummary.textContent = `${runName} / ${caseCount} generated cases / ${selection}`;
  const requestedId = new URLSearchParams(window.location.hash.slice(1)).get("case");
  const requestedIndex = state.filtered.findIndex((item) => item.utt_id === requestedId);
  state.caseIndex = requestedIndex >= 0 ? requestedIndex : 0;
  rebuildCaseList();
}

els.previousCase.addEventListener("click", () => moveCase(-1));
els.nextCase.addEventListener("click", () => moveCase(1));
els.caseSearch.addEventListener("input", (event) => filterCases(event.target.value));
els.clearSearch.addEventListener("click", () => {
  els.caseSearch.value = "";
  filterCases("");
  els.caseSearch.focus();
});
els.playTrajectory.addEventListener("click", playTrajectory);
els.showCumulative.addEventListener("change", () => {
  document.body.classList.toggle("show-cumulative", els.showCumulative.checked);
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.key === "ArrowLeft") moveCase(-1);
  if (event.key === "ArrowRight") moveCase(1);
});

load().catch((error) => {
  els.runSummary.textContent = error.message;
  console.error(error);
});
