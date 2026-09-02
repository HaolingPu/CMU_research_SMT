const state = {
  payload: null,
  filtered: [],
  caseIndex: 0,
  timer: null,
};

const els = {
  runSummary: document.querySelector("#run-summary"),
  caseSelect: document.querySelector("#case-select"),
  caseSearch: document.querySelector("#case-search"),
  previousCase: document.querySelector("#previous-case"),
  nextCase: document.querySelector("#next-case"),
  playTrajectory: document.querySelector("#play-trajectory"),
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
  els.metricGrid.replaceChildren(
    metricCard("CHAR BLEU", formatMetric(item.metrics.bleu_char)),
    metricCard("LAAL", formatMetric(item.metrics.laal_text, 2)),
    metricCard("WRITE STEPS", String(item.write_steps)),
    metricCard("TOTAL STEPS", String(item.steps.length)),
  );
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
  els.caseSelect.value = item.utt_id;
  els.casePosition.textContent = `CASE ${state.caseIndex + 1} OF ${state.filtered.length} / ${item.task}`;
  els.caseId.textContent = item.utt_id;
  els.sourceFull.textContent = item.source_full_text;
  els.finalPrediction.textContent = item.prediction || "No final prediction";
  els.referenceText.textContent = item.reference_text || "Reference unavailable";
  renderMetrics(item);
  els.trajectory.replaceChildren(...item.steps.map(renderStep));

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

function rebuildCaseOptions() {
  els.caseSelect.replaceChildren();
  for (const item of state.filtered) {
    const option = document.createElement("option");
    option.value = item.utt_id;
    option.textContent = `${item.utt_id} - ${item.source_full_text.slice(0, 72)}`;
    els.caseSelect.append(option);
  }
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
  rebuildCaseOptions();
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
  state.filtered = [...state.payload.cases];
  const { run_name: runName, case_count: caseCount, selection } = state.payload.meta;
  els.runSummary.textContent = `${runName} / ${caseCount} generated cases / ${selection}`;
  rebuildCaseOptions();
}

els.caseSelect.addEventListener("change", () => {
  state.caseIndex = state.filtered.findIndex((item) => item.utt_id === els.caseSelect.value);
  renderCase();
});
els.previousCase.addEventListener("click", () => moveCase(-1));
els.nextCase.addEventListener("click", () => moveCase(1));
els.caseSearch.addEventListener("input", (event) => filterCases(event.target.value));
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
