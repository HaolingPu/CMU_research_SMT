"use strict";

const $ = id => document.getElementById(id);
const state = { registry: null, bundles: new Map(), id: null, left: null, right: null, revision: 0 };
const finite = value => typeof value === "number" && Number.isFinite(value);
const metric = value => finite(value) ? value.toFixed(2) : "未提供";
const change = (a, b) => finite(a) && finite(b) ? b - a : null;
const signed = value => finite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(2)}` : "等待成对结果";
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function field(parent, name, value, exact = false) {
  parent.append(el("p", name, "field-label"), el(exact ? "pre" : "p", value ?? "未记录", exact ? "exact" : ""));
}
function experiment(id) { return state.registry.experiments.find(item => item.id === id); }
function itemFor(exp, id = state.id) { return state.bundles.get(exp.id)?.get(id); }
function caseUrl(exp, id) { return `${exp.bundle_url}#case=${encodeURIComponent(id)}`; }
function options(select, entries, selected) {
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = el("option", label);
    option.value = value;
    return option;
  }));
  select.value = selected;
}
function saveHash() {
  const params = new URLSearchParams({ case: state.id, left: state.left, right: state.right });
  history.replaceState(null, "", `#${params}`);
}
function validateBundle(exp, payload) {
  if (!Array.isArray(payload.cases)) throw new Error(`${exp.label}: 无 case 列表`);
  const map = new Map(payload.cases.map(item => [item.utt_id, item]));
  const expected = state.registry.case_ids;
  const known = new Set(expected);
  if (map.size !== payload.cases.length || [...map.keys()].some(id => !known.has(id))) {
    throw new Error(`${exp.label}: ID 重复或不在冻结输入中，拒绝错配对比`);
  }
  // A partial arm (e.g. an 8-case check) may cover a subset; a complete arm must cover all 50.
  if (exp.status === "complete" && map.size !== expected.length) {
    throw new Error(`${exp.label}: ID 缺失，拒绝错配对比`);
  }
  for (const item of map.values()) {
    if (!Array.isArray(item.steps) || !item.steps.length) throw new Error(`${exp.label}: trajectory 缺失`);
  }
  return map;
}
async function loadBundle(exp) {
  if (exp.status !== "complete" && exp.status !== "partial") { state.bundles.delete(exp.id); return; }
  const response = await fetch(`${exp.bundle_url}data/review.json`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${exp.label}: HTTP ${response.status}，不是有效的等待状态`);
  state.bundles.set(exp.id, validateBundle(exp, await response.json()));
}
function checkPair(left, right) {
  if (!left || !right) return;
  if (left.utt_id !== right.utt_id || left.source_full_text !== right.source_full_text ||
      left.reference_text !== right.reference_text ||
      JSON.stringify(left.src_text_full) !== JSON.stringify(right.src_text_full) ||
      JSON.stringify(left.steps.map(s => [s.step, s.source_chunk])) !== JSON.stringify(right.steps.map(s => [s.step, s.source_chunk]))) {
    throw new Error(`Case ${left.utt_id}: 输入、参考译文或 chunk 不一致，禁止作为同输入实验比较`);
  }
}
function covered(exp) { return state.bundles.get(exp.id)?.size ?? 0; }
function waiting(exp) {
  if (exp.status === "failed") return "任务异常，等待修复；没有可比较结果";
  if (exp.status === "partial") return `该 case 未在此实验中生成（已生成 ${covered(exp)} / ${state.registry.case_ids.length} cases）`;
  return "等待结果：尚未导入校验通过的完整 50-case 包";
}
function metadata(exp) {
  const panel = el("article", undefined, "panel");
  panel.append(el("h3", exp.label));
  const loaded = exp.status === "complete" || exp.status === "partial";
  panel.append(el("p", loaded ? `已保存 ${covered(exp)} / ${state.registry.case_ids.length} cases` : waiting(exp), exp.status === "complete" ? "" : "pending"));
  panel.append(el("p", exp.description));
  panel.append(el("small", `${exp.run_name} | ${exp.prompt_version}`));
  if (exp.jobs?.decode) panel.append(el("p", `生成 ${exp.jobs.decode} / 校验打包 ${exp.jobs.report}`, "guide"));
  return panel;
}
function outcome(exp, item) {
  const panel = el("article", undefined, "panel");
  panel.append(el("h3", exp.label));
  if (!item) { panel.append(el("p", waiting(exp), "pending")); return panel; }
  panel.append(el("p", `char-BLEU ${metric(item.metrics?.bleu_char)} | text-LAAL ${metric(item.metrics?.laal_text)} 词`));
  field(panel, "最终译文", item.prediction || "空译文");
  const link = el("a", "打开完整 trajectory / 原始日志");
  link.href = caseUrl(exp, item.utt_id);
  panel.append(link);
  return panel;
}
function samplingLabel(sampling) {
  if (!sampling || sampling.status === "unknown") return "未记录实际调用；不从全文推测";
  if (sampling.status === "invoked") return "已进入采样流程 / invoked";
  const reasons = { source_sentence_end: "英文句子结束，不采样", final_chunk: "最后一个 chunk，不采样", no_new_source: "没有新英文，不采样" };
  return reasons[sampling.reason] || `未调用采样 / ${sampling.reason || "原因未记录"}`;
}
function chunkCell(exp, step) {
  const cell = el("td");
  if (!step) { cell.append(el("p", waiting(exp), "pending")); return cell; }
  cell.append(el("span", step.action, `action ${step.action === "WRITE" ? "write" : ""}`));
  field(cell, "本步中文新增量", "");
  cell.lastElementChild.remove();
  cell.append(el("p", step.translation_delta || "无新增译文 (READ)", `delta${step.translation_delta ? "" : " empty"}`));
  field(cell, "本步之后累计中文", step.translation_cumulative || "(empty)");
  const sampling = step.sampling;
  field(cell, "采样器输入 source prefix", samplingLabel(sampling));
  if (sampling?.status === "invoked") cell.append(el("pre", sampling.input_source_prefix ?? "未记录精确输入", "exact sampler-prefix"));
  const details = el("details");
  details.append(el("summary", "实际输入 + 全部 futures + 句末修正"));
  field(details, "Sampler prefix 精确字符串", sampling?.status === "invoked" && typeof sampling.input_source_prefix === "string" ? JSON.stringify(sampling.input_source_prefix) : "没有已记录的调用输入", true);
  field(details, "Sampler 中文上下文", sampling?.context_mode === "source-only" ? "不传入中文 (source-only)" : sampling?.committed_target_context);
  field(details, "采样证据 / window", `${sampling?.evidence ?? "未记录"} / ${sampling?.window_mode ?? "未记录"}`);
  const cs = step.consensus_summary || {};
  const mode = cs.no_new_source ? "不调用" : cs.too_few_futures ? "有效 futures 不足，不探测" : cs.final_completion || cs.sentence_completion ? "不拼接 futures，直接补完" : "逐 future 共识探测；不是 HTTP 抓包";
  field(details, "Translator 调用方式", mode);
  field(details, `Translator 观察英文 (${step.source_input_provenance || "未记录"})`, step.source_cumulative, true);
  field(details, "Translator 本步开始前的中文", step.committed_before || "(empty)", true);
  field(details, "Translator 拼接模式", step.future_join_mode || "未记录");
  details.append(el("p", "共识探测将观察英文与各条 future 按上述模式拼接，并以本步之前中文及本步 pending tokens 约束翻译；以下不是完整 HTTP 请求重放。", "guide"));
  for (const group of step.selected_futures || []) {
    details.append(el("p", `${group.label} / ${group.mode} / ${group.candidates.length} retained`, "field-label"));
    const list = el("ol", undefined, "future-list");
    list.append(...group.candidates.map(candidate => el("li", candidate)));
    details.append(list);
  }
  if (!(step.selected_futures || []).length) details.append(el("p", "本步没有保留的 futures"));
  for (const stats of step.raw_stats || []) details.append(el("p", `${stats.label} ${stats.mode}: kept ${stats.kept}/${stats.count}; dropped ${stats.dropped}`, "guide"));
  if (step.boundary_close) {
    field(details, "句末修正前 delta", step.boundary_close.raw_delta, true);
    field(details, "句末修正后 delta", step.boundary_close.delta, true);
    field(details, "句末修正原因", step.boundary_close.reason);
  }
  cell.append(details);
  return cell;
}
function renderScores(leftExp, rightExp) {
  const rows = [], bleuPairs = [], laalPairs = [];
  for (const id of state.registry.case_ids) {
    const a = itemFor(leftExp, id), b = itemFor(rightExp, id);
    checkPair(a, b);
    const av = a?.metrics || {}, bv = b?.metrics || {};
    const db = change(av.bleu_char, bv.bleu_char), dl = change(av.laal_text, bv.laal_text);
    if (finite(db)) bleuPairs.push(db);
    if (finite(dl)) laalPairs.push(dl);
    const row = el("tr");
    const cell = el("td"), link = el("a", id);
    link.href = `#${new URLSearchParams({ case: id, left: state.left, right: state.right })}`;
    cell.append(link); row.append(cell);
    for (const value of [a ? metric(av.bleu_char) : "等待结果", b ? metric(bv.bleu_char) : "等待结果"]) row.append(el("td", value));
    row.append(el("td", signed(db), finite(db) && db !== 0 ? (db > 0 ? "positive" : "negative") : ""));
    for (const value of [a ? metric(av.laal_text) : "等待结果", b ? metric(bv.laal_text) : "等待结果"]) row.append(el("td", value));
    row.append(el("td", signed(dl), finite(dl) && dl !== 0 ? (dl < 0 ? "positive" : "negative") : ""));
    rows.push(row);
  }
  $("score-rows").replaceChildren(...rows);
  const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  $("aggregate").textContent = `可比较 char-BLEU ${bleuPairs.length}/50，平均变化 ${signed(mean(bleuPairs))}；text-LAAL ${laalPairs.length}/50，平均变化 ${signed(mean(laalPairs))}。BLEU 越高越好，LAAL 越低越好。`;
}
function render() {
  const leftExp = experiment(state.left), rightExp = experiment(state.right);
  const left = itemFor(leftExp), right = itemFor(rightExp), source = left || right;
  checkPair(left, right);
  $("experiment-info").replaceChildren(metadata(leftExp), metadata(rightExp));
  $("case-select").value = state.id;
  $("previous").disabled = state.registry.case_ids.indexOf(state.id) <= 0;
  $("next").disabled = state.registry.case_ids.indexOf(state.id) === state.registry.case_ids.length - 1;
  $("case-content").hidden = !source;
  if (source) {
    $("case-id").textContent = state.id;
    $("source-full").textContent = source.source_full_text;
    $("reference").textContent = source.reference_text;
    $("source-units").replaceChildren(...(source.src_text_full || []).map(unit => el("li", unit)));
    $("outcomes").replaceChildren(outcome(leftExp, left), outcome(rightExp, right));
    $("case-difference").textContent = left && right ? `右减左：char-BLEU ${signed(change(left.metrics?.bleu_char, right.metrics?.bleu_char))}；text-LAAL ${signed(change(left.metrics?.laal_text, right.metrics?.laal_text))} 词。` : "新版未完成：旧译文和旧指标照常显示，不把缺失结果当成 0。";
    $("left-heading").textContent = leftExp.label;
    $("right-heading").textContent = rightExp.label;
    $("chunks").replaceChildren(...source.steps.map((step, index) => {
      const a = left?.steps[index], b = right?.steps[index];
      const row = el("tr", undefined, a && b && (a.translation_delta !== b.translation_delta || a.action !== b.action) ? "changed" : "");
      row.append(el("td", step.step), el("td", step.source_chunk), chunkCell(leftExp, a), chunkCell(rightExp, b));
      return row;
    }));
    $("trajectory-link").href = caseUrl(left ? leftExp : rightExp, state.id);
  }
  renderScores(leftExp, rightExp);
  saveHash();
}
function fail(error) {
  $("load-error").hidden = false;
  $("load-error").textContent = `读取或校验失败：${error.message}。不能据此判断实验分数。`;
  $("case-content").hidden = true;
  $("score-rows").replaceChildren();
  $("aggregate").textContent = "结果校验失败，已停止比较。";
}
async function selectExperiments() {
  const revision = ++state.revision;
  $("load-error").hidden = true;
  try {
    await loadBundle(experiment(state.left));
    await loadBundle(experiment(state.right));
    if (revision !== state.revision) return;
    render();
  } catch (error) { if (revision === state.revision) fail(error); }
}
async function refresh() {
  $("refresh").disabled = true;
  try {
    const response = await fetch("experiments.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`实验目录 HTTP ${response.status}`);
    state.registry = await response.json();
    const params = new URLSearchParams(location.hash.slice(1));
    state.left = params.get("left") || state.registry.default_left;
    state.right = params.get("right") || state.registry.default_right;
    if (!experiment(state.left) || !experiment(state.right)) throw new Error("未知实验 ID");
    state.id = params.get("case") || state.registry.case_ids[0];
    if (!state.registry.case_ids.includes(state.id)) throw new Error("该 case 不在冻结的 50-case 输入中");
    options($("left-experiment"), state.registry.experiments.map(e => [e.id, e.label]), state.left);
    options($("right-experiment"), state.registry.experiments.map(e => [e.id, e.label]), state.right);
    options($("case-select"), state.registry.case_ids.map((id, i) => [id, `${i + 1} / 50 · ${id}`]), state.id);
    $("load-status").textContent = `本页每 60 秒检查已发布结果；不实时查询 Slurm。实验目录更新于 ${state.registry.updated_at}。`;
    await selectExperiments();
  } catch (error) { fail(error); }
  finally { $("refresh").disabled = false; }
}
for (const side of ["left", "right"]) $(`${side}-experiment`).addEventListener("change", () => {
  state[side] = $(`${side}-experiment`).value;
  selectExperiments();
});
$("case-select").addEventListener("change", () => { state.id = $("case-select").value; try { render(); } catch (error) { fail(error); } });
for (const [name, delta] of [["previous", -1], ["next", 1]]) $(name).addEventListener("click", () => {
  state.id = state.registry.case_ids[state.registry.case_ids.indexOf(state.id) + delta];
  try { render(); } catch (error) { fail(error); }
});
$("refresh").addEventListener("click", refresh);
window.addEventListener("hashchange", refresh);
// Do not rebuild open chunk/future details unless a new registry is published.
setInterval(async () => {
  if (!state.registry || document.hidden) return;
  try {
    const response = await fetch("experiments.json", { cache: "no-store" });
    if (response.ok && (await response.json()).updated_at !== state.registry.updated_at) await refresh();
  } catch (_) { $("load-status").textContent = "自动刷新暂时无法连接；当前显示已加载的结果。可手动刷新。"; }
}, 60000);
refresh();
