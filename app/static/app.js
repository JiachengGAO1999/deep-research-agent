const $ = (id) => document.getElementById(id);
const stageNames = {
  initialize: "初始化研究任务", plan_queries: "分析问题与规划检索",
  search_sources: "检索学术数据源", normalize_and_deduplicate: "规范化与文献去重",
  rank_and_select: "筛选相关论文", download_pdfs: "下载开放全文",
  parse_and_chunk: "解析论文全文", extract_evidence: "检索与验证证据",
  assess_gaps: "分析证据缺口", supplementary_search: "补充检索",
  build_claims: "构建 Claim–Evidence", synthesize_report: "撰写研究报告",
  build_literature_relations: "构建文献关系", validate_evidence: "程序化证据校验", validate_citations: "校验引用完整性", finalize: "完成"
};
let pollTimer = null;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

function numberOrNull(id) {
  const value = $(id).value.trim();
  return value ? Number(value) : null;
}

function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  })[c]);
}

function showMessages(warnings = [], errors = []) {
  $("messages").innerHTML = [
    ...warnings.map(x => `<div class="notice">${escapeHtml(x)}</div>`),
    ...errors.map(x => `<div class="notice error">${escapeHtml(x)}</div>`)
  ].join("");
}

async function poll(taskId) {
  try {
    const status = await api(`/api/research/${taskId}`);
    const progress = status.progress_percent || 0;
    $("stage-label").textContent = stageNames[status.current_stage] || status.status;
    $("progress-number").textContent = `${progress}%`;
    $("bar-fill").style.width = `${progress}%`;
    $("rounds").textContent = status.current_round + 1;
    $("papers-found").textContent = status.papers_found;
    $("papers-selected").textContent = status.papers_selected;
    showMessages(status.warnings, status.errors);
    if (status.status === "completed") {
      clearInterval(pollTimer);
      await loadResults(taskId);
      $("submit").disabled = false;
      $("submit").textContent = "再次研究";
    } else if (status.status === "failed") {
      clearInterval(pollTimer);
      $("submit").disabled = false;
      $("submit").textContent = "重新尝试";
    }
  } catch (error) {
    showMessages([], [error.message]);
  }
}

async function loadResults(taskId) {
  const [report, papers, evidence, claims] = await Promise.all([
    api(`/api/research/${taskId}/report`),
    api(`/api/research/${taskId}/papers?selected_only=true`),
    api(`/api/research/${taskId}/evidence`),
    api(`/api/research/${taskId}/claims`)
  ]);
  $("report").textContent = report.report;
  $("papers").innerHTML = papers.papers.map(p => `
    <article class="card">
      <h3>${escapeHtml(p.title)}</h3>
      <div class="meta">${escapeHtml(p.publication_year || "n.d.")} · ${escapeHtml(p.venue || "Unknown venue")} · relevance ${escapeHtml(p.relevance_score ?? "—")}</div>
      <p>${escapeHtml((p.abstract || "").slice(0, 600))}</p>
      ${p.doi ? `<a href="https://doi.org/${encodeURIComponent(p.doi)}" target="_blank" rel="noreferrer">DOI ${escapeHtml(p.doi)}</a>` : ""}
    </article>`).join("");
  $("evidence").innerHTML = evidence.evidence.map(e => `
    <article class="card">
      <h3>${escapeHtml(e.paper_id)} · ${escapeHtml(e.section_title || "Abstract")}</h3>
      <div class="meta">${escapeHtml(e.evidence_type)} · ${escapeHtml(e.verification_status)} ${e.page_start ? `· p.${e.page_start}` : ""}</div>
      <p class="quote">${escapeHtml(e.evidence_quote || "")}</p>
    </article>`).join("");
  $("claims").innerHTML = `
    <div class="meta">Evidence quality: ${escapeHtml(JSON.stringify(claims.evidence_quality || {}))}</div>
    ${claims.claims.map(c => `<article class="card">
      <h3>${escapeHtml(c.claim_text)}</h3>
      <span class="badge">${escapeHtml(c.support_status)}</span>
      <span class="badge">confidence ${escapeHtml(c.confidence)}</span>
      <div class="meta">Papers: ${escapeHtml((c.paper_ids || []).join(", "))}</div>
    </article>`).join("")}`;
  $("results").classList.remove("hidden");
}

$("research-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearInterval(pollTimer);
  $("submit").disabled = true;
  $("submit").textContent = "正在创建任务…";
  $("results").classList.add("hidden");
  $("progress-panel").classList.remove("hidden");
  try {
    const payload = {
      topic: $("topic").value.trim(),
      year_from: numberOrNull("year-from"),
      year_to: numberOrNull("year-to"),
      max_papers: Number($("max-papers").value),
      research_depth: $("depth").value,
      evidence_backend: $("backend").value,
      enable_full_text: $("full-text").checked,
      report_language: $("language").value
    };
    const task = await api("/api/research", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    $("task-id").textContent = `Task ${task.task_id}`;
    $("submit").textContent = "研究进行中…";
    await poll(task.task_id);
    pollTimer = setInterval(() => poll(task.task_id), 1800);
  } catch (error) {
    showMessages([], [error.message]);
    $("submit").disabled = false;
    $("submit").textContent = "重新尝试";
  }
});

document.querySelectorAll(".tabs button").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(x => x.classList.remove("active"));
    button.classList.add("active");
    $(button.dataset.tab).classList.add("active");
  });
});

$("backend").addEventListener("change", () => {
  if ($("backend").value !== "abstract") $("full-text").checked = true;
});

api("/health").then(() => {
  $("health").textContent = "服务正常";
  $("health").classList.add("ok");
}).catch(() => $("health").textContent = "服务不可用");
