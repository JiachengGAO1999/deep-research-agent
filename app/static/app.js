const $ = (id) => document.getElementById(id);
const stageNames = {
  // Strict mode
  initialize: "初始化研究任务", plan_queries: "分析问题与规划检索",
  search_sources: "检索学术数据源", normalize_and_deduplicate: "规范化与文献去重",
  rank_and_select: "筛选相关论文", download_pdfs: "下载开放全文",
  parse_and_chunk: "解析论文全文", extract_evidence: "检索与验证证据",
  assess_gaps: "分析证据缺口", supplementary_search: "补充检索",
  build_claims: "构建 Claim–Evidence", synthesize_report: "撰写研究报告",
  build_literature_relations: "构建文献关系", validate_evidence: "程序化证据校验", validate_citations: "校验引用完整性", finalize: "完成",
  // Quick mode
  classify_question: "分析研究问题", quick_plan_queries: "规划检索查询",
  tavily_search: "搜索公开网页", quick_select_sources: "筛选高质量来源",
  tavily_extract: "提取网页正文", build_research_notes: "构建研究笔记",
  quick_assess_coverage: "评估覆盖度", quick_supplementary_search: "补充检索",
  build_comparison_matrix: "构建比较矩阵", synthesize_quick_report: "撰写研究报告",
  lightweight_citation_check: "引用完整性检查", quick_finalize: "完成"
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

let currentMode = "quick";

async function poll(taskId) {
  try {
    const status = await api(`/api/research/${taskId}`);
    const progress = status.progress_percent || 0;
    const mode = status.research_mode || "quick";
    currentMode = mode;
    $("stage-label").textContent = stageNames[status.current_stage] || status.status;
    $("progress-number").textContent = `${progress}%`;
    $("bar-fill").style.width = `${progress}%`;
    $("rounds").textContent = status.current_round + 1;
    $("llm-calls").textContent = status.llm_calls;

    // Toggle mode-specific stats visibility
    const isQuick = mode === "quick";
    document.querySelectorAll(".quick-stat").forEach(el => el.style.display = isQuick ? "" : "none");
    document.querySelectorAll(".strict-stat").forEach(el => el.style.display = isQuick ? "none" : "");
    $("runtime-meta").textContent = isQuick
      ? `research=quick · source=tavily · cost $${Number(status.estimated_cost_usd || 0).toFixed(4)}`
      : `research=strict · retrieval=${status.retrieval_backend || "—"} · cost $${Number(status.estimated_cost_usd || 0).toFixed(4)}`;

    // Quick mode stats
    $("web-results").textContent = status.web_results || 0;
    $("sources-selected").textContent = status.sources_selected || 0;
    $("research-notes").textContent = status.research_notes_count || 0;
    // Strict mode stats
    $("papers-found").textContent = status.papers_found;
    $("papers-selected").textContent = status.papers_selected;
    $("passages").textContent = status.retrieved_passages;
    $("verified-evidence").textContent = status.verified_evidence;

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
  $("report").innerHTML = renderMarkdown(report.report);
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
      research_question: $("research-question").value.trim(),
      topic: $("topic").value.trim() || null,
      year_from: numberOrNull("year-from"),
      year_to: numberOrNull("year-to"),
      max_papers: Number($("max-papers").value),
      research_depth: $("depth").value,
      research_mode: $("research-mode").value,
      retrieval_profile: $("profile").value,
      full_text_required: $("full-text-required").checked,
      language: $("language").value,
      max_cost_usd: numberOrNull("max-cost")
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

function renderMarkdown(md) {
  if (!md) return "";
  return md
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>")
    .replace(/\[S(\d+)\]/g, "<sup class=\"cite\">[$1]</sup>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/<p>/, "<p>");
}

api("/health").then(() => {
  $("health").textContent = "服务正常";
  $("health").classList.add("ok");
}).catch(() => $("health").textContent = "服务不可用");
