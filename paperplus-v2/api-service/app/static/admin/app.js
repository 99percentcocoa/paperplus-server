/* PaperPlus admin dashboard. Plain JS, no build step. Talks to /api/admin/*.
 * All dynamic content is inserted via h() as text nodes (never innerHTML), so student names,
 * error reasons etc. can't inject markup. */
"use strict";

const app = document.getElementById("app");
const whoamiInput = document.getElementById("whoami");
whoamiInput.value = localStorage.getItem("paperplus_admin_name") || "";
whoamiInput.addEventListener("input", () => localStorage.setItem("paperplus_admin_name", whoamiInput.value.trim()));

let refreshTimer = null;

// ---------- helpers ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "value") el.value = v;
    else if (k === "selected" || k === "disabled" || k === "checked" || k === "hidden") el[k] = !!v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function api(path, options = {}) {
  const res = await fetch("/api/admin" + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString() : "");
const pill = (status) => h("span", { class: "pill " + status }, status.replace("_", " "));

function toast(message, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " error" : "");
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), isError ? 6000 : 3000);
}

function requireName() {
  const name = whoamiInput.value.trim();
  if (!name) {
    toast("Enter your name (top right) before saving — it's recorded with the correction.", true);
    whoamiInput.focus();
    return null;
  }
  return name;
}

function emptyRow(cols, text) {
  return h("tr", {}, h("td", { colspan: cols, class: "empty" }, text));
}

function setActiveNav(name) {
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
}

async function updateBadge() {
  try {
    const s = await api("/summary");
    const badge = document.getElementById("open-badge");
    badge.hidden = !s.open_reviews;
    badge.textContent = s.open_reviews;
  } catch (_) { /* badge is best-effort */ }
}

const STATUS_LABEL = { ok: "Healthy", warning: "Warning", critical: "Critical" };

function timeAgo(iso) {
  if (!iso) return "never";
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  return hours < 48 ? `${hours} h ago` : `${Math.round(hours / 24)} days ago`;
}

// Topbar dot: green/amber/red for the worst active alert; red too when the API itself is unreachable.
function applyStatus(m) {
  const dot = document.getElementById("status-dot");
  if (!m) {
    dot.className = "status-dot critical";
    dot.title = "Monitoring unavailable — can't reach the dashboard API";
    return;
  }
  dot.className = "status-dot " + m.status;
  dot.title = m.alerts.length ? m.alerts.map((a) => a.message).join("\n") : "All systems healthy";
}

async function updateStatus() {
  try { applyStatus(await api("/monitoring")); } catch (_) { applyStatus(null); }
}

function statusPanel(m) {
  if (!m) return h("div", { class: "notice error" }, "Monitoring data is unavailable right now.");
  const { services, storage, activity: a, backlog: b } = m;
  const has = (code) => m.alerts.some((x) => x.code === code);
  const stat = (label, value, level, hint) =>
    h("div", { class: "card kpi" + (level ? " " + level : ""), title: hint || null },
      h("div", { class: "label" }, label), h("div", { class: "value" }, value));
  const svc = (s) => (s.ok ? `OK · ${s.latency_ms} ms` : `DOWN · ${s.error}`);
  const rate = a && a.failure_rate_24h !== null ? `${Math.round(a.failure_rate_24h * 100)}%` : "—";
  return h("div", { class: "card status-panel" },
    h("div", { class: "status-head" },
      h("h2", {}, "System status"),
      h("span", { class: "pill " + m.status }, STATUS_LABEL[m.status]),
      h("span", { class: "muted" }, `checked ${new Date(m.generated_at).toLocaleTimeString()}`)),
    m.alerts.map((x) => h("div", { class: "notice" + (x.severity === "critical" ? " error" : "") }, x.message)),
    h("div", { class: "kpis compact" },
      stat("Database", svc(services.database), services.database.ok ? "" : "alert"),
      stat("Vision service", svc(services.vision_service), services.vision_service.ok ? "" : "alert"),
      a && stat("Scans (1h / 24h)", `${a.scans_1h} / ${a.scans_24h}`),
      a && stat("Failure rate (24h)", rate, has("high_failure_rate") ? "warn" : ""),
      a && stat("Last scan", timeAgo(a.last_scan_at), has("no_recent_scans") ? "warn" : ""),
      b && stat("Oldest failed scan", b.oldest_open_review_hours === null ? "—" : `${b.oldest_open_review_hours} h`,
        has("stale_review") ? "warn" : ""),
      storage.ok
        ? stat("Storage used", `${storage.used_pct}%`, has("disk_critical") ? "alert" : has("disk_warning") ? "warn" : "",
            `${storage.free_gb} GB free of ${storage.total_gb} GB`)
        : stat("Storage", "unknown", "warn")));
}

function pager(total, offset, limit, go) {
  return h("div", { class: "pager" },
    h("button", { disabled: offset <= 0, onclick: () => go(Math.max(0, offset - limit)) }, "← Prev"),
    h("span", { class: "muted" }, total ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total}` : "0 results"),
    h("button", { disabled: offset + limit >= total, onclick: () => go(offset + limit) }, "Next →"));
}

function submissionsTable(items) {
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Student", "School", "Worksheet", "Score", "Submitted"].map((t) => h("th", {}, t)))),
    h("tbody", {}, items.length ? items.map((s) =>
      h("tr", { class: "clickable", onclick: () => (location.hash = `#/submissions/${s.submission_id}`) },
        h("td", {}, `${s.student_name} `, h("span", { class: "muted" }, `(${s.student_id})`)),
        h("td", {}, s.school_code || "—"),
        h("td", {}, s.worksheet_id),
        h("td", {}, `${s.score ?? "—"}/${s.total_questions}`),
        h("td", {}, fmtDate(s.submitted_at)))) : [emptyRow(5, "No submissions yet.")])));
}

function reviewsTable(items) {
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Status", "Roll no.", "Worksheet", "Reason", "When"].map((t) => h("th", {}, t)))),
    h("tbody", {}, items.length ? items.map((r) =>
      h("tr", { class: "clickable", onclick: () => (location.hash = `#/reviews/${r.review_id}`) },
        h("td", {}, pill(r.status)),
        h("td", {}, r.detected_roll_number || r.student_id || "Unknown"),
        h("td", {}, r.worksheet_id ?? "—"),
        h("td", {}, r.error_reason || ""),
        h("td", {}, fmtDate(r.created_at)))) : [emptyRow(5, "No failed scans. 🎉")])));
}

// ---------- image viewer ----------
function imageViewer(scans) {
  // scans: [{page_no, images:{checked, upload}}]
  const views = [];
  for (const scan of scans) {
    const page = scan.page_no ? `Page ${scan.page_no} ` : "";
    for (const [key, label] of [["checked", "checked"], ["upload", "original"]]) {
      if (scan.images[key]) views.push({ label: page + label, url: scan.images[key] });
    }
  }
  if (!views.length) {
    return h("div", { class: "card viewer" }, h("p", { class: "muted" }, "No scan image stored for this entry."));
  }
  const img = h("img", { src: views[0].url, alt: "scan" });
  const tabs = h("div", { class: "tabs" });
  views.forEach((v, i) => {
    const btn = h("button", {
      class: i === 0 ? "active" : "",
      onclick: () => {
        img.src = v.url;
        tabs.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      },
    }, v.label);
    tabs.append(btn);
  });
  return h("div", { class: "card viewer" }, tabs, h("a", { href: views[0].url, target: "_blank" }, img));
}

// ---------- views ----------
async function overviewView() {
  setActiveNav("overview");
  // Monitoring is fetched independently: when the database is down the other calls fail, and that is
  // exactly when the status panel matters most.
  const monitoringPromise = api("/monitoring").catch(() => null);
  const scheduleRefresh = () => {
    refreshTimer = setInterval(() => { if (!location.hash || location.hash === "#/") overviewView().catch(() => {}); }, 30000);
  };
  let summary, subs, reviews;
  try {
    [summary, subs, reviews] = await Promise.all([
      api("/summary"), api("/submissions?limit=10"), api("/reviews?status=open&limit=10"),
    ]);
  } catch (e) {
    const m = await monitoringPromise;
    applyStatus(m);
    app.replaceChildren(
      h("h1", {}, "PaperPlus Dashboard"), statusPanel(m),
      h("div", { class: "notice error" }, `Couldn't load dashboard data: ${e.message}`));
    scheduleRefresh();
    return;
  }
  const monitoring = await monitoringPromise;
  applyStatus(monitoring);
  const kpi = (label, value, alert) =>
    h("div", { class: "card kpi" + (alert ? " alert" : "") }, h("div", { class: "label" }, label), h("div", { class: "value" }, value));
  app.replaceChildren(
    h("h1", {}, "PaperPlus Dashboard"),
    h("p", { class: "muted" }, "Auto-refreshes every 30 seconds."),
    statusPanel(monitoring),
    h("div", { class: "kpis" },
      kpi("Schools", summary.schools), kpi("Students", summary.students),
      kpi("Submissions", summary.submissions), kpi("Last 24h", summary.submissions_24h),
      kpi("Failed scans to review", summary.open_reviews, summary.open_reviews > 0)),
    h("div", { class: "grid2" },
      h("div", { class: "card" }, h("h2", {}, "Recent submissions"), submissionsTable(subs.items),
        h("p", {}, h("a", { href: "#/submissions" }, "All submissions →"))),
      h("div", { class: "card" }, h("h2", {}, "Failed scans"), reviewsTable(reviews.items),
        h("p", {}, h("a", { href: "#/reviews" }, "All failed scans →")))));
  const badge = document.getElementById("open-badge");
  badge.hidden = !summary.open_reviews;
  badge.textContent = summary.open_reviews;
  scheduleRefresh();
}

async function submissionsView(params) {
  setActiveNav("submissions");
  const limit = 25;
  const offset = parseInt(params.get("offset") || "0", 10);
  const studentId = params.get("student_id") || "";
  const worksheetId = params.get("worksheet_id") || "";
  const qs = new URLSearchParams({ limit, offset });
  if (studentId) qs.set("student_id", studentId);
  if (worksheetId) qs.set("worksheet_id", worksheetId);
  const data = await api("/submissions?" + qs);

  const navigate = (newOffset) => {
    const p = new URLSearchParams();
    if (newOffset) p.set("offset", newOffset);
    if (sid.value.trim()) p.set("student_id", sid.value.trim());
    if (wid.value.trim()) p.set("worksheet_id", wid.value.trim());
    location.hash = "#/submissions" + (p.toString() ? "?" + p : "");
  };
  const sid = h("input", { placeholder: "Student ID", value: studentId, size: 8, onkeydown: (e) => e.key === "Enter" && navigate(0) });
  const wid = h("input", { placeholder: "Worksheet ID", value: worksheetId, size: 10, onkeydown: (e) => e.key === "Enter" && navigate(0) });
  app.replaceChildren(
    h("h1", {}, "Submissions"),
    h("div", { class: "toolbar" }, sid, wid, h("button", { onclick: () => navigate(0) }, "Filter")),
    h("div", { class: "card" }, submissionsTable(data.items), pager(data.total, offset, limit, navigate)));
}

async function submissionView(id) {
  setActiveNav("submissions");
  const data = await api(`/submissions/${id}`);
  const original = new Map(data.questions.map((q) => [q.question_index, q.selected_option || ""]));
  const changes = new Map();
  const counter = h("span", { class: "muted" });
  const saveBtn = h("button", { class: "primary", disabled: true }, "Save corrections");

  const rows = data.questions.map((q) => {
    const resultCell = h("td", {});
    const select = h("select", {},
      h("option", { value: "" }, "— blank —"),
      q.labels.map((label) => h("option", { value: label, selected: label === (q.selected_option || "") }, label)));
    const tr = h("tr", { class: q.scanned ? "" : "unscanned" });
    const paint = () => {
      const current = select.value;
      const isChange = current !== original.get(q.question_index);
      tr.classList.toggle("changed", isChange);
      if (isChange) changes.set(q.question_index, current); else changes.delete(q.question_index);
      const correct = current && q.correct_option && current === q.correct_option;
      resultCell.replaceChildren(
        current === "" ? h("span", { class: "mark-blank" }, "–")
          : q.correct_option ? h("span", { class: correct ? "mark-ok" : "mark-bad" }, correct ? "✔" : "✘")
          : h("span", { class: "muted", title: "no answer key" }, "?"));
      counter.textContent = changes.size ? `${changes.size} unsaved change(s)` : "";
      saveBtn.disabled = changes.size === 0;
    };
    select.addEventListener("change", paint);
    tr.append(
      h("td", { class: "q" }, q.question_index),
      h("td", { class: "qtext" }, q.question_text || "", q.options ? h("div", { class: "muted" }, q.options.join("  ·  ")) : null),
      h("td", {}, select),
      h("td", {}, q.correct_option || "—"),
      resultCell);
    paint();
    return tr;
  });
  changes.clear();
  counter.textContent = "";
  saveBtn.disabled = true;

  saveBtn.addEventListener("click", async () => {
    const name = requireName();
    if (!name) return;
    saveBtn.disabled = true;
    try {
      const result = await api(`/submissions/${id}/answers`, {
        method: "PATCH",
        body: {
          corrected_by: name,
          corrections: [...changes].map(([question_index, selected_option]) => ({ question_index, selected_option })),
        },
      });
      toast(`Saved. New score: ${result.score}/${result.total_questions}`);
      await submissionView(id);
    } catch (e) {
      toast(e.message, true);
      saveBtn.disabled = false;
    }
  });

  app.replaceChildren(
    h("p", {}, h("a", { href: "#/submissions" }, "← Submissions")),
    h("h1", {}, `${data.student.student_name || "Unknown student"} `, h("span", { class: "muted" }, `(${data.student.student_id})`)),
    h("p", { class: "muted" },
      `Worksheet ${data.worksheet.worksheet_id}${data.worksheet.title ? " · " + data.worksheet.title : ""}` +
      ` · Score ${data.score ?? "—"}/${data.total_questions} · ${fmtDate(data.submitted_at)}` +
      (data.question_paper_code ? ` · Code ${data.question_paper_code}` : "") +
      (data.from_number ? ` · ${data.from_number}` : "")),
    h("div", { class: "split" },
      imageViewer(data.scans),
      h("div", { class: "card" },
        h("div", { class: "toolbar" }, h("h2", { style: "margin:0" }, "Answers"), h("span", { class: "spacer" }), counter, saveBtn),
        h("div", { class: "table-wrap" }, h("table", { class: "qgrid" },
          h("thead", {}, h("tr", {}, ["#", "Question", "Answer", "Correct", ""].map((t) => h("th", {}, t)))),
          h("tbody", {}, rows))),
        data.history.length ? h("div", {},
          h("h2", { style: "margin-top:20px" }, "Correction history"),
          h("table", {}, h("tbody", {}, data.history.map((r) => h("tr", {},
            h("td", {}, fmtDate(r.corrected_at)), h("td", {}, r.corrected_by || "—"),
            h("td", {}, `${r.original_score ?? "—"} → ${r.corrected_score ?? "—"}`)))))) : null)));
}

async function reviewsView(params) {
  setActiveNav("reviews");
  const limit = 50;
  const offset = parseInt(params.get("offset") || "0", 10);
  const status = params.get("status") || "open";
  const data = await api(`/reviews?status=${encodeURIComponent(status)}&limit=${limit}&offset=${offset}`);
  const go = (newOffset, newStatus = status) => {
    const p = new URLSearchParams({ status: newStatus });
    if (newOffset) p.set("offset", newOffset);
    location.hash = "#/reviews?" + p;
  };
  const filter = h("select", { onchange: (e) => go(0, e.target.value) },
    ["open", "all", "failed", "needs_review", "corrected", "approved"].map((s) =>
      h("option", { value: s, selected: s === status }, s === "open" ? "open (failed + needs review)" : s.replace("_", " "))));
  app.replaceChildren(
    h("h1", {}, "Failed scans"),
    h("p", { class: "muted" }, "Scans that couldn't be graded — unrecognized roll number or worksheet, missing answer key, or an unreadable image."),
    h("div", { class: "toolbar" }, filter),
    h("div", { class: "card" }, reviewsTable(data.items), pager(data.total, offset, limit, go)));
}

async function reviewView(id, overrides = {}, keep = {}) {
  setActiveNav("reviews");
  const qs = new URLSearchParams();
  if (overrides.worksheet_id) qs.set("worksheet_id", overrides.worksheet_id);
  if (overrides.question_paper_code) qs.set("question_paper_code", overrides.question_paper_code);
  const data = await api(`/reviews/${id}` + (qs.toString() ? "?" + qs : ""));

  const header = [
    h("p", {}, h("a", { href: "#/reviews" }, "← Failed scans")),
    h("h1", {}, `Failed scan #${data.review_id} `, pill(data.status)),
    h("p", { class: "muted" },
      `${fmtDate(data.created_at)}` + (data.scan?.from_number ? ` · from ${data.scan.from_number}` : "") +
      (data.scan?.page_no ? ` · page ${data.scan.page_no}` : "")),
    h("div", { class: "notice error" }, data.error_reason || "Unknown failure"),
  ];
  const viewer = data.scan ? imageViewer([data.scan]) : imageViewer([]);
  const dismiss = async () => {
    const name = requireName();
    if (!name || !confirm("Dismiss this scan without grading it?")) return;
    try {
      await api(`/reviews/${id}/status`, { method: "POST", body: { status: "approved", corrected_by: name } });
      toast("Dismissed.");
      location.hash = "#/reviews";
    } catch (e) { toast(e.message, true); }
  };

  if (!data.resolvable) {
    const alreadyDone = data.status === "corrected" || data.status === "approved";
    const triedWorksheetId = overrides.worksheet_id;
    const note = alreadyDone
      ? h("p", {}, "This entry is already ", data.status, ".", data.submission_id ? [" ", h("a", { href: `#/submissions/${data.submission_id}` }, "View submission →")] : null)
      : triedWorksheetId
      ? h("div", { class: "notice error" }, `Worksheet ${triedWorksheetId} doesn't exist. Check the ID and try again.`)
      : h("div", { class: "notice" }, "The tags on the page couldn't be read, or the roll number came back unreadable, so vision-service couldn't return any marks. Enter what you can read off the photo — worksheet ID, roll number, and the paper code if it has one. Retry re-sends the same photo, skipping the corner-tag check and the handwriting OCR you've filled in; or skip straight to grading it by hand while looking at the photo on the left.");
    let retrySection = null;
    if (!alreadyDone && data.scan) {
      const retryInput = h("input", { type: "number", value: triedWorksheetId ?? data.worksheet?.worksheet_id ?? "", placeholder: "worksheet id" });
      const rollInput = h("input", { value: data.detected_roll_number ?? "", placeholder: "4 digits", maxlength: 4, size: 6 });
      const codeInput = h("input", { value: overrides.question_paper_code ?? "", placeholder: "A–F (OMR only)", maxlength: 1, size: 6 });
      const entered = () => {
        const worksheet_id = parseInt(retryInput.value, 10);
        if (!worksheet_id) { toast("Enter a worksheet ID first.", true); return null; }
        return { worksheet_id, roll_number: rollInput.value.trim(), question_paper_code: codeInput.value.trim().toUpperCase() };
      };
      const retryBtn = h("button", { class: "primary" }, "Retry with these details");
      retryBtn.addEventListener("click", async () => {
        const body = entered();
        if (!body) return;
        retryBtn.disabled = true;
        try {
          await api(`/reviews/${id}/retry`, { method: "POST", body });
          toast("Retried — checking whether marks were recovered.");
          reviewView(id).catch((e) => toast(e.message, true));
        } catch (e) {
          toast(e.message, true);
          retryBtn.disabled = false;
        }
      });
      const manualBtn = h("button", {}, "Skip retry — grade manually");
      manualBtn.addEventListener("click", () => {
        const body = entered();
        if (!body) return;
        reviewView(id, { worksheet_id: body.worksheet_id, question_paper_code: body.question_paper_code }, { search: body.roll_number, roll: body.roll_number })
          .catch((e) => toast(e.message, true));
      });
      retrySection = h("div", { class: "fields" },
        h("label", {}, "Worksheet ID", retryInput),
        h("label", {}, "Roll number", rollInput),
        h("label", {}, "Question paper code", codeInput),
        h("label", {}, " ", retryBtn),
        h("label", {}, " ", manualBtn));
    } else if (!alreadyDone) {
      retrySection = h("p", { class: "muted" }, "No photo was even stored for this scan, so there's nothing to retry or grade — ask the student to resend it.");
    }
    app.replaceChildren(...header, h("div", { class: "split" }, viewer,
      h("div", { class: "card" }, note, retrySection,
        alreadyDone ? null : h("button", { class: "danger", onclick: dismiss }, "Dismiss"))));
    return;
  }

  // Resolvable: pick the student, confirm worksheet/code, adjust answers, save.
  const state = { student: keep.student || null, corrections: keep.corrections || new Map() };
  const studentBox = h("div", { class: "results" });
  const studentLabel = h("div", { class: "muted" });
  const paintStudent = () => {
    studentLabel.textContent = state.student ? `Selected: ${state.student.student_name} (${state.student.student_id})` : "No student selected";
  };
  const searchInput = h("input", { placeholder: "Search by roll number or name", value: keep.search ?? data.detected_roll_number ?? "" });
  let searchTimer;
  const search = async () => {
    const results = await api("/students?q=" + encodeURIComponent(searchInput.value.trim()));
    studentBox.replaceChildren(...results.map((s) => {
      const btn = h("button", {
        class: state.student?.student_id === s.student_id ? "picked" : "",
        onclick: () => { state.student = s; paintStudent(); studentBox.querySelectorAll("button").forEach((b) => b.classList.remove("picked")); btn.classList.add("picked"); },
      }, `${s.student_id} · ${s.student_name}${s.school_code ? " · " + s.school_code : ""}`);
      return btn;
    }));
    if (!results.length) studentBox.replaceChildren(h("span", { class: "muted" }, "No matching students. Add the student first (see docs), then search again."));
    const knownRoll = keep.roll || data.detected_roll_number;
    if (!state.student && knownRoll) {
      const exact = results.find((s) => s.student_id === knownRoll);
      if (exact) { state.student = exact; paintStudent(); studentBox.querySelectorAll("button").forEach((b) => { if (b.textContent.startsWith(exact.student_id)) b.classList.add("picked"); }); }
    }
  };
  searchInput.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => search().catch((e) => toast(e.message, true)), 250); });

  const worksheetInput = h("input", { type: "number", value: overrides.worksheet_id ?? data.worksheet?.worksheet_id ?? data.scan?.worksheet_id ?? "", placeholder: "worksheet id" });
  const codeInput = h("input", { value: overrides.question_paper_code ?? data.scan?.question_paper_code ?? "", placeholder: "e.g. D (OMR only)", maxlength: 1, size: 4 });
  const reload = () => reviewView(id, { worksheet_id: worksheetInput.value.trim(), question_paper_code: codeInput.value.trim().toUpperCase() },
    { student: state.student, corrections: state.corrections, search: searchInput.value }).catch((e) => toast(e.message, true));

  const warnings = [
    data.worksheet_missing ? h("div", { class: "notice error" }, "That worksheet doesn't exist in the database. Enter the right worksheet ID and reload the preview.") : null,
    data.answer_key_missing ? h("div", { class: "notice error" }, "No answer key found for this worksheet/code. Enter the question paper code (OMR) or seed an answer key, then reload.") : null,
    data.manual_entry ? h("div", { class: "notice" }, "No vision result for this scan — every question below starts blank. Look at the photo on the left and pick each answer by hand.") : null,
  ];

  const counter = h("span", { class: "muted" });
  const rows = data.questions.map((q) => {
    const initial = state.corrections.has(q.question_index) ? state.corrections.get(q.question_index) : q.detected_option;
    const select = h("select", {}, h("option", { value: "" }, "— blank —"),
      q.labels.map((label) => h("option", { value: label, selected: label === initial }, label)));
    const resultCell = h("td", {});
    const tr = h("tr", {});
    const paint = () => {
      const changed = select.value !== q.detected_option;
      tr.classList.toggle("changed", changed);
      if (changed) state.corrections.set(q.question_index, select.value); else state.corrections.delete(q.question_index);
      const ok = select.value && q.correct_option && select.value === q.correct_option;
      resultCell.replaceChildren(select.value === "" ? h("span", { class: "mark-blank" }, "–")
        : q.correct_option ? h("span", { class: ok ? "mark-ok" : "mark-bad" }, ok ? "✔" : "✘") : h("span", { class: "muted" }, "?"));
      counter.textContent = state.corrections.size ? `${state.corrections.size} edited` : "";
    };
    select.addEventListener("change", paint);
    tr.append(h("td", { class: "q" }, q.question_index),
      h("td", { class: "qtext" }, q.question_text || ""),
      h("td", {}, q.detected_option || "—", q.confidence != null && q.detected_option ? h("span", { class: "muted" }, ` (${Math.round(q.confidence * 100)}%)`) : null),
      h("td", {}, select), h("td", {}, q.correct_option || "—"), resultCell);
    paint();
    return tr;
  });

  const saveBtn = h("button", { class: "primary" }, "Save & create submission");
  saveBtn.addEventListener("click", async () => {
    const name = requireName();
    if (!name) return;
    if (!state.student) { toast("Pick the student this scan belongs to first.", true); return; }
    saveBtn.disabled = true;
    try {
      const body = {
        student_id: state.student.student_id,
        corrected_by: name,
        corrections: [...state.corrections].map(([question_index, selected_option]) => ({ question_index, selected_option })),
      };
      if (worksheetInput.value.trim()) body.worksheet_id = parseInt(worksheetInput.value, 10);
      if (codeInput.value.trim()) body.question_paper_code = codeInput.value.trim().toUpperCase();
      const result = await api(`/reviews/${id}/resolve`, { method: "POST", body });
      toast(`Saved. Score: ${result.score}/${result.total_questions}`);
      location.hash = `#/submissions/${result.submission_id}`;
    } catch (e) {
      toast(e.message, true);
      saveBtn.disabled = false;
    }
  });

  app.replaceChildren(...header, ...warnings, h("div", { class: "split" }, viewer,
    h("div", { class: "card" },
      h("h2", {}, "1. Which student is this?"),
      h("p", { class: "muted" }, `Detected roll number: ${data.detected_roll_number || "none"}`),
      searchInput, studentBox, studentLabel,
      h("h2", { style: "margin-top:20px" }, "2. Worksheet"),
      h("div", { class: "fields" },
        h("label", {}, "Worksheet ID", worksheetInput),
        h("label", {}, "Question paper code", codeInput),
        h("label", {}, " ", h("button", { onclick: reload }, "Reload answer key"))),
      h("div", { class: "toolbar" }, h("h2", { style: "margin:0" }, "3. Check the answers"), h("span", { class: "spacer" }), counter),
      h("div", { class: "table-wrap" }, h("table", { class: "qgrid" },
        h("thead", {}, h("tr", {}, ["#", "Question", "Detected", "Answer", "Correct", ""].map((t) => h("th", {}, t)))),
        h("tbody", {}, rows))),
      h("div", { class: "toolbar" }, saveBtn, h("button", { class: "danger", onclick: dismiss }, "Dismiss")))));
  paintStudent();
  search().catch((e) => toast(e.message, true));
}

async function schoolsView() {
  setActiveNav("schools");
  const data = await api("/schools");
  app.replaceChildren(
    h("h1", {}, "Schools"),
    h("div", { class: "card" }, h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["School", "Code", "Students"].map((t) => h("th", {}, t)))),
      h("tbody", {}, data.length ? data.map((s) =>
        h("tr", { class: "clickable", onclick: () => (location.hash = `#/schools/${s.school_code}`) },
          h("td", {}, s.school_name || "—"),
          h("td", {}, s.school_code),
          h("td", {}, s.student_count))) : [emptyRow(3, "No schools yet.")])))));
}

async function schoolView(code) {
  setActiveNav("schools");
  const data = await api(`/schools/${encodeURIComponent(code)}`);
  const worksheetChips = (student) => student.recent_worksheets.length
    ? student.recent_worksheets.map((w) =>
        h("a", { class: "pill", href: `#/submissions/${w.submission_id}` }, `${w.level}: ${w.score ?? "—"}/${w.total_questions}`))
    : h("span", { class: "muted" }, "—");
  app.replaceChildren(
    h("p", {}, h("a", { href: "#/schools" }, "← Schools")),
    h("h1", {}, data.school.school_name || data.school.school_code, " ", h("span", { class: "muted" }, `(${data.school.school_code})`)),
    h("div", { class: "card" }, h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Student", "Roll no.", "Current level", "Last submitted", "Recent worksheets"].map((t) => h("th", {}, t)))),
      h("tbody", {}, data.students.length ? data.students.map((s) =>
        h("tr", {},
          h("td", {}, s.student_name),
          h("td", {}, s.student_id),
          h("td", {}, s.current_level || "—"),
          h("td", {}, s.last_submitted_at ? fmtDate(s.last_submitted_at) : h("span", { class: "muted" }, "Never")),
          h("td", { class: "chips" }, worksheetChips(s)))) : [emptyRow(5, "No students at this school.")])))));
}

// ---------- router ----------
async function route() {
  clearInterval(refreshTimer);
  const [path, query = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query);
  try {
    if (!parts.length) await overviewView();
    else if (parts[0] === "submissions" && parts[1]) await submissionView(parts[1]);
    else if (parts[0] === "submissions") await submissionsView(params);
    else if (parts[0] === "reviews" && parts[1]) await reviewView(parts[1]);
    else if (parts[0] === "reviews") await reviewsView(params);
    else if (parts[0] === "schools" && parts[1]) await schoolView(parts[1]);
    else if (parts[0] === "schools") await schoolsView();
    else app.replaceChildren(h("p", {}, "Page not found. ", h("a", { href: "#/" }, "Go to overview")));
    if (parts.length) { updateBadge(); updateStatus(); }
  } catch (e) {
    app.replaceChildren(h("div", { class: "notice error" }, `Something went wrong: ${e.message}`), h("p", {}, h("a", { href: "#/" }, "← Back to overview")));
  }
  window.scrollTo(0, 0);
}

setInterval(updateStatus, 60000);
window.addEventListener("hashchange", route);
route();
