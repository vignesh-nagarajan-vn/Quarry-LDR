"use strict";

const view = document.getElementById("view");
const TOTAL_STAGES = 16;
const STAGE_WORDS = {
  plan: "Planning the questions",
  search: "Searching the web",
  fetch: "Fetching pages",
  extract: "Extracting text",
  chunk: "Splitting into passages",
  embed: "Indexing passages",
  dedup: "Removing duplicates",
  index: "Indexing passages",
  retrieve: "Finding relevant passages",
  rerank: "Ranking passages",
  triage: "Reading the evidence",
  gap: "Checking what is missing",
  conflicts: "Comparing sources",
  synthesize: "Writing the report",
  verify: "Checking every citation",
  render: "Producing the report",
};
const JOB_WORDS = {
  queued: "Queued",
  waiting_search: "Waiting for search",
  running: "Running",
  completed: "Finished",
  failed: "Failed",
  stopped: "Stopped",
};
const DEPTHS = [
  { iterations: 1, name: "Quick", note: "One search pass. About 25 minutes on Local." },
  { iterations: 3, name: "Deep", note: "Up to three search passes. About 90 minutes on Local." },
];
let timer = null;
let currentRoute = 0;
let queueShown = null;

async function api(path, options) {
  const init = options || {};
  if (init.body) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
  if (response.status === 204) {
    return null;
  }
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "That did not work.");
  }
  return data;
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") {
      node.className = value;
    } else if (key === "on") {
      for (const [name, handler] of Object.entries(value)) {
        node.addEventListener(name, handler);
      }
    } else if (value === true) {
      node.setAttribute(key, "");
    } else if (value !== false && value !== null && value !== undefined) {
      node.setAttribute(key, value);
    }
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function setView(route, ...children) {
  // A slow page must not draw over the one the reader has since moved to.
  if (route === currentRoute) {
    view.replaceChildren(...children);
  }
}

function money(value) {
  return `$${Number(value).toFixed(2)}`;
}

function when(iso) {
  const date = new Date(iso);
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function elapsed(startIso) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(startIso).getTime()) / 60000));
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function showError(target, error) {
  target.textContent = error.message;
}

// ---- New report

async function renderNew(route) {
  const options = await api("/api/options");
  const error = el("p", { class: "error", role: "alert" });
  const topic = el("textarea", { id: "topic", rows: "3", maxlength: "300", required: true });
  const capInput = el("input", { id: "cap", type: "number", min: "0.01", step: "0.05" });
  const confirm = el("input", { id: "confirm", type: "checkbox" });
  const spendText = el("label", { for: "cap" }, "Stop this report if it reaches");
  const spend = el(
    "div",
    { class: "spend", hidden: true },
    spendText,
    capInput,
    el("label", { for: "confirm" }, confirm, "I accept spending up to this limit on the Claude API."),
  );

  function engineChoice(engine) {
    const blocked = engine.needs_key && !options.key_present;
    const input = el("input", {
      type: "radio",
      name: "engine",
      value: engine.id,
      checked: engine.id === "local",
      disabled: blocked,
      on: {
        change: () => {
          spend.hidden = !engine.needs_key;
          capInput.value = engine.default_cap;
          capInput.max = options.max_cost_allowed;
          confirm.checked = false;
        },
      },
    });
    return el(
      "label",
      { class: "choice" },
      input,
      el(
        "span",
        {},
        el("strong", {}, engine.name),
        blocked ? "Needs an API key, which is not set." : engine.summary,
      ),
      el("span", { class: "price" }, engine.measured_cost),
    );
  }

  function depthChoice(depth) {
    return el(
      "label",
      { class: "choice" },
      el("input", {
        type: "radio",
        name: "depth",
        value: depth.iterations,
        checked: depth.iterations === 3,
      }),
      el("span", {}, el("strong", {}, depth.name), depth.note),
    );
  }

  const form = el(
    "form",
    {
      novalidate: true, // the server decides; a hidden limit field must not block silently
      on: {
        submit: async (event) => {
          event.preventDefault();
          error.textContent = "";
          if (topic.value.trim().length < 3) {
            error.textContent = "Describe the topic in a few words.";
            topic.focus();
            return;
          }
          const data = new FormData(form);
          const engine = options.engines.find((item) => item.id === data.get("engine"));
          try {
            await api("/api/jobs", {
              method: "POST",
              body: {
                topic: topic.value,
                engine: engine.id,
                iterations: Number(data.get("depth")),
                max_cost: engine.needs_key ? Number(capInput.value) : 0,
                confirm_spend: confirm.checked,
              },
            });
            location.hash = "#queue";
          } catch (problem) {
            showError(error, problem);
          }
        },
      },
    },
    el("label", { for: "topic" }, "What should the report cover?"),
    topic,
    el(
      "fieldset",
      {},
      el("legend", {}, "Who does the reasoning?"),
      el("div", { class: "choices" }, ...options.engines.map(engineChoice)),
    ),
    spend,
    el(
      "fieldset",
      {},
      el("legend", {}, "How deep?"),
      el("div", { class: "choices" }, ...DEPTHS.map(depthChoice)),
    ),
    error,
    el("button", { type: "submit" }, "Add to queue"),
  );
  setView(
    route,
    el("h1", {}, "New report"),
    el(
      "p",
      { class: "lede" },
      "Reports run one at a time, and only start once search is answering. You can close this tab.",
    ),
    form,
  );
  if (route === currentRoute) {
    topic.focus();
  }
}

// ---- Queue

function stageProgress(detail) {
  const done = new Set(
    detail.stages.filter((stage) => stage.status === "completed").map((stage) => stage.stage),
  );
  const current = detail.stages.filter((stage) => stage.status === "running").pop();
  return { done: done.size, current };
}

function jobActions(job, refresh, error) {
  const act = (label, path, method, quiet) =>
    el(
      "button",
      {
        type: "button",
        class: quiet ? "quiet" : "",
        on: {
          click: async () => {
            try {
              await api(path, { method });
              await refresh();
            } catch (problem) {
              showError(error, problem);
            }
          },
        },
      },
      label,
    );
  const open = ["queued", "waiting_search", "running"].includes(job.status);
  const actions = el("div", { class: "actions" });
  if (open) {
    actions.append(act("Stop", `/api/jobs/${job.id}/stop`, "POST", true));
  } else {
    if (job.status === "completed" && job.run_id) {
      actions.append(el("a", { href: `#report/${job.run_id}` }, "Read report"));
    }
    if (["failed", "stopped"].includes(job.status) && job.run_id) {
      actions.append(act("Resume", `/api/jobs/${job.id}/resume`, "POST", false));
    }
    actions.append(act("Remove", `/api/jobs/${job.id}`, "DELETE", true));
  }
  return actions;
}

async function jobCard(job, refresh) {
  const error = el("p", { class: "error", role: "alert" });
  const lines = [
    el("h2", {}, job.topic),
    el(
      "p",
      { class: "meta" },
      el("span", { class: `status status-${job.status}` }, JOB_WORDS[job.status]),
      ` · ${job.engine}`,
      job.max_cost > 0 ? ` · limit ${money(job.max_cost)}` : "",
    ),
  ];
  if (job.status === "running" && job.run_id) {
    const detail = await api(`/api/runs/${job.run_id}`);
    const { done, current } = stageProgress(detail);
    const parts = [
      current ? STAGE_WORDS[current.stage] || current.stage : "Starting",
      current ? `pass ${current.iteration + 1}` : null,
      job.started_at ? `${elapsed(job.started_at)} so far` : null,
      detail.n_urls ? `${detail.n_urls} sources found` : null,
      money(detail.cost_usd),
    ];
    lines.push(
      el("progress", { value: done, max: TOTAL_STAGES, "aria-label": "Stages finished" }),
      el("p", { class: "meta" }, parts.filter(Boolean).join(" · ")),
    );
  }
  if (job.note) {
    lines.push(el("p", { class: "meta" }, job.note));
  }
  lines.push(jobActions(job, refresh, error), error);
  return el("article", { class: "card" }, ...lines);
}

async function renderQueue(route) {
  const jobs = (await api("/api/jobs")).reverse();
  if (route !== currentRoute) {
    return;
  }
  const refresh = () => {
    queueShown = null;
    return renderQueue(route);
  };
  const cards = await Promise.all(jobs.map((job) => jobCard(job, refresh)));
  const shown = cards.map((card) => card.textContent).join("|");
  // Redrawing an unchanged list would swallow a click in progress.
  if (shown !== queueShown || view.dataset.page !== "queue") {
    queueShown = shown;
    view.dataset.page = "queue";
    setView(
      route,
      el("h1", {}, "Queue"),
      el(
        "p",
        { class: "lede" },
        "One report runs at a time, with a rest between them so search engines keep answering.",
      ),
      ...(cards.length ? cards : [el("p", { class: "muted" }, "Nothing queued yet.")]),
    );
  }
  if (route === currentRoute) {
    clearTimeout(timer);
    timer = setTimeout(() => renderQueue(route), 3000);
  }
}

async function refreshCount() {
  const badge = document.getElementById("queue-count");
  try {
    const jobs = await api("/api/jobs");
    const active = jobs.filter((job) =>
      ["queued", "waiting_search", "running"].includes(job.status),
    ).length;
    badge.textContent = active;
    badge.hidden = active === 0;
  } catch (_problem) {
    badge.hidden = true;
  }
}

// ---- Library

function runCard(run) {
  const title = run.has_report ? el("a", { href: `#report/${run.run_id}` }, run.topic) : run.topic;
  return el(
    "article",
    { class: "card" },
    el("h2", {}, title),
    el(
      "p",
      { class: "meta" },
      `${when(run.updated_at)} · ${money(run.cost_usd)} · ${run.run_id}`,
      run.has_report ? "" : ` · ${run.status}, no report`,
    ),
  );
}

async function renderLibrary(route) {
  const results = el("div", {});
  const search = el("input", {
    type: "search",
    placeholder: "Search topics and report text",
    "aria-label": "Search reports",
  });
  async function load() {
    const runs = await api(`/api/runs?q=${encodeURIComponent(search.value)}`);
    results.replaceChildren(
      ...(runs.length ? runs.map(runCard) : [el("p", { class: "muted" }, "No reports found.")]),
    );
  }
  let debounce = null;
  search.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(load, 250);
  });
  setView(
    route,
    el("h1", {}, "Library"),
    el("p", { class: "lede" }, "Every report on this machine, from the app or the terminal."),
    search,
    el("div", { class: "eyebrow" }),
    results,
  );
  await load();
}

// ---- Reader

function closePassage() {
  document.getElementById("passage").hidden = true;
}

async function openPassage(runId, number) {
  const panel = document.getElementById("passage");
  const title = document.getElementById("passage-title");
  const link = document.getElementById("passage-link");
  const text = document.getElementById("passage-text");
  document.getElementById("passage-number").textContent = `Source [${number}]`;
  panel.hidden = false;
  try {
    const found = await api(`/api/runs/${runId}/passages/${number}`);
    title.textContent = found.title;
    link.textContent = found.url;
    if (found.url) {
      link.href = found.url;
    } else {
      link.removeAttribute("href");
    }
    text.textContent = found.text || "The saved text for this source is not available.";
  } catch (problem) {
    title.textContent = "";
    link.textContent = "";
    link.removeAttribute("href");
    text.textContent = problem.message;
  }
}

async function renderReport(route, runId) {
  const [detail, report] = await Promise.all([
    api(`/api/runs/${runId}`),
    api(`/api/runs/${runId}/report`),
  ]);
  const body = el("div", { class: "report" });
  // Server-rendered with raw HTML escaped, because reports quote web pages.
  body.innerHTML = report.html;
  body.addEventListener("click", (event) => {
    const cite = event.target.closest(".cite");
    if (cite) {
      openPassage(runId, cite.dataset.n);
    }
  });
  const actions = el(
    "div",
    { class: "actions" },
    el("a", { href: "#library" }, "Back to library"),
    detail.has_pdf ? el("a", { href: `/api/runs/${runId}/pdf` }, "Download PDF") : null,
  );
  setView(route, actions, body);
  if (route === currentRoute) {
    view.focus();
  }
}

// ---- Setup

function checkRow(check) {
  const mark = { ok: "✓", missing: "✗", skip: "–" }[check.status] || "?";
  return el(
    "div",
    { class: "check" },
    el("span", { class: `mark-${check.status}`, "aria-label": check.status }, mark),
    el("strong", {}, check.name),
    el("span", { class: "detail" }, check.detail),
  );
}

async function renderSetup(route) {
  const message = el("p", { class: "meta", role: "status" });
  const checks = el("div", {}, el("p", { class: "muted" }, "Checking. The first time can take several seconds..."));
  async function load() {
    const setup = await api("/api/setup");
    checks.replaceChildren(...setup.checks.map(checkRow));
  }
  const startSearch = el(
    "button",
    {
      type: "button",
      class: "quiet",
      on: {
        click: async () => {
          message.textContent = "Starting search...";
          try {
            const result = await api("/api/setup/search", { method: "POST" });
            message.textContent = result.detail;
            await load();
          } catch (problem) {
            showError(message, problem);
          }
        },
      },
    },
    "Start search",
  );
  const testSearch = el(
    "button",
    {
      type: "button",
      class: "quiet",
      on: {
        click: async () => {
          message.textContent = "Testing search...";
          try {
            const health = await api("/api/search-health");
            message.textContent = health.healthy
              ? "Search is answering."
              : "Search returned too few results. Engines may be refusing queries for now.";
          } catch (problem) {
            showError(message, problem);
          }
        },
      },
    },
    "Test search",
  );
  setView(
    route,
    el("h1", {}, "Setup"),
    el("p", { class: "lede" }, "What a report needs on this machine, and how to fix what is missing."),
    checks,
    el("div", { class: "actions" }, startSearch, testSearch),
    message,
  );
  await load();
}

// ---- Routing

async function route() {
  currentRoute += 1;
  const token = currentRoute;
  clearTimeout(timer);
  view.dataset.page = "";
  closePassage();
  const hash = location.hash.replace(/^#/, "") || "new";
  const [name, argument] = hash.split("/");
  const section = name === "report" ? "library" : name;
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === section) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
  const pages = {
    new: renderNew,
    queue: renderQueue,
    library: renderLibrary,
    setup: renderSetup,
    report: (token) => renderReport(token, argument),
  };
  try {
    await (pages[name] || renderNew)(token);
  } catch (problem) {
    setView(
      token,
      el("h1", {}, "Something went wrong"),
      el("p", { class: "error" }, problem.message),
    );
  }
}

document.getElementById("passage-close").addEventListener("click", closePassage);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closePassage();
  }
});
window.addEventListener("hashchange", route);
setInterval(refreshCount, 5000);
refreshCount();
route();
