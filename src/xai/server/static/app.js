"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

let cy = null;
let current = null; // current engagement
let stepByName = {}; // name -> step (shared by graph + timeline)

async function loadEngagements() {
  const list = await (await fetch("/api/engagements")).json();
  const ul = $("#engagement-list");
  ul.innerHTML = "";
  list.reverse().forEach((e) => {
    const li = document.createElement("li");
    li.dataset.id = e.id;
    li.innerHTML =
      `<div class="eng-name">${esc(e.name)}</div>` +
      `<div class="eng-meta">${e.step_count} steps · ` +
      `<span class="pill ${e.status}">${e.status}</span></div>`;
    li.onclick = () => selectEngagement(e.id, li);
    ul.appendChild(li);
  });
  const first = ul.querySelector("li");
  if (first) first.click();
}

async function selectEngagement(id, li) {
  document.querySelectorAll("#engagement-list li").forEach((n) => n.classList.remove("active"));
  if (li) li.classList.add("active");

  const [eng, timeline] = await Promise.all([
    fetch(`/api/engagements/${id}`).then((r) => r.json()),
    fetch(`/api/engagements/${id}/timeline`).then((r) => r.json()),
  ]);
  current = eng;
  stepByName = {};
  eng.steps.forEach((s) => (stepByName[s.name] = s));

  renderGraph(eng);
  renderTimeline(timeline);
  $("#detail-title").textContent = "Select a node";
  $("#detail-body").innerHTML = '<span class="muted">Click a step in the graph or timeline.</span>';
}

// ---- graph ----------------------------------------------------------------

function executedInfo(eng) {
  const names = eng.steps.map((s) => s.name);
  const executed = new Set(names);
  const pairs = new Set();
  for (let i = 0; i + 1 < names.length; i++) pairs.add(names[i] + "→" + names[i + 1]);
  (eng.intent_switches || []).forEach((sw) => {
    if (sw.from_step) pairs.add(sw.from_step + "→" + sw.to_step);
  });
  return { executed, pairs };
}

function renderGraph(eng) {
  const { executed, pairs } = executedInfo(eng);

  const elements = [];
  eng.topology.nodes.forEach((n) => {
    const step = stepByName[n.name];
    let cls = executed.has(n.name) ? "executed" : "idle";
    if (step && step.status === "failed") cls = "failed";
    if (n.is_global) cls += " global";
    elements.push({ data: { id: n.name, label: n.name }, classes: cls });
  });
  eng.topology.edges.forEach((e) => {
    const taken = pairs.has(e.source + "→" + e.target);
    elements.push({
      data: { id: e.source + "_" + e.target, source: e.source, target: e.target, label: e.condition || "" },
      classes: taken ? "executed" : "idle",
    });
  });

  if (cy) cy.destroy();
  cy = cytoscape({
    container: $("#graph"),
    elements,
    style: [
      { selector: "node", style: {
          label: "data(label)", "text-valign": "center", color: "#fff",
          "font-size": 13, "font-weight": 600, width: 96, height: 40, shape: "round-rectangle",
          "background-color": "#3a434f", "border-width": 2, "border-color": "#2a3340",
      }},
      { selector: "node.executed", style: { "background-color": "#238636", "border-color": "#3fb950" } },
      { selector: "node.failed", style: { "background-color": "#8b1a17", "border-color": "#f85149" } },
      { selector: "node.global", style: { shape: "round-diamond", width: 120, height: 70, "background-color": "#6e40c9", "border-color": "#d2a8ff" } },
      { selector: "node.idle", style: { "background-color": "#1c2430", color: "#7d8794" } },
      { selector: "node.sel", style: { "border-width": 4, "border-color": "#4f9cff" } },
      { selector: "edge", style: {
          width: 2, "line-color": "#3a434f", "target-arrow-color": "#3a434f",
          "target-arrow-shape": "triangle", "curve-style": "bezier",
          label: "data(label)", "font-size": 10, color: "#7d8794", "text-background-color": "#0e1116",
          "text-background-opacity": 1, "text-background-padding": 2,
      }},
      { selector: "edge.executed", style: { width: 3, "line-color": "#3fb950", "target-arrow-color": "#3fb950" } },
    ],
    layout: { name: "breadthfirst", directed: true, spacingFactor: 1.3, padding: 24 },
  });

  cy.on("tap", "node", (evt) => selectStep(evt.target.id()));
}

// ---- timeline -------------------------------------------------------------

function renderTimeline(tv) {
  const total = tv.total_ms || 1;
  const el = $("#timeline");
  el.innerHTML = "";
  $("#timeline-total").textContent = `${tv.total_ms.toFixed(2)} ms`;

  if (tv.intent_switches.length) {
    const row = document.createElement("div");
    row.className = "tl-switch-row";
    tv.intent_switches.forEach((sw) => {
      const m = document.createElement("div");
      m.className = "tl-switch";
      m.style.left = (sw.offset_ms / total) * 100 + "%";
      m.title = `${sw.name} @ ${sw.offset_ms.toFixed(2)}ms`;
      row.appendChild(m);
    });
    el.appendChild(row);
  }

  tv.lanes.forEach((lane) => {
    const row = document.createElement("div");
    row.className = "tl-lane";
    row.dataset.step = lane.name;

    const label = document.createElement("div");
    label.className = "tl-label";
    label.textContent = lane.name;
    label.title = lane.name;

    const track = document.createElement("div");
    track.className = "tl-track";

    const bar = document.createElement("div");
    let cls = lane.status === "failed" ? "failed" : "executed";
    if (lane.is_global) cls = "global";
    bar.className = "tl-bar " + cls;
    bar.style.left = (lane.offset_ms / total) * 100 + "%";
    bar.style.width = Math.max(0.4, (lane.duration_ms / total) * 100) + "%";
    bar.title = `${lane.name} · ${lane.duration_ms.toFixed(2)}ms`;
    track.appendChild(bar);

    lane.events.forEach((ev) => {
      const mk = document.createElement("div");
      mk.className = "tl-mark " + ev.kind;
      mk.style.left = (ev.offset_ms / total) * 100 + "%";
      mk.title = `${ev.kind}: ${ev.name}`;
      track.appendChild(mk);
    });

    row.onclick = () => selectStep(lane.name);
    row.appendChild(label);
    row.appendChild(track);
    el.appendChild(row);
  });
}

// ---- shared selection (the link between graph and timeline) ---------------

function selectStep(name) {
  showStep(stepByName[name], name);
  if (cy) {
    cy.nodes().removeClass("sel");
    const node = cy.$id(name);
    if (node) node.addClass("sel");
  }
  document
    .querySelectorAll(".tl-lane")
    .forEach((row) => row.classList.toggle("sel", row.dataset.step === name));
}

function showStep(step, name) {
  $("#detail-title").textContent = name;
  if (!step) {
    $("#detail-body").innerHTML = '<span class="muted">This node was not executed in this engagement.</span>';
    return;
  }
  const rows = [
    ["status", step.status],
    ["duration", step.duration_ms != null ? step.duration_ms.toFixed(2) + " ms" : "—"],
    ["global", step.is_global ? "yes" : "no"],
    ["tools", step.tools_available.join(", ") || "—"],
  ];
  let html = rows.map(([k, v]) => `<div class="row"><span>${k}</span><span>${esc(v)}</span></div>`).join("");

  if (step.extraction) {
    html += `<h2>Extraction · ${esc(step.extraction.schema_name)}</h2>`;
    html += `<pre>${esc(JSON.stringify(step.extraction.values, null, 2))}</pre>`;
  }
  html += `<h2>Events (${step.events.length})</h2>`;
  step.events.forEach((ev) => {
    html += `<div class="event ${ev.kind}"><div class="k">${ev.kind} · ${esc(ev.name)}</div>` +
      (Object.keys(ev.payload || {}).length ? `<pre>${esc(JSON.stringify(ev.payload, null, 2))}</pre>` : "") +
      (ev.error ? `<pre>${esc(ev.error)}</pre>` : "") + `</div>`;
  });
  $("#detail-body").innerHTML = html;
}

// ---- live stream ----------------------------------------------------------

function connectStream() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
  const log = $("#stream-log");
  ws.onopen = () => $("#live").classList.add("on");
  ws.onclose = () => $("#live").classList.remove("on");
  ws.onmessage = (msg) => {
    const rec = JSON.parse(msg.data);
    const li = document.createElement("li");
    li.innerHTML = `<b>${esc(rec.type)}</b> · ${esc(rec.engagement_id)}`;
    log.prepend(li);
    while (log.children.length > 40) log.removeChild(log.lastChild);
    if (rec.type === "engagement_started") loadEngagements();
  };
}

loadEngagements();
connectStream();
