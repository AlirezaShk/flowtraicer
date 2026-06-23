"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

let cy = null;
let current = null; // currently rendered engagement

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
  current = await (await fetch(`/api/engagements/${id}`)).json();
  renderGraph(current);
  $("#detail-title").textContent = "Select a node";
  $("#detail-body").innerHTML = '<span class="muted">Click a step in the graph.</span>';
}

// The ordered list of executed step names, and the directed pairs between them.
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
  const stepByName = {};
  eng.steps.forEach((s) => (stepByName[s.name] = s));

  const elements = [];
  eng.topology.nodes.forEach((n) => {
    const step = stepByName[n.name];
    let cls = executed.has(n.name) ? "executed" : "idle";
    if (step && step.status === "failed") cls = "failed";
    if (n.is_global) cls += " global";
    elements.push({ data: { id: n.name, label: n.name, isGlobal: n.is_global }, classes: cls });
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

  cy.on("tap", "node", (evt) => showStep(stepByName[evt.target.id()], evt.target.id()));
}

function showStep(step, name) {
  $("#detail-title").textContent = name;
  if (!step) {
    $("#detail-body").innerHTML = '<span class="muted">This node was not executed in this engagement.</span>';
    return;
  }
  const rows = [
    ["status", step.status],
    ["duration", step.duration_ms != null ? step.duration_ms.toFixed(1) + " ms" : "—"],
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
    // If a new engagement appears, refresh the list.
    if (rec.type === "engagement_started") loadEngagements();
  };
}

loadEngagements();
connectStream();
