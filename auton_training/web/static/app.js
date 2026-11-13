"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = v => v == null ? "–" : (100 * v).toFixed(0) + "%";
const num = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const big = v => v >= 1e9 ? (v / 1e9).toFixed(2) + " B" : v >= 1e6 ? (v / 1e6).toFixed(0) + " M" : String(v);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const icon = (id, s = 14) => `<svg width="${s}" height="${s}" aria-hidden="true"><use href="#i-${id}"/></svg>`;
const ARROW_UP = `<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1.5L9 8H1z" fill="currentColor"/></svg>`;
const ARROW_DN = `<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 8.5L1 2h8z" fill="currentColor"/></svg>`;

// A run keeps its colour and its name whatever else is on the page
const RUNS = {
  body_priv: { label: "Gate + privileged critic", color: "--s1" },
  traves:    { label: "Traves specialist",        color: "--s2" },
  body:      { label: "Gate, plain critic",       color: "--s3" },
  nogate:    { label: "No gate",                  color: "--s4" },
  ten:       { label: "10 swings, costly misses", color: "--s5" },
  vertdrum:  { label: "Vert and drum specialist", color: "--s6" },
  vdhead:    { label: "Vert and drum, heading known (test)", color: "--s3" },
};
const runKey = name => name.replace(/^saw_v8a_/, "");
const runLabel = name => (RUNS[runKey(name)] || {}).label || name;
const runColor = name => `var(${(RUNS[runKey(name)] || { color: "--s6" }).color})`;
const runOrder = name => { const i = Object.keys(RUNS).indexOf(runKey(name)); return i < 0 ? 99 : i; };
const PANELS = { T: "Traves-like spinners", F: "Field", L: "Old field (saw-v7)" };
const PANEL_SHORT = { T: "Traves-like", F: "Field", L: "Old field" };

const state = { training: null, yard: null, replays: null, smooth: 10, tab: "overview", heroShown: false };

// ───────────────────────────── data ─────────────────────────────
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}
function setSync(ok, text) {
  const el = $("#refreshed");
  el.className = "sync " + (ok ? "ok" : "err");
  $("span", el).textContent = text;
}
async function refresh() {
  try {
    const [t, y, r] = await Promise.all([getJSON("/api/training"), getJSON("/api/yardstick"), getJSON("/api/replays")]);
    state.training = t; state.yard = y;
    const changed = !state.replays || JSON.stringify(state.replays.replays.map(x => x.file)) !== JSON.stringify(r.replays.map(x => x.file));
    state.replays = r;
    setSync(true, "Updated " + new Date().toLocaleTimeString());
    if (state.tab === "replays" && R.playing) state.stale = true;
    else { renderOverview(); renderTraining(); renderYardstick(); }
    if (changed) renderFightList();
  } catch (e) {
    console.error(e);
    setSync(false, "Could not reach the viewer server");
  }
}
const tail = (arr, n = 10) => { const v = arr.filter(x => x != null).slice(-n); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
function smooth(xs, ys, w) {
  const out = [];
  for (let i = 0; i < ys.length; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - w + 1); j <= i; j++) if (ys[j] != null) { s += ys[j]; c++; }
    if (ys[i] != null && c) out.push([xs[i], s / c]);
  }
  return out;
}
const standard = r => r.settings.swings === 10 && r.settings.heading === "none" && r.settings.reach === 0.5;
function baseline() {
  return (state.yard?.reports || []).find(r => r.run === "saw_v7_run7" && standard(r) && r.settings.gate === "on" && r.gate_rule === "body");
}
function best() {
  const all = (state.yard?.reports || []).filter(r => r.run.startsWith("saw_v8a") && standard(r) && r.panels.T && r.panels.F);
  const full = all.filter(r => r.settings.n >= 512);
  const pool = full.length ? full : all;
  const score = r => r.panels.T.poc_share + r.panels.F.poc_share;
  return [...pool].sort((a, b) => score(b) - score(a) || b.update - a.update)[0];
}

// ───────────────────────────── overview ─────────────────────────────
// a delta chip: arrow follows the sign, colour follows whether that is good
function chip(now, was, unit, lowerIsBetter) {
  if (was == null || now == null) return "";
  const d = now - was, good = lowerIsBetter ? d < 0 : d > 0;
  const txt = unit === "pt" ? Math.abs(100 * d).toFixed(0) + " pts" : Math.abs(d).toFixed(unit === "n1" ? 1 : 2);
  return `<span class="chip ${good ? "up" : "down"}">${d >= 0 ? ARROW_UP : ARROW_DN}${txt}</span>`;
}
function countTo(el, target, decimals = 0) {
  el.textContent = target.toFixed(decimals);           // the resting value, even if no frame ever runs
  if (reduceMotion() || state.heroShown || document.visibilityState !== "visible") return;
  const t0 = performance.now(), dur = 1100;
  const step = now => {
    const u = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - u, 3);
    el.textContent = (target * e).toFixed(decimals);
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function tile(label, value, unit, deltaHtml, wasTxt) {
  return `<div class="tile panel"><div class="tile-label">${label}</div><div class="tile-value">${value}${unit ? `<small>${unit}</small>` : ""}</div>
    <div class="tile-delta">${deltaHtml || ""}${wasTxt ? `<span>against ${wasTxt} for the starting policy</span>` : "&nbsp;"}</div></div>`;
}
function renderOverview() {
  const b = best(), base = baseline();
  if (!b) {
    $("#hero-num").textContent = "–";
    $("#hero-delta").innerHTML = `<span>No yardstick scores yet.</span>`;
    $("#board").innerHTML = `<div class="empty">No yardstick scores yet.</div>`;
    $("#tiles").innerHTML = "";
  } else {
    const P = b.panels, B = base ? base.panels : {}, quick = b.settings.n < 512, col = runColor(b.run);
    countTo($("#hero-num"), 100 * P.T.poc_share);
    $("#hero-delta").innerHTML = B.T ? `${chip(P.T.poc_share, B.T.poc_share, "pt")}<span>against ${pct(B.T.poc_share)} for the starting policy behind the same gate</span>` : "";
    $("#hero-note").textContent = `Best scored checkpoint: ${runLabel(b.run)}, update ${b.update}, ${b.settings.n} fights per panel${quick ? " (quick score, about ±8 points)" : ""}.`;
    $("#board-now-key").style.background = col;
    const rows = ["T", "F", "L"].filter(k => P[k]).map(k => {
      const now = P[k].poc_share, was = B[k]?.poc_share;
      const band = quick ? `<span class="qs" style="left:${Math.max(0, 100 * now - 8)}%"></span><span class="qs" style="left:${Math.min(100, 100 * now + 8)}%"></span>` : "";
      return `<div class="brow">
        <div class="bname">${PANEL_SHORT[k]}<small>${P[k].n_fights} fights</small></div>
        <div class="bbar"><i style="width:${100 * now}%;background:${col}"></i>${band}</div>
        <div class="bval">${pct(now)}</div>
        <div class="bbar thin">${was != null ? `<i style="width:${100 * was}%"></i>` : ""}</div>
        <div class="bval sm">${was != null ? pct(was) + " " + chip(now, was, "pt").replace("chip ", "chip mini ") : "–"}</div>
      </div>`;
    }).join("");
    $("#board").classList.toggle("anim", !state.heroShown);
    $("#board").innerHTML = rows +`<div class="board-scale"><span></span><div>${[0, 25, 50, 75, 100].map(v => `<span style="left:${v}%">${v}%</span>`).join("")}</div><span></span></div>`;
    $("#tiles").innerHTML =
        tile("Solid legal strikes per fight · Traves-like", num(P.T.solid_per_fight), "per fight", chip(P.T.solid_per_fight, B.T?.solid_per_fight, "n"), B.T ? num(B.T.solid_per_fight) : "")
      + tile("Swings landing on a legal side · Traves-like", (100 * P.T.land_legal).toFixed(1), "%", chip(P.T.land_legal, B.T?.land_legal, "pt"), B.T ? (100 * B.T.land_legal).toFixed(1) + "%" : "")
      + tile("Hits taken per fight · Traves-like", num(P.T.hits_taken, 1), "per fight", chip(P.T.hits_taken, B.T?.hits_taken, "n1", true), B.T ? num(B.T.hits_taken, 1) : "");
    state.heroShown = true;
  }
  renderRunCards();
}

function sparkline(pts, color, id) {
  const W = 300, H = 56, p = 3;
  if (pts.length < 2) return `<svg class="spark" viewBox="0 0 ${W} ${H}"></svg>`;
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys, 1e-9);
  const X = x => p + (x - x0) / Math.max(1, x1 - x0) * (W - 2 * p - 4), Y = y => H - p - y / y1 * (H - 2 * p - 4);
  const line = pts.map(q => X(q[0]).toFixed(1) + " " + Y(q[1]).toFixed(1)).join("L");
  const last = pts.at(-1);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".32"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
    <path d="M${X(x0)} ${H - p}L${line}L${X(last[0])} ${H - p}Z" fill="url(#${id})"/>
    <path d="M${line}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
    <line x1="${X(x0)}" x2="${W}" y1="${H - p}" y2="${H - p}" stroke="var(--axis)" vector-effect="non-scaling-stroke"/>
    <circle cx="${X(last[0])}" cy="${Y(last[1])}" r="3.5" fill="${color}" stroke="var(--surface)" stroke-width="2" vector-effect="non-scaling-stroke"/>
  </svg>`;
}
function renderRunCards() {
  const runs = [...(state.training?.runs || [])].sort((a, b) => runOrder(a.name) - runOrder(b.name));
  if (!runs.length) { $("#run-cards").innerHTML = `<div class="empty">No training logs found.</div>`; return; }
  $("#run-cards").innerHTML = runs.map(r => {
    const s = r.series, off = r.offset || 0, st = r.crashed ? ["crashed", "Crashed"] : r.live ? ["live", "Training"] : ["", "Stopped"];
    const pts = smooth(s.update.map(u => u + off), s.solid, 10), col = runColor(r.name);
    return `<article class="run panel" style="--rc:${col}">
      <div class="run-top"><div><div class="run-name">${esc(runLabel(r.name))}</div><div class="run-id">${esc(r.name)}${off ? ` · from update ${off}` : ""}</div></div>
        <span class="pill ${st[0]}"><i></i>${st[1]}</span></div>
      <div class="run-upd"><b>${s.update.at(-1) + off}</b><span>update · ${big(s.steps.at(-1))} fight steps${r.live ? ` · ${Math.round(tail(s.sps, 5) / 1000)}k steps/s` : ""}</span></div>
      <div>${sparkline(pts, col, "sp-" + runKey(r.name))}<div class="spark-cap"><span>Solid legal strikes per 3-min fight</span><span>now ${num(tail(s.solid))}</span></div></div>
      <div class="run-stats">
        <div><b>${pct(tail(s.win))}</b><span>wins</span></div>
        <div><b>${pct(tail(s.legal))}</b><span>swings landing legal</span></div>
        <div><b>${pct(tail(s.side))}</b><span>on the weapon side</span></div>
        <div><b>${num(tail(s.taken), 1)}</b><span>damage taken per fight</span></div>
      </div>
    </article>`;
  }).join("");
}

// ───────────────────────────── charts ─────────────────────────────
function niceTicks(lo, hi, n = 4) {
  if (hi <= lo) hi = lo + 1;
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}
// spread labels vertically so none overlap (min gap g), keeping them inside [lo, hi]
function spread(items, g, lo, hi) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < g) items[i].y = items[i - 1].y + g;
  const over = items.length ? items.at(-1).y - hi : 0;
  if (over > 0) for (let i = items.length - 1; i >= 0; i--) { items[i].y -= over; if (i && items[i].y - items[i - 1].y >= g) break; if (i) items[i - 1].y = Math.min(items[i - 1].y, items[i].y - g); }
  for (const it of items) it.y = Math.max(lo, it.y);
  return items;
}
let gradId = 0;
// series: [{key, name, color, pts: [[x, y]...], marks: [{x, y, filled, note}]}]
function chart(host, cfg) {
  const W = Math.max(280, host.clientWidth - 36), H = 240, m = { l: 40, r: 50, t: 12, b: 28 };
  const all = cfg.series.flatMap(s => s.pts);
  const head = `<div class="chart-head"><div><h3>${esc(cfg.title)}</h3><div class="sub">${esc(cfg.sub || "")}</div></div></div>`;
  if (!all.length) { host.innerHTML = head + `<div class="empty">No data yet.</div>`; return; }
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]).concat((cfg.refs || []).map(r => r.y));
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = cfg.zero ? 0 : Math.min(...ys), y1 = Math.max(...ys);
  const yt = niceTicks(y0, y1 * 1.05 + 1e-9); y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt.at(-1));
  const X = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), Y = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  const fmt = cfg.fmt || (v => num(v, 1));
  let g = "", defs = "";
  for (const t of yt) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${fmt(t)}</text>`;
  for (const t of niceTicks(x0, x1, 5)) if (t >= x0 && t <= x1) g += `<text x="${X(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`;
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  for (const r of cfg.refs || []) g += `<g class="ref"><line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="var(--ink-2)" stroke-dasharray="5 4" opacity=".8"/><text x="${m.l + 6}" y="${Y(r.y) - 6}">${esc(r.label)}</text></g>`;
  const ends = [];
  for (const s of cfg.series) {
    if (!s.pts.length) continue;
    let sg = "";
    const d = s.pts.map(p => X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("L");
    if (s.pts.length > 1) {
      if (!cfg.points) {
        const id = "ag" + (++gradId);
        defs += `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${s.color}" stop-opacity=".10"/><stop offset="1" stop-color="${s.color}" stop-opacity="0"/></linearGradient>`;
        sg += `<path d="M${X(s.pts[0][0]).toFixed(1)} ${H - m.b}L${d}L${X(s.pts.at(-1)[0]).toFixed(1)} ${H - m.b}Z" fill="url(#${id})"/>`;
      }
      sg += `<path d="M${d}" fill="none" stroke="${s.color}" stroke-width="${cfg.points ? 1.5 : 2}" stroke-linejoin="round" stroke-linecap="round"${cfg.points ? ' opacity=".7"' : ""}/>`;
    }
    for (const k of s.marks || []) sg += `<circle cx="${X(k.x)}" cy="${Y(k.y)}" r="4.5" fill="${k.filled ? s.color : "var(--surface)"}" stroke="${k.filled ? "var(--surface)" : s.color}" stroke-width="2"/>`;
    const last = s.pts.at(-1);
    if (!cfg.points) sg += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
    ends.push({ y: Y(last[1]), v: fmt(last[1]), color: s.color, key: s.key });
    g += `<g class="s" data-k="${esc(s.key)}">${sg}</g>`;
  }
  for (const e of spread(ends, 14, m.t, H - m.b)) g += `<g class="s" data-k="${esc(e.key)}"><line x1="${W - m.r + 3}" x2="${W - m.r + 9}" y1="${e.y}" y2="${e.y}" stroke="${e.color}" stroke-width="2"/><text class="end" x="${W - m.r + 12}" y="${e.y + 4}">${esc(e.v)}</text></g>`;
  g += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink-2)" stroke-dasharray="2 3" visibility="hidden"/><g class="dots"></g>
    <g class="xtag" visibility="hidden"><rect x="-22" y="${H - m.b + 3}" width="44" height="17" rx="2"/><text x="0" y="${H - m.b + 15}" text-anchor="middle"></text></g>
    <rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  host.innerHTML = head + `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(cfg.title)}"><defs>${defs}</defs>${g}</svg>`;
  const svg = $("svg", host), cross = $(".cross", svg), dots = $(".dots", svg), xtag = $(".xtag", svg), tip = $("#tip");
  const hide = () => { cross.setAttribute("visibility", "hidden"); xtag.setAttribute("visibility", "hidden"); dots.innerHTML = ""; tip.hidden = true; };
  $(".hit", svg).addEventListener("mouseleave", hide);
  $(".hit", svg).addEventListener("mousemove", ev => {
    const box = svg.getBoundingClientRect(), px = (ev.clientX - box.left) * W / box.width, py = (ev.clientY - box.top) * H / box.height;
    let rows = [], title = "", cx = px, xv;
    if (cfg.points) {                                   // nearest marker
      let near = null;
      for (const s of cfg.series) for (const k of s.marks || []) {
        const d = Math.hypot(X(k.x) - px, Y(k.y) - py);
        if (!near || d < near.d) near = { d, s, k };
      }
      if (!near || near.d > 40) return hide();
      cx = X(near.k.x); xv = near.k.x; title = `Update ${near.k.x}`;
      rows = [[near.s.color, near.s.name, fmt(near.k.y)], ...(near.k.note || []).map(n => [null, n[0], n[1]])];
      dots.innerHTML = `<circle cx="${cx}" cy="${Y(near.k.y)}" r="8" fill="none" stroke="${near.s.color}" stroke-width="2"/>`;
    } else {                                            // crosshair: every series at the nearest update
      const want = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
      let nearest = null;
      for (const s of cfg.series) for (const p of s.pts) if (nearest == null || Math.abs(p[0] - want) < Math.abs(nearest - want)) nearest = p[0];
      cx = X(nearest); xv = nearest; title = "Update " + nearest; let d = "";
      const found = [];
      for (const s of cfg.series) {
        const p = s.pts.find(q => q[0] === nearest);
        if (!p) continue;
        found.push([s.color, s.name, p[1]]);
        d += `<circle cx="${cx}" cy="${Y(p[1])}" r="4.5" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
      }
      rows = found.sort((a, b) => b[2] - a[2]).map(r => [r[0], r[1], fmt(r[2])]);
      dots.innerHTML = d;
    }
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    xtag.setAttribute("transform", `translate(${Math.min(W - m.r - 22, Math.max(m.l + 22, cx))},0)`); $("text", xtag).textContent = xv; xtag.setAttribute("visibility", "visible");
    tip.innerHTML = `<div class="t">${esc(title)} · ${esc(cfg.title)}</div><div class="rows">` + rows.map(r => `<div class="r${r[0] ? "" : " sub"}"><span>${r[0] ? `<i class="sw" style="background:${r[0]}"></i>` : ""}${esc(r[1])}</span><b>${esc(r[2])}</b></div>`).join("") + "</div>";
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight, left = ev.clientX + 18 + tw > innerWidth ? ev.clientX - tw - 18 : ev.clientX + 18;
    tip.style.left = left + "px"; tip.style.top = Math.min(innerHeight - th - 8, Math.max(8, ev.clientY - 24)) + "px";
  });
}
const legend = names => names.map(n => `<span data-k="${esc(runKey(n))}"><i class="line" style="background:${runColor(n)}"></i>${esc(runLabel(n))}</span>`).join("");
// hovering a legend entry brings that run forward in every chart of the section
function wireLegend(el) {
  const grid = () => document.getElementById(el.dataset.for);
  el.addEventListener("mouseover", ev => {
    const k = ev.target.closest("[data-k]")?.dataset.k; if (!k) return;
    for (const host of [el, grid()]) { host.classList.add("focusing"); $$("[data-k]", host).forEach(x => x.classList.toggle("on", x.dataset.k === k)); }
  });
  el.addEventListener("mouseleave", () => { for (const host of [el, grid()]) host.classList.remove("focusing"); });
}
$$(".legend[data-for]").forEach(wireLegend);

const TRAIN_CHARTS = [
  ["win", "Wins", "share of finished fights won on damage", pct, true],
  ["solid", "Solid legal strikes", "per three-minute training fight", v => num(v, 1), true],
  ["legal", "Swings landing on a legal side", "share of swings in three-minute fights", pct, true],
  ["side", "Swings landing on the weapon side", "share of swings in three-minute fights; lower is better", pct, true],
  ["taken", "Damage taken", "per three-minute training fight; lower is better", v => num(v, 1), true],
  ["ret", "Reward", "average per control step", v => num(v, 3), false],
];
function renderTraining() {
  const runs = [...(state.training?.runs || [])].sort((a, b) => runOrder(a.name) - runOrder(b.name)), host = $("#train-charts");
  $("#train-legend").innerHTML = legend(runs.map(r => r.name));
  if (host.children.length !== TRAIN_CHARTS.length) host.innerHTML = TRAIN_CHARTS.map(() => `<div class="panel chart"></div>`).join("");
  if ($("#training").hidden || !state.training) return;
  TRAIN_CHARTS.forEach(([key, title, sub, fmt, zero], i) => chart(host.children[i], {
    title, sub, fmt, zero,
    series: runs.map(r => ({ key: runKey(r.name), name: runLabel(r.name), color: runColor(r.name), pts: smooth(r.series.update.map(u => u + (r.offset || 0)), r.series[key], state.smooth) })),
  }));
}

function renderYardstick() {
  const reports = state.yard?.reports || [], base = baseline();
  const mine = reports.filter(r => r.run.startsWith("saw_v8a") && standard(r));
  const names = [...new Set(mine.map(r => r.run))].sort((a, b) => runOrder(a) - runOrder(b));
  $("#yard-legend").innerHTML = legend(names)
    + `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--ink-2)"/></svg>512 fights</span>`
    + `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--ink-2)" stroke-width="2"/></svg>128 fights (quick)</span>`
    + `<span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="var(--ink-2)" stroke-dasharray="5 4"/></svg>Starting policy behind the same gate</span>`;
  const host = $("#yard-charts");
  if (host.children.length !== 3) host.innerHTML = `<div class="panel chart"></div>`.repeat(3);
  if (!$("#yardstick").hidden && state.yard) ["T", "F", "L"].forEach((k, i) => chart(host.children[i], {
    title: PANELS[k], sub: "fights with a solid legal strike", fmt: pct, zero: true, points: true,
    refs: base?.panels[k] ? [{ y: base.panels[k].poc_share, label: "starting policy " + pct(base.panels[k].poc_share) }] : [],
    series: names.map(n => {
      const rs = mine.filter(r => r.run === n && r.panels[k]).sort((a, b) => a.update - b.update || a.settings.n - b.settings.n);
      return { key: runKey(n), name: runLabel(n), color: runColor(n), pts: rs.map(r => [r.update, r.panels[k].poc_share]),
        marks: rs.map(r => ({ x: r.update, y: r.panels[k].poc_share, filled: r.settings.n >= 512,
          note: [["Fights scored", String(r.settings.n)], ["Solid strikes per fight", num(r.panels[k].solid_per_fight)],
                 ["Swings landing legal", pct(r.panels[k].land_legal)], ["Hits taken per fight", num(r.panels[k].hits_taken, 1)]] })) };
    }),
  }));
  const gate = r => !r.arm_gate ? "off" : "on, " + (r.gate_rule || "heading") + " rule";
  const share = v => v == null ? "–" : `<span class="cellbar">${pct(v)}<i><b style="width:${100 * v}%"></b></i></span>`;
  const rows = [...reports].sort((a, b) => runOrder(a.run) - runOrder(b.run) || a.run.localeCompare(b.run) || b.update - a.update || b.settings.n - a.settings.n);
  $("#yard-table").innerHTML = `<thead><tr><th>Run</th><th>Update</th><th>Fights per panel</th><th class="left">Arm gate</th><th class="left">Heading reading</th><th>Traves-like</th><th>Field</th><th>Old field</th><th>Solid strikes per<br>Traves-like fight</th><th>Swings legal<br>(Traves-like)</th><th>Hits taken<br>(Traves-like)</th></tr></thead><tbody>`
    + rows.map(r => { const T = r.panels.T || {}, v8 = r.run.startsWith("saw_v8a"); return `<tr><td><i class="swatch" style="background:${v8 ? runColor(r.run) : "var(--muted)"}"></i>${esc(v8 ? runLabel(r.run) : "Starting policy (saw-v7 run 7)")}</td>
      <td class="n">${r.update}</td><td><span class="qtag ${r.settings.n >= 512 ? "full" : ""}">${r.settings.n >= 512 ? "Full" : "Quick"} ${r.settings.n}</span></td><td class="left">${gate(r)}</td><td class="left">${r.settings.heading === "none" ? "none" : esc(r.settings.heading)}</td>
      <td>${share(r.panels.T?.poc_share)}</td><td>${share(r.panels.F?.poc_share)}</td><td>${share(r.panels.L?.poc_share)}</td>
      <td class="n">${num(T.solid_per_fight)}</td><td class="n">${pct(T.land_legal)}</td><td class="n">${num(T.hits_taken, 1)}</td></tr>`; }).join("") + "</tbody>";
}

// ───────────────────────────── replays ─────────────────────────────
const R = { doc: null, t: 0, playing: false, speed: 1, last: 0, swings: [], hits: [], file: null, follow: true, cam: null, dmgMax: 1, arenaFailed: false };
const C = {};
const mmss = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const OUTCOME = { solid: ["solid", "Solid legal strike"], weak: ["weak", "Legal, weak"], side: ["side", "Weapon side"], floor: ["floor", "Floor"] };
const theme = () => { const d = document.documentElement.dataset.theme; return d || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"); };
const weaponName = e => e.traves ? "Traves-like bar" : e.weapon === "none" ? "dead weapon" : e.weapon + " spinner";

function renderFightList() {
  const list = state.replays?.replays || [], host = $("#fight-list");
  $("#fight-count").textContent = list.length ? `${list.length} fight${list.length === 1 ? "" : "s"}` : "";
  if (!list.length) { host.innerHTML = `<div class="empty">No replays yet. Make some with<br><code>uv run python web/export_replays.py CHECKPOINT</code></div>`; return; }
  let html = "", group = "";
  for (const f of [...list].sort((a, b) => b.update - a.update || a.run.localeCompare(b.run) || a.file.localeCompare(b.file))) {
    const g = `${runLabel(f.run)}|${f.update}`;
    if (g !== group) { html += `<div class="group"><b>${esc(runLabel(f.run))}</b> update ${f.update}</div>`; group = g; }
    const s = f.summary, pips = [];
    for (let i = 0; i < Math.round(s.solid); i++) pips.push("solid");
    for (let i = 0; i < Math.round(s.side); i++) pips.push("side");
    for (let i = 0; i < Math.round(s.floor); i++) pips.push("floor");
    while (pips.length < (f.swings || 10)) pips.push("");
    html += `<button class="fight" data-file="${esc(f.file)}" title="${s.solid.toFixed(0)} solid, ${s.side.toFixed(0)} weapon side, ${s.floor.toFixed(0)} floor, ${s.hits_taken.toFixed(0)} hits taken">
      <div class="a"><b>${esc(f.enemy.tactic)} · ${esc(weaponName(f.enemy))}</b><span>${esc(f.panel_name)} ${f.fight}</span></div>
      <div class="b"><span class="mini-pips">${pips.slice(0, f.swings || 10).map(k => `<i class="${k}"></i>`).join("")}</span><span>${s.solid.toFixed(0)} solid · ${s.hits_taken.toFixed(0)} hits taken</span></div></button>`;
  }
  host.innerHTML = html;
  $$(".fight", host).forEach(b => b.addEventListener("click", () => loadFight(b.dataset.file)));
  const q = new URLSearchParams(location.search), want = q.get("fight");
  if (!R.doc) loadFight(list.some(f => f.file === want) ? want : $(".fight", host).dataset.file).then(() => { if (q.get("t")) { R.t = +q.get("t"); draw(); } });
  else markCurrent();
}
const markCurrent = () => $$(".fight").forEach(b => b.setAttribute("aria-current", b.dataset.file === R.file));

async function loadFight(file) {
  if (LIVE.on) stopLive();
  const doc = await getJSON("/replays/" + encodeURIComponent(file));
  R.doc = doc; R.file = file; R.t = 0; R.cam = null; setPlaying(false);
  if (window.ArenaRenderer?.reset) try { window.ArenaRenderer.reset(); } catch (e) { console.error(e); }
  indexFight(doc);
  showFight(doc);
}
function indexFight(doc) {
  const byFrame = new Map();
  R.hits = [];
  for (const [f, kind] of doc.events) {
    if (kind === "hit") { R.hits.push(f); continue; }
    if (kind === "swing") continue;
    const set = byFrame.get(f) || new Set(); set.add(kind); byFrame.set(f, set);
  }
  R.swings = [...byFrame].sort((a, b) => a[0] - b[0]).map(([f, set]) =>
    ({ f, t: f * doc.dt, kind: set.has("solid") ? "solid" : set.has("legal") ? "weak" : set.has("side") ? "side" : "floor" }));
  R.fired = doc.events.filter(ev => ev[1] === "swing").map(ev => ev[0]).sort((a, b) => a - b);
  const lastF = doc.frames.at(-1);
  R.dmgMax = Math.max(1, lastF[10], lastF[11]);
}
function showFight(doc) {
  const e = doc.enemy;
  R.liveHtml = R.logKey = null;
  $("#fight-title").textContent = doc.live ? "You, driving a Traves-like horizontal bar" : `${e.tactic} with a ${e.traves ? "Traves-like horizontal bar" : e.weapon === "none" ? "dead weapon" : e.weapon + " spinner"}`;
  $("#fight-sub").textContent = `${runLabel(doc.run)}, update ${doc.update} · ${doc.panel_name} panel, fight ${doc.fight} · opponent level ${e.level.toFixed(2)}, weakness: ${e.weakness.replace("_", " ")} · ${doc.swings} swings, arm gate ${doc.arm_gate ? "on" : "off"}`;
  $("#bug-en-name").textContent = e.tactic;
  const total = (doc.frames.length - 1) * doc.dt;
  $("#bug-total").textContent = "of " + mmss(doc.live ? doc.seconds : total).replace(/\.0$/, "");
  $("#ticks").innerHTML = R.swings.map(s => `<i class="${s.kind}" style="left:${100 * s.t / total}%" title="${OUTCOME[s.kind][1]} at ${mmss(s.t)}">${icon(s.kind, 12)}</i>`).join("");
  $("#hit-ticks").innerHTML = R.hits.map(f => `<i style="left:${100 * f * doc.dt / total}%"></i>`).join("");
  const marks = []; for (let s = 0; s <= total + 1e-6; s += 30) marks.push(s);
  $("#tl-scale").innerHTML = marks.map(s => `<span style="left:${100 * s / total}%">${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}</span>`).join("");
  $("#bug-pips").innerHTML = Array.from({ length: doc.swings }, () => "<i></i>").join("");
  $("#swing-log").innerHTML = R.swings.length ? R.swings.map((s, i) => `<button class="${s.kind}" data-i="${i}"><span class="time">${mmss(s.t)}</span>${icon(s.kind, 16)}<span>${OUTCOME[s.kind][1]}</span></button>`).join("")
    : `<div class="empty">No swing landed in this fight.</div>`;
  $$("#swing-log button").forEach(b => b.addEventListener("click", () => { R.t = Math.max(0, R.swings[+b.dataset.i].t - 2); R.cam = null; draw(); setPlaying(true); }));
  markCurrent(); sizeArena(); layoutTicks(); draw();
}
// swings closer than a marker's width go to the upper lane
function layoutTicks() {
  if (!R.doc) return;
  const w = $("#tl-track").clientWidth, total = (R.doc.frames.length - 1) * R.doc.dt, last = [-1e9, -1e9];
  $$("#ticks i").forEach((el, i) => {
    const x = w * R.swings[i].t / total;
    const lane = x - last[0] >= 18 ? 0 : x - last[1] >= 18 ? 1 : 0;
    last[lane] = x; el.classList.toggle("l1", lane === 1);
  });
}
// ───────────────────────────── fight the policy ─────────────────────────────
// A person drives the opponent with the keyboard. The server steps the sim (web/play.py) and sends the new
// frames; the fight is drawn a moment behind them so it moves smoothly, and stays as a replay once it ends.
const LIVE = { on: false, keys: new Set(), busy: false, timer: 0, last: 0 };
const postJSON = async (url, body) => {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const doc = await r.json(); if (!r.ok || doc.error) throw new Error(doc.error || r.status); return doc;
};
async function startLive() {
  const btn = $("#fight-live");
  btn.disabled = true; btn.textContent = "Loading the sim…";
  try {
    const doc = await postJSON("/api/play/start", {});
    if (window.ArenaRenderer?.reset) window.ArenaRenderer.reset();
    R.doc = doc; R.file = null; R.t = 0; R.cam = null;
    R.follow = false;                                          // a live fight opens on the whole arena
    $$("#view button").forEach(x => x.setAttribute("aria-pressed", x.dataset.v === "0"));
    indexFight(doc); showFight(doc);
    LIVE.on = true; LIVE.keys.clear(); LIVE.last = performance.now();
    btn.textContent = "End the fight"; btn.setAttribute("aria-pressed", "true");
    $("#live-hint").hidden = false; $("#seek").disabled = true;
    setPlaying(true);
    LIVE.timer = setInterval(tickLive, 33);                    // one request per control step: keys are read 30 times a second
  } catch (e) { console.error(e); btn.textContent = "Could not start: " + e.message; setTimeout(() => { btn.textContent = "Fight the policy"; }, 4000); }
  btn.disabled = false;
}
function stopLive() {
  if (!LIVE.on) return;
  LIVE.on = false; clearInterval(LIVE.timer); LIVE.keys.clear();
  const btn = $("#fight-live"); btn.textContent = "Fight the policy"; btn.setAttribute("aria-pressed", "false");
  $("#live-hint").hidden = true; $("#seek").disabled = false;
  setPlaying(false);
  if (R.doc) { showFight(R.doc); draw(); }
}
async function tickLive() {
  if (!LIVE.on || LIVE.busy || !R.doc) return;
  const now = performance.now(), steps = Math.max(1, Math.min(8, Math.round((now - LIVE.last) / 1000 / R.doc.dt)));
  LIVE.last = now; LIVE.busy = true;
  const k = LIVE.keys, has = (...c) => c.some(x => k.has(x));
  const throttle = (has("KeyW", "ArrowUp") ? 1 : 0) - (has("KeyS", "ArrowDown") ? 1 : 0);
  const turn = (has("KeyD", "ArrowRight") ? 1 : 0) - (has("KeyA", "ArrowLeft") ? 1 : 0);
  try {
    const out = await postJSON("/api/play/step", { throttle, turn, steps });
    if (!LIVE.on) return;
    const doc = R.doc;
    for (const f of out.frames) doc.frames.push(f);
    if (out.events.length) { for (const ev of out.events) doc.events.push(ev); indexFight(doc); showFight(doc); }
    else R.dmgMax = Math.max(R.dmgMax, doc.frames.at(-1)[10], doc.frames.at(-1)[11]);
    if (window.ArenaRenderer?.grow) window.ArenaRenderer.grow(doc);
    const total = (doc.frames.length - 1) * doc.dt;
    if (total - R.t > 0.15) R.t = total - 0.04;               // stay within a frame or two of the sim: what you see is what you are steering
    if (!R.playing) setPlaying(true);
    if (out.done) stopLive();
  } catch (e) { console.error(e); stopLive(); }
  finally { LIVE.busy = false; }
}
$("#fight-live").addEventListener("click", ev => { ev.currentTarget.blur(); LIVE.on ? stopLive() : startLive(); });
const DRIVE = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
document.addEventListener("keydown", ev => {
  if (!LIVE.on) return;
  if (ev.code === "Escape") return stopLive();
  if (DRIVE.has(ev.code)) { LIVE.keys.add(ev.code); ev.preventDefault(); ev.stopImmediatePropagation(); }
  else if (ev.code === "Space") { ev.preventDefault(); ev.stopImmediatePropagation(); }
}, true);
document.addEventListener("keyup", ev => { LIVE.keys.delete(ev.code); }, true);
addEventListener("blur", () => LIVE.keys.clear());

function setPlaying(on) {
  const was = R.playing;
  R.playing = on; R.last = performance.now();
  if (was !== on || !$("#play").firstChild) {
    const b = $("#play"); b.innerHTML = icon(on ? "pause" : "play", 18); b.setAttribute("aria-label", on ? "Pause" : "Play");
    b.setAttribute("aria-pressed", on);
  }
  cancelAnimationFrame(R.raf); R.raf = 0;
  if (on) R.raf = requestAnimationFrame(loop);
}
function loop(now) {
  R.raf = 0;
  if (!R.playing || !R.doc) return;
  const total = (R.doc.frames.length - 1) * R.doc.dt;
  R.t += Math.min(0.1, Math.max(0, (now - R.last) / 1000)) * R.speed; R.last = now;   // a stalled frame does not skip the fight ahead
  if (R.t >= total) { R.t = total; if (!LIVE.on) setPlaying(false); }
  draw();
  if (R.playing) R.raf = requestAnimationFrame(loop);
}
function palette() {
  for (const k of ["--arena", "--grid", "--axis", "--ink", "--ink-2", "--muted", "--s1", "--s2", "--good", "--warning", "--critical", "--hazard", "--surface"]) C[k] = css(k);
}
const lerp = (a, b, u) => a + (b - a) * u;
const lerpAngle = (a, b, u) => a + (((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) * u;

// the arena is square and as large as the stage allows while the deck stays on screen
function sizeArena() {
  const stage = $(".stage"); if (!stage || $("#replays").hidden) return;
  const top = $(".top").offsetHeight, chrome = $(".deck").offsetHeight;
  const avail = innerHeight - top - chrome - 44;
  stage.style.setProperty("--arena-max", (innerWidth < 1000 ? stage.clientWidth : Math.max(440, Math.min(stage.clientWidth, avail))) + "px");
}

function frameState(doc) {
  const fi = Math.min(doc.frames.length - 1, R.t / doc.dt), i0 = Math.floor(fi), i1 = Math.min(doc.frames.length - 1, i0 + 1), u = fi - i0;
  const a = doc.frames[i0], b = doc.frames[i1];
  return { fi, i0, S: { ox: lerp(a[0], b[0], u), oy: lerp(a[1], b[1], u), oa: lerpAngle(a[2], b[2], u), ex: lerp(a[3], b[3], u), ey: lerp(a[4], b[4], u), ea: lerpAngle(a[5], b[5], u),
    arm: lerp(a[6], b[6], u), blade: a[7], eblade: a[8], flags: a[9], dealt: lerp(a[10], b[10], u), taken: lerp(a[11], b[11], u) } };
}

function draw() {
  const doc = R.doc, cv = $("#arena");
  if (!doc || $("#replays").hidden) return;
  const { fi, i0, S } = frameState(doc);
  let painted = false;
  if (window.ArenaRenderer && !R.arenaFailed) {
    try { window.ArenaRenderer.draw(cv, doc, { t: R.t, follow: R.follow, playing: R.playing, speed: R.speed, theme: theme() }); painted = true; }
    catch (e) { console.error("ArenaRenderer failed, using the built-in painter", e); R.arenaFailed = true; }
  }
  if (!painted) legacyArena(cv, doc, S, i0);

  // score bug, side panel, clock and timeline
  const total = (doc.frames.length - 1) * doc.dt, past = R.swings.filter(s => s.t <= R.t), n = kind => past.filter(s => s.kind === kind).length;
  const fired = R.fired.filter(f => f <= fi).length, hits = R.hits.filter(f => f <= fi).length;
  $("#bug-clock").textContent = mmss(R.t);
  $("#bug-us-dmg").textContent = S.dealt.toFixed(1);
  $("#bug-en-dmg").textContent = S.taken.toFixed(1);
  $("#bug-us-bar").style.width = (100 * S.dealt / R.dmgMax) + "%";
  $("#bug-en-bar").style.width = (100 * S.taken / R.dmgMax) + "%";
  $("#bug-hits").textContent = hits;
  $$("#bug-pips i").forEach((p, i) => { p.className = i < fired ? (past[i] ? past[i].kind : "fired") : ""; });
  $("#bug-swings").textContent = `${fired} of ${doc.swings} swings`; $("#bug-pips").title = `${fired} of ${doc.swings} swings used`;
  const armState = S.arm <= doc.arm_parked ? "Parked behind" : S.arm >= doc.arm_down ? "Down: hammer landed" : "Swinging over";
  const liveHtml = `
    <div><div class="k">Swings fired</div><div class="v">${fired} <small>of ${doc.swings}</small></div></div>
    <div><div class="k">Solid legal strikes</div><div class="v">${n("solid")}</div></div>
    <div><div class="k">Weapon-side landings</div><div class="v">${n("side")}</div></div>
    <div><div class="k">Floor landings</div><div class="v">${n("floor")}</div></div>
    <div><div class="k">Hits taken</div><div class="v">${hits}</div></div>
    <div><div class="k">Damage dealt / taken</div><div class="v">${S.dealt.toFixed(1)} <small>/ ${S.taken.toFixed(1)}</small></div></div>
    <div class="wide"><div class="k">Arm: ${armState}${doc.arm_gate ? (S.flags & 4 ? " · gate open" : S.flags & 8 ? " · gate ready" : " · gate closed") : ""}</div><div class="meter"><div style="width:${Math.round(100 * S.arm)}%"></div></div></div>
    <div class="wide"><div class="k">Our blade speed ${pct(S.blade)}</div><div class="meter seg5"><div style="width:${Math.round(100 * S.blade)}%"></div></div></div>`;
  if (liveHtml !== R.liveHtml) { $("#live").innerHTML = liveHtml; R.liveHtml = liveHtml; }
  $("#clock").textContent = `${mmss(R.t)} / ${mmss(total)}`;
  const at = 100 * R.t / total + "%";
  $("#tl-head").style.left = at; $("#tl-fill").style.width = at;
  if (document.activeElement !== $("#seek") || R.playing) $("#seek").value = Math.round(1000 * R.t / total);
  const recent = past.length && R.t - past.at(-1).t < 3 ? past.length - 1 : -1;
  const logKey = `${R.file}|${past.length}|${recent}`;
  if (logKey !== R.logKey) {
    R.logKey = logKey;
    $$("#swing-log button").forEach((bt, i) => bt.className = `${R.swings[i].kind} ${i === recent ? "now" : R.swings[i].t <= R.t ? "past" : "future"}`);
    $$("#ticks i").forEach((tk, i) => tk.classList.toggle("future", R.swings[i].t > R.t));
  }
}

// the built-in painter, used only while arena.js is missing or has failed
function legacyArena(cv, doc, S, i0) {
  const size = cv.clientWidth, dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(size * dpr)) { cv.width = cv.height = Math.round(size * dpr); }
  const g = cv.getContext("2d");
  if (!C["--ink"]) palette();
  const A = doc.arena_ft, gap = Math.hypot(S.ox - S.ex, S.oy - S.ey);
  const want = R.follow ? { span: Math.min(A, Math.max(16, gap * 1.5 + 10)), x: (S.ox + S.ex) / 2, y: (S.oy + S.ey) / 2 } : { span: A, x: A / 2, y: A / 2 };
  const ease = R.playing && R.cam ? Math.min(1, 0.08 * R.speed) : 1;
  R.cam = R.cam ? { span: lerp(R.cam.span, want.span, ease), x: lerp(R.cam.x, want.x, ease), y: lerp(R.cam.y, want.y, ease) } : want;
  const span = R.cam.span, k = size * dpr / span;
  const vx = Math.min(A - span, Math.max(0, R.cam.x - span / 2)), vy = Math.min(A - span, Math.max(0, R.cam.y - span / 2));
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = C["--arena"]; g.fillRect(0, 0, cv.width, cv.height);
  g.translate(-vx * k, -vy * k);
  g.fillStyle = C["--hazard"];
  for (const [x, y] of doc.hazards) g.fillRect(x * k, y * k, k, k);
  g.strokeStyle = C["--grid"]; g.lineWidth = dpr; g.beginPath();
  for (let i = 4; i < A; i += 4) { g.moveTo(i * k, 0); g.lineTo(i * k, A * k); g.moveTo(0, i * k); g.lineTo(A * k, i * k); }
  g.stroke();

  const back = Math.max(0, i0 - Math.round(4 / doc.dt));
  for (const [cx, cy, col] of [[0, 1, C["--s1"]], [3, 4, C["--s2"]]]) {
    g.strokeStyle = col; g.lineWidth = 2 * dpr; g.lineCap = "round";
    for (let i = back + 1; i <= i0; i += 2) {
      g.globalAlpha = 0.5 * (i - back) / (i0 - back + 1);
      g.beginPath(); g.moveTo(doc.frames[i - 1][cx] * k, doc.frames[i - 1][cy] * k); g.lineTo(doc.frames[Math.min(i0, i + 1)][cx] * k, doc.frames[Math.min(i0, i + 1)][cy] * k); g.stroke();
    }
  }
  g.globalAlpha = 1;

  const body = (x, y, ang, hl, hw, col, stunned) => {
    g.save(); g.translate(x * k, y * k); g.rotate(ang);
    g.beginPath(); g.roundRect(-hl * k, -hw * k, 2 * hl * k, 2 * hw * k, 0.12 * k);
    g.globalAlpha = 0.22; g.fillStyle = col; g.fill(); g.globalAlpha = 1;
    g.strokeStyle = col; g.lineWidth = 2 * dpr; g.setLineDash(stunned ? [5 * dpr, 4 * dpr] : []); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.moveTo(hl * k * 0.75, 0); g.lineTo(hl * k * 0.2, -hw * k * 0.4); g.lineTo(hl * k * 0.2, hw * k * 0.4); g.closePath(); g.fillStyle = col; g.fill();
    return () => g.restore();
  };

  const e = doc.enemy;
  let done = body(S.ex, S.ey, S.ea, e.half_len, e.half_wid, C["--s2"], S.flags & 2);
  if (e.weapon !== "none") {
    g.fillStyle = C["--s2"]; g.strokeStyle = C["--s2"];
    if (e.rad > 0.8) {
      g.globalAlpha = 0.12 + 0.12 * S.eblade; g.beginPath(); g.arc(e.fore * k, 0, e.rad * k, 0, 7); g.fill();
      g.globalAlpha = 0.9; g.lineWidth = 1.5 * dpr; g.beginPath(); g.arc(e.fore * k, 0, e.rad * k, 0, 7); g.stroke();
      const spin = R.t * 9 * S.eblade;
      g.lineWidth = 0.22 * k; g.lineCap = "round"; g.beginPath();
      g.moveTo(e.fore * k - Math.cos(spin) * e.rad * k * 0.94, -Math.sin(spin) * e.rad * k * 0.94); g.lineTo(e.fore * k + Math.cos(spin) * e.rad * k * 0.94, Math.sin(spin) * e.rad * k * 0.94); g.stroke();
    } else {
      g.globalAlpha = 0.45 + 0.5 * S.eblade; g.beginPath();
      g.roundRect((e.fore - e.rad) * k, -(e.wide + e.rad) * k, 2 * e.rad * k, 2 * (e.wide + e.rad) * k, e.rad * k); g.fill();
    }
    g.globalAlpha = 1;
  }
  done();

  const o = doc.our, along = -o.aft + (o.aft + o.fore) * S.arm, flying = S.arm > doc.arm_parked && S.arm < doc.arm_down;
  done = body(S.ox, S.oy, S.oa, o.half_len, o.half_wid, C["--s1"], S.flags & 1);
  g.strokeStyle = C["--ink"]; g.fillStyle = C["--ink"]; g.lineWidth = 2.5 * dpr; g.lineCap = "round";
  g.beginPath(); g.moveTo(0, 0); g.lineTo(along * k, 0); g.stroke();
  g.beginPath(); g.arc(along * k, 0, Math.max(o.rad * k, 4 * dpr) * (flying ? 0.8 : 1), 0, 7);
  if (flying) { g.lineWidth = 2 * dpr; g.stroke(); } else { g.globalAlpha = S.arm >= doc.arm_down ? 0.95 : 0.55; g.fill(); g.globalAlpha = 1; }
  done();

  for (const s of R.swings) {
    const age = R.t - s.t;
    if (age < 0 || age > 0.7) continue;
    const f = doc.frames[s.f], hx = f[0] + Math.cos(f[2]) * o.fore, hy = f[1] + Math.sin(f[2]) * o.fore;
    g.strokeStyle = { solid: C["--good"], weak: C["--good"], side: C["--warning"], floor: C["--muted"] }[s.kind];
    g.globalAlpha = 1 - age / 0.7; g.lineWidth = (s.kind === "solid" ? 4 : 2.5) * dpr;
    g.beginPath(); g.arc(hx * k, hy * k, (0.6 + 2.6 * age / 0.7) * k, 0, 7); g.stroke(); g.globalAlpha = 1;
  }
  const hitAge = (() => { let b = 9; for (const f of R.hits) { const d = R.t - f * doc.dt; if (d >= 0 && d < b) b = d; if (f * doc.dt > R.t) break; } return b; })();
  if (hitAge < 0.25) { g.strokeStyle = C["--critical"]; g.globalAlpha = 1 - hitAge / 0.25; g.lineWidth = 3 * dpr; g.beginPath(); g.arc(S.ox * k, S.oy * k, 1.6 * k, 0, 7); g.stroke(); g.globalAlpha = 1; }

  g.font = `600 ${12 * dpr}px Bahnschrift, "Segoe UI", system-ui, sans-serif`; g.textAlign = "center"; g.fillStyle = C["--ink"];
  g.fillText("Us" + (S.flags & 1 ? " (thrown)" : ""), S.ox * k, (S.oy + Math.max(o.half_len, o.half_wid) + 0.9) * k + 10 * dpr);
  g.fillText(e.tactic + (S.flags & 2 ? " (thrown)" : ""), S.ex * k, (S.ey - Math.max(e.half_len, e.half_wid, e.rad > 0.8 ? e.rad : 0) - 0.7) * k);
}

// ───────────────────────────── wiring ─────────────────────────────
function showTab(name) {
  state.tab = name;
  $$(".tab-panel").forEach(p => p.hidden = p.id !== name);
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  if (state.stale) { state.stale = false; renderOverview(); }
  if (name === "training") renderTraining();
  if (name === "yardstick") renderYardstick();
  if (name === "replays") { sizeArena(); layoutTicks(); draw(); }
  if (name !== "replays") { stopLive(); setPlaying(false); }
  history.replaceState(null, "", location.search + "#" + name);
}
$$(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
function seg(id, apply) {
  $$(`#${id} button`).forEach(b => b.addEventListener("click", () => {
    $$(`#${id} button`).forEach(x => x.setAttribute("aria-pressed", x === b)); apply(+b.dataset.v);
  }));
}
seg("smooth", v => { state.smooth = v; renderTraining(); });
seg("speed", v => { R.speed = v; draw(); });
seg("view", v => { R.follow = !!v; R.cam = null; draw(); });
$("#play").addEventListener("click", () => {
  if (!R.doc || LIVE.on) return;
  if (!R.playing && R.t >= (R.doc.frames.length - 1) * R.doc.dt - 0.01) R.t = 0;
  setPlaying(!R.playing);
  if (!R.playing) draw();
});
$("#seek").addEventListener("input", ev => { if (R.doc) { R.t = ev.target.value / 1000 * (R.doc.frames.length - 1) * R.doc.dt; R.last = performance.now(); R.cam = null; draw(); } });
document.addEventListener("keydown", ev => {
  if (state.tab !== "replays" || !R.doc || (ev.target.tagName === "INPUT" && ev.target.id !== "seek")) return;
  const total = (R.doc.frames.length - 1) * R.doc.dt;
  if (ev.code === "Space" && ev.target.tagName !== "BUTTON") { ev.preventDefault(); if (!ev.repeat) $("#play").click(); }
  if (ev.code === "ArrowRight") { ev.preventDefault(); R.t = Math.min(total, R.t + 1); R.cam = null; draw(); }
  if (ev.code === "ArrowLeft") { ev.preventDefault(); R.t = Math.max(0, R.t - 1); R.cam = null; draw(); }
});
function themeChanged() { palette(); draw(); }
$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  root.dataset.theme = theme() === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
  themeChanged();
});
matchMedia("(prefers-color-scheme: light)").addEventListener("change", themeChanged);
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderTraining(); renderYardstick(); sizeArena(); layoutTicks(); draw(); }, 150); });
showTab(["overview", "training", "yardstick", "replays"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview");
// the tab hash matches a section id, so the browser would scroll to it; tabs always open at the top
if ("scrollRestoration" in history) history.scrollRestoration = "manual";
addEventListener("load", () => scrollTo(0, 0));
addEventListener("hashchange", () => { const t = location.hash.slice(1); if (t !== state.tab && $("#" + t + ".tab-panel")) showTab(t); });
refresh();
setInterval(refresh, 30000);
