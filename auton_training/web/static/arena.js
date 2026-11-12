/* arena.js: the fight-replay renderer for the dodge_master viewer.
 *
 * Defines one global, window.ArenaRenderer = { reset(), draw(canvas, doc, view) }.
 *   doc  = a replay written by web/export_replays.py
 *   view = { t, follow, playing, speed, theme: "dark" | "light" }
 *
 * Everything drawn is a pure function of (doc, t) except the camera easing,
 * so seeking anywhere paints correctly. Static art (floor, chassis, shadows)
 * is painted once into offscreen canvases and blitted each frame.
 * Units: world coordinates are feet, y down, angles in radians (0 = +x).
 */
(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const RIM = 1.25;                 // ft of wall shown around the 48 ft floor
  const SPIN_O = 46, SPIN_E = 27;   // rad/s shown at full blade speed (slowed so the eye can follow it)
  const SHADOW = { x: 0.17, y: 0.26 };   // light from the upper left
  const FONT = 'Bahnschrift, "DIN Alternate", "DIN Condensed", "Barlow Condensed", "Roboto Condensed", "Arial Narrow", sans-serif';
  const REDUCED = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, u) => a + (b - a) * u;
  const wrap = a => ((a + Math.PI) % TAU + TAU) % TAU - Math.PI;
  const lerpA = (a, b, u) => a + wrap(b - a) * u;
  const ease3 = u => 1 - Math.pow(1 - u, 3);
  function rng(seed) {                // mulberry32: the same seed always throws the same sparks
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const mk = (w, h) => { const c = document.createElement("canvas"); c.width = Math.max(1, Math.ceil(w)); c.height = Math.max(1, Math.ceil(h)); return c; };
  const rr = (g, x, y, w, h, r) => { g.beginPath(); g.roundRect(x, y, Math.max(0, w), Math.max(0, h), Math.max(0, Math.min(r, w / 2, h / 2))); };
  const poly = (g, pts) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.closePath(); };

  // ───────────────────────────── palettes ─────────────────────────────
  // Team colours follow the page: blue is us (--s1), orange is the opponent (--s2).
  const THEMES = {
    dark: {
      void: "#06080b", rimA: "#3d444d", rimB: "#16191e", rimEdge: "rgba(160,175,190,.35)", rimBolt: "#7c8794",
      floor: "#24282e", plateVar: 0.035, grain: 22, scuff: "rgba(0,0,0,.22)", scuffHi: "rgba(255,255,255,.035)",
      stain: "rgba(0,0,0,.16)", light: "rgba(190,210,235,.055)", seam: "rgba(0,0,0,.6)", seamHi: "rgba(255,255,255,.055)",
      bolt: "#4b525b", paint: "rgba(140,170,210,.10)", tick: "rgba(225,232,240,.6)", wallAO: "rgba(0,0,0,.5)",
      grateA: "#0c0d0f", grateB: "#2a2d31", glow: "rgba(255,110,40,.13)", stripeA: "#f3c12b", stripeB: "#151516",
      track: "rgba(0,0,0,.5)", scorch: "rgba(0,0,0,.62)", cut: "rgba(235,240,245,.55)", dust: "196,186,168",
      us: "#4a97f0", usDeep: "#1a4f98", usHi: "#9cc8ff", them: "#ef6a30", themDeep: "#8a2b0a", themHi: "#ffb08a",
      amber: "#ffc21f", gold: "#ffd75e", hot: "#fff6dc", red: "#ff3344", miss: "#9aa3ad",
      tagBg: "rgba(9,12,16,.84)", tagInk: "#f3f6f9", tagLine: "rgba(255,255,255,.14)",
      shadow: 0.62, vignette: "rgba(0,0,0,.42)", additive: true,
    },
    light: {
      void: "#b9bfc6", rimA: "#7b858f", rimB: "#4c545d", rimEdge: "rgba(255,255,255,.35)", rimBolt: "#c3cad1",
      floor: "#c9ced3", plateVar: 0.03, grain: 20, scuff: "rgba(30,35,40,.07)", scuffHi: "rgba(255,255,255,.16)",
      stain: "rgba(40,45,50,.07)", light: "rgba(255,255,255,.18)", seam: "rgba(40,48,56,.42)", seamHi: "rgba(255,255,255,.55)",
      bolt: "#9aa2ab", paint: "rgba(30,80,150,.12)", tick: "rgba(255,255,255,.85)", wallAO: "rgba(40,48,56,.22)",
      grateA: "#2c3036", grateB: "#5b626a", glow: "rgba(255,110,40,.10)", stripeA: "#f0b90f", stripeB: "#1d1e20",
      track: "rgba(25,28,32,.22)", scorch: "rgba(20,20,22,.42)", cut: "rgba(255,255,255,.9)", dust: "120,108,92",
      us: "#1f6fd1", usDeep: "#123f80", usHi: "#8fbcf5", them: "#e0581d", themDeep: "#7a2606", themHi: "#ffb08a",
      amber: "#d98a00", gold: "#e8a400", hot: "#ff8a00", red: "#e0182c", miss: "#5d6670",
      tagBg: "rgba(255,255,255,.93)", tagInk: "#11161c", tagLine: "rgba(0,0,0,.14)",
      shadow: 0.36, vignette: "rgba(40,50,60,.16)", additive: false,
    },
  };

  // ───────────────────────────── state ─────────────────────────────
  const S = { doc: null, P: null, cam: null, lastNow: 0, lastT: -1, cache: new Map() };
  function reset() { S.doc = null; S.P = null; S.cam = null; S.lastT = -1; S.cache.clear(); }
  function cached(key, build) { let v = S.cache.get(key); if (!v) { v = build(); S.cache.set(key, v); } return v; }

  // ───────────────────────────── data prep (once per fight) ─────────────────────────────
  function prep(doc) {
    const F = doc.frames, n = F.length, dt = doc.dt || 0.03, A = doc.arena_ft || 48, o = doc.our, e = doc.enemy;
    const P = { n, dt, A, total: Math.max(0, (n - 1) * dt) };
    P.spinO = new Float64Array(n); P.spinE = new Float64Array(n);
    P.odo = [0, 1, 2, 3].map(() => new Float64Array(n));          // tread travel: our L, our R, their L, their R
    const wo = o.half_wid * 0.85, we = e.half_wid * 0.85;
    for (let i = 1; i < n; i++) {
      const a = F[i - 1], b = F[i];
      P.spinO[i] = P.spinO[i - 1] + (a[7] + b[7]) * 0.5 * dt * SPIN_O;
      P.spinE[i] = P.spinE[i - 1] + (a[8] + b[8]) * 0.5 * dt * SPIN_E;
      for (const [c, w, j] of [[0, wo, 0], [3, we, 2]]) {
        const ds = (b[c] - a[c]) * Math.cos(a[c + 2]) + (b[c + 1] - a[c + 1]) * Math.sin(a[c + 2]);
        const da = wrap(b[c + 2] - a[c + 2]);
        P.odo[j][i] = P.odo[j][i - 1] + ds - da * w;
        P.odo[j + 1][i] = P.odo[j + 1][i - 1] + ds + da * w;
      }
    }
    const byF = new Map(); P.hits = []; P.fires = [];
    for (const [f, k] of doc.events || []) {
      if (k === "hit") P.hits.push(f);
      else if (k === "swing") P.fires.push(f);
      else { if (!byF.has(f)) byF.set(f, new Set()); byF.get(f).add(k); }
    }
    P.lands = [...byF].sort((x, y) => x[0] - y[0]).map(([f, set]) => {
      const r = F[Math.min(f, n - 1)], along = -o.aft + (o.aft + o.fore) * r[6];
      const kind = set.has("solid") ? "solid" : set.has("legal") ? "weak" : set.has("side") ? "side" : "floor";
      return { f, t: f * dt, kind, x: r[0] + Math.cos(r[2]) * along, y: r[1] + Math.sin(r[2]) * along, a: r[2], seed: Math.imul(f + 1, 2654435761) >>> 0 };
    });
    P.hitFx = P.hits.map(f => {
      const r = F[Math.min(f, n - 1)];
      const wx = r[3] + Math.cos(r[5]) * e.fore, wy = r[4] + Math.sin(r[5]) * e.fore;
      const dx = wx - r[0], dy = wy - r[1], d = Math.hypot(dx, dy) || 1, reach = Math.min(d, Math.max(o.half_len, o.half_wid) * 0.95);
      return { f, t: f * dt, x: r[0] + dx / d * reach, y: r[1] + dy / d * reach, a: Math.atan2(dy, dx), seed: Math.imul(f + 7, 40503) >>> 0 };
    });
    // hazard cells and the boundary they share with the open floor
    const grid = new Uint8Array(A * A);
    for (const [x, y] of doc.hazards || []) if (x >= 0 && y >= 0 && x < A && y < A) grid[y * A + x] = 1;
    const hz = (x, y) => (x < 0 || y < 0 || x >= A || y >= A ? 1 : grid[y * A + x]);
    const edges = [];
    for (let y = 0; y < A; y++) for (let x = 0; x < A; x++) {
      if (!grid[y * A + x]) continue;
      if (!hz(x, y - 1)) edges.push([x, y, x + 1, y]);
      if (!hz(x, y + 1)) edges.push([x, y + 1, x + 1, y + 1]);
      if (!hz(x - 1, y)) edges.push([x, y, x, y + 1]);
      if (!hz(x + 1, y)) edges.push([x + 1, y, x + 1, y + 1]);
    }
    P.grid = grid; P.edges = edges;
    // saw-v9 replays carry the walls as smooth outlines: [[x, y], ...] along the x = 0, x = A and y = 0 walls
    P.walls = doc.walls && doc.walls.x0 ? [[doc.walls.x0, [0, 0], [0, A]], [doc.walls.x1, [A, 0], [A, A]], [doc.walls.y0, [0, 0], [A, 0]]] : null;
    return P;
  }

  function sample(doc, P, t) {
    const F = doc.frames, fi = clamp(t / P.dt, 0, P.n - 1), i0 = Math.floor(fi), i1 = Math.min(P.n - 1, i0 + 1), u = fi - i0;
    const a = F[i0], b = F[i1], L = arr => lerp(arr[i0], arr[i1], u);
    return {
      fi, i0, u,
      ox: lerp(a[0], b[0], u), oy: lerp(a[1], b[1], u), oa: lerpA(a[2], b[2], u),
      ex: lerp(a[3], b[3], u), ey: lerp(a[4], b[4], u), ea: lerpA(a[5], b[5], u),
      arm: lerp(a[6], b[6], u), blade: lerp(a[7], b[7], u), eblade: lerp(a[8], b[8], u),
      flags: a[9] | 0, spinO: L(P.spinO), spinE: L(P.spinE), odo: P.odo.map(L),
    };
  }

  // ───────────────────────────── floor (painted once per theme and resolution) ─────────────────────────────
  function noiseTile(th) {                                   // brushed steel grain, tileable
    const N = 256, c = mk(N, N), g = c.getContext("2d"), img = g.createImageData(N, N), d = img.data, r = rng(99);
    const rows = new Float32Array(N); let acc = 0;
    for (let y = 0; y < N; y++) { acc = acc * 0.55 + (r() - 0.5); rows[y] = acc; }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const v = rows[y] * (0.65 + 0.35 * Math.sin((x / N) * TAU * 3 + y * 1.7)) + (r() - 0.5) * 0.45;
      const i = (y * N + x) * 4, lum = v > 0 ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = lum; d[i + 3] = Math.min(255, Math.abs(v) * th.grain);
    }
    g.putImageData(img, 0, 0);
    return c;
  }
  function stripeTile(th) {                                  // yellow and black hazard tape
    const N = 64, c = mk(N, N), g = c.getContext("2d");
    g.fillStyle = th.stripeB; g.fillRect(0, 0, N, N);
    g.fillStyle = th.stripeA;
    for (let o = -2 * N; o <= N; o += N) poly(g, [[o, 0], [o + N / 2, 0], [o + N / 2 + N, N], [o + N, N]]), g.fill();
    return c;
  }
  function buildFloor(P, th, ppf) {
    const A = P.A, c = mk((A + 2 * RIM) * ppf, (A + 2 * RIM) * ppf), g = c.getContext("2d"), r = rng(1234);
    g.scale(ppf, ppf); g.translate(RIM, RIM);

    // the wall kerb around the floor
    let gr = g.createLinearGradient(-RIM, -RIM, A + RIM, A + RIM);
    gr.addColorStop(0, th.rimA); gr.addColorStop(1, th.rimB);
    g.fillStyle = gr; g.fillRect(-RIM, -RIM, A + 2 * RIM, A + 2 * RIM);
    g.strokeStyle = th.rimEdge; g.lineWidth = 0.04;
    g.strokeRect(-RIM + 0.12, -RIM + 0.12, A + 2 * RIM - 0.24, A + 2 * RIM - 0.24);
    g.strokeRect(-0.42, -0.42, A + 0.84, A + 0.84);
    g.fillStyle = th.rimBolt;
    for (let s = 2; s < A; s += 4) for (const [bx, by] of [[s, -RIM + 0.36], [s, A + RIM - 0.36], [-RIM + 0.36, s], [A + RIM - 0.36, s]]) {
      g.beginPath(); g.arc(bx, by, 0.07, 0, TAU); g.fill();
    }
    // ruler along the top and left kerbs: the arena is in feet
    g.strokeStyle = th.tick; g.lineCap = "butt";
    for (let s = 0; s <= A; s++) {
      const len = s % 8 === 0 ? 0.42 : s % 4 === 0 ? 0.28 : 0.14;
      g.lineWidth = s % 4 === 0 ? 0.05 : 0.03;
      g.beginPath(); g.moveTo(s, -0.02); g.lineTo(s, -0.02 - len); g.moveTo(-0.02, s); g.lineTo(-0.02 - len, s);
      g.moveTo(s, A + 0.02); g.lineTo(s, A + 0.02 + len); g.moveTo(A + 0.02, s); g.lineTo(A + 0.02 + len, s); g.stroke();
    }
    g.save(); g.scale(1 / ppf, 1 / ppf);
    g.fillStyle = th.tick; g.font = `600 ${0.36 * ppf}px ${FONT}`; g.textBaseline = "middle";
    for (let s = 8; s < A; s += 8) {
      g.textAlign = "center"; g.fillText(s + "", s * ppf, -0.8 * ppf);
      g.save(); g.translate(-0.8 * ppf, s * ppf); g.rotate(-Math.PI / 2); g.fillText(s + "", 0, 0); g.restore();
    }
    g.textAlign = "left"; g.fillText("0", 0.08 * ppf, -0.8 * ppf);
    g.textAlign = "right"; g.fillText(A + " FT", A * ppf, -0.8 * ppf);
    g.restore();

    // steel plates, 8 ft square, grain turned on alternate plates
    g.fillStyle = th.floor; g.fillRect(0, 0, A, A);
    const pat = g.createPattern(noiseTile(th), "repeat");
    pat.setTransform(new DOMMatrix().scale(3.2 / 256));
    const PL = 8;
    for (let py = 0; py < A; py += PL) for (let px = 0; px < A; px += PL) {
      g.save(); g.beginPath(); g.rect(px, py, PL, PL); g.clip();
      const v = (r() - 0.5) * 2 * th.plateVar;
      g.fillStyle = v > 0 ? `rgba(255,255,255,${v})` : `rgba(0,0,0,${-v * 1.4})`; g.fillRect(px, py, PL, PL);
      g.translate(px + PL / 2, py + PL / 2); if (((px + py) / PL) % 2) g.rotate(Math.PI / 2);
      g.fillStyle = pat; g.fillRect(-PL, -PL, 2 * PL, 2 * PL);
      g.restore();
    }
    // wear: stains and the scuffs of a hundred earlier fights
    for (let i = 0; i < 22; i++) {
      const x = r() * A, y = r() * A, rad = 1 + r() * 3.5;
      const sg = g.createRadialGradient(x, y, 0, x, y, rad); sg.addColorStop(0, th.stain); sg.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = sg; g.fillRect(x - rad, y - rad, 2 * rad, 2 * rad);
    }
    g.lineCap = "round";
    for (let i = 0; i < 260; i++) {
      const x = r() * A, y = r() * A, rad = 0.8 + r() * 6, a0 = r() * TAU, sw = 0.15 + r() * 0.7;
      g.strokeStyle = r() < 0.75 ? th.scuff : th.scuffHi; g.lineWidth = 0.015 + r() * 0.06;
      g.beginPath(); g.arc(x, y, rad, a0, a0 + sw); g.stroke();
    }
    // overhead lights
    for (const [lx, ly, lr] of [[12, 12, 15], [36, 12, 15], [12, 36, 15], [36, 36, 15], [24, 24, 19]]) {
      const lg = g.createRadialGradient(lx, ly, 0, lx, ly, lr); lg.addColorStop(0, th.light); lg.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = lg; g.fillRect(lx - lr, ly - lr, 2 * lr, 2 * lr);
    }
    // centre marking
    g.strokeStyle = th.paint; g.lineWidth = 0.14;
    g.beginPath(); g.arc(A / 2, A / 2, 5, 0, TAU); g.stroke();
    g.lineWidth = 0.08; g.beginPath(); g.arc(A / 2, A / 2, 1.1, 0, TAU); g.stroke();
    for (let k = 0; k < 12; k++) {
      const a = k * TAU / 12, r0 = k % 3 === 0 ? 4.3 : 4.6;
      g.beginPath(); g.moveTo(A / 2 + Math.cos(a) * r0, A / 2 + Math.sin(a) * r0); g.lineTo(A / 2 + Math.cos(a) * 5, A / 2 + Math.sin(a) * 5); g.stroke();
    }
    // plate seams with a lit lip, and countersunk bolts
    for (let s = PL; s < A; s += PL) {
      g.strokeStyle = th.seam; g.lineWidth = 0.07;
      g.beginPath(); g.moveTo(s, 0); g.lineTo(s, A); g.moveTo(0, s); g.lineTo(A, s); g.stroke();
      g.strokeStyle = th.seamHi; g.lineWidth = 0.03;
      g.beginPath(); g.moveTo(s + 0.06, 0); g.lineTo(s + 0.06, A); g.moveTo(0, s + 0.06); g.lineTo(A, s + 0.06); g.stroke();
    }
    for (let y = 0; y <= A; y += 2) for (let x = 0; x <= A; x += 2) {
      if (x % PL && y % PL) continue;
      for (const [bx, by] of [[x + 0.22, y + 0.22]]) {
        if (bx > A || by > A) continue;
        g.fillStyle = th.seam; g.beginPath(); g.arc(bx, by, 0.065, 0, TAU); g.fill();
        g.fillStyle = th.bolt; g.beginPath(); g.arc(bx - 0.01, by - 0.01, 0.045, 0, TAU); g.fill();
      }
    }
    // hazard zones: recessed grating, warm underglow, tape along the edge
    if (P.walls || P.edges.length || P.grid.some(Boolean)) {
      const zone = new Path2D(), rim = new Path2D();
      if (P.walls) {
        for (const [pts, a, b] of P.walls) {
          zone.moveTo(a[0], a[1]); for (const [x, y] of pts) zone.lineTo(x, y); zone.lineTo(b[0], b[1]); zone.closePath();
          rim.moveTo(pts[0][0], pts[0][1]); for (const [x, y] of pts) rim.lineTo(x, y);
        }
      } else {
        for (let y = 0; y < A; y++) for (let x = 0; x < A; x++) if (P.grid[y * A + x]) zone.rect(x, y, 1, 1);
        for (const [x1, y1, x2, y2] of P.edges) { rim.moveTo(x1, y1); rim.lineTo(x2, y2); }
      }
      g.save(); g.clip(zone);
      g.fillStyle = th.grateA; g.fillRect(0, 0, A, A);
      g.strokeStyle = th.grateB; g.lineWidth = 0.07;
      g.beginPath(); for (let s = 0.125; s < A; s += 0.25) { g.moveTo(s, 0); g.lineTo(s, A); } g.stroke();
      g.lineWidth = 0.12; g.strokeStyle = th.grateA;
      g.beginPath(); for (let s = 0; s < A; s += 1) { g.moveTo(0, s); g.lineTo(A, s); } g.stroke();
      g.fillStyle = th.glow; g.fillRect(0, 0, A, A);
      const sp = g.createPattern(stripeTile(th), "repeat"); sp.setTransform(new DOMMatrix().scale(0.7 / 64));
      g.strokeStyle = sp; g.lineWidth = 0.78; g.lineCap = "square"; g.lineJoin = "round";
      g.stroke(rim);
      g.restore();
      g.strokeStyle = P.walls ? "rgba(0,0,0,.8)" : "rgba(0,0,0,.55)"; g.lineWidth = P.walls ? 0.12 : 0.06; g.lineCap = "square";
      g.stroke(rim);
    }
    // walls throw a soft shadow onto the floor
    for (const [x0, y0, x1, y1] of [[0, 0, 0, 1.8], [0, A, 0, A - 1.8], [0, 0, 1.8, 0], [A, 0, A - 1.8, 0]]) {
      const ag = g.createLinearGradient(x0, y0, x1, y1); ag.addColorStop(0, th.wallAO); ag.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = ag; g.fillRect(0, 0, A, A);
    }
    g.strokeStyle = "rgba(0,0,0,.7)"; g.lineWidth = 0.1; g.strokeRect(-0.05, -0.05, A + 0.1, A + 0.1);
    return { c, ppf };
  }

  // ───────────────────────────── robot art ─────────────────────────────
  function sprite(x0, y0, w, h, ppf, paint) {
    const c = mk(w * ppf, h * ppf), g = c.getContext("2d");
    g.scale(ppf, ppf); g.translate(-x0, -y0); paint(g, ppf);
    return { c, x0, y0, w: c.width / ppf, h: c.height / ppf };
  }
  const blit = (g, s) => g.drawImage(s.c, s.x0, s.y0, s.w, s.h);

  const wheelsOf = (L, W, xs) => {
    const l = clamp(L * 0.48, 0.45, 0.85), w = clamp(W * 0.2, 0.22, 0.32), out = [];
    for (const x of xs) for (const side of [-1, 1]) out.push({ x, y: side * (W - w / 2 - 0.02), l, w, side });
    return out;
  };
  const usWheels = o => wheelsOf(o.half_len, o.half_wid, [-0.56 * o.half_len, 0.52 * o.half_len]);
  const themWheels = e => wheelsOf(e.half_len, e.half_wid, [-0.56 * e.half_len, 0.38 * e.half_len]);

  function steel(g, x0, y0, x1, y1, stops) {
    const gr = g.createLinearGradient(x0, y0, x1, y1);
    stops.forEach((c, i) => gr.addColorStop(i / (stops.length - 1), c));
    return gr;
  }
  function bolt(g, x, y, r = 0.042) {
    g.fillStyle = "rgba(0,0,0,.55)"; g.beginPath(); g.arc(x + 0.01, y + 0.012, r * 1.15, 0, TAU); g.fill();
    g.fillStyle = "#d9dee4"; g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill();
    g.fillStyle = "#4a5058"; g.beginPath(); g.arc(x, y, r * 0.42, 0, TAU); g.fill();
  }
  function boltRow(g, x0, y0, x1, y1, step) {
    const n = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0) / step));
    for (let i = 0; i <= n; i++) bolt(g, lerp(x0, x1, i / n), lerp(y0, y1, i / n));
  }
  function brushed(g, x, y, w, h, seed, along = "x") {
    const r = rng(seed);
    g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); g.lineWidth = 0.012;
    for (let i = 0; i < 170; i++) {
      g.strokeStyle = r() < 0.5 ? "rgba(255,255,255,.10)" : "rgba(0,0,0,.08)";
      const p = r(), q = r(), len = 0.3 + r() * 1.4;
      g.beginPath();
      if (along === "x") { g.moveTo(x + q * w - len / 2, y + p * h); g.lineTo(x + q * w + len / 2, y + p * h); }
      else { g.moveTo(x + p * w, y + q * h - len / 2); g.lineTo(x + p * w, y + q * h + len / 2); }
      g.stroke();
    }
    g.restore();
  }
  function cutWells(g, wheels) {
    g.save(); g.globalCompositeOperation = "destination-out";
    for (const w of wheels) { rr(g, w.x - w.l / 2 - 0.05, w.y - w.w / 2 - 0.05, w.l + 0.1, w.w + 0.1, 0.07); g.fill(); }
    g.restore();
    g.strokeStyle = "rgba(0,0,0,.65)"; g.lineWidth = 0.05;
    for (const w of wheels) { rr(g, w.x - w.l / 2 - 0.05, w.y - w.w / 2 - 0.05, w.l + 0.1, w.w + 0.1, 0.07); g.stroke(); }
  }

  // Ours: Nemesis. Matte black armour, red flame livery, gold sunburst on the lids, the arm slot running aft.
  const FORK_FT = 1.3;               // how far the decorative forks stand out of our front bar
  const NEM = { red: "#e3202b", redDeep: "#8f0f17", redHi: "#ff5a5f", gold: "#d9b25a", goldHi: "#f6e2a6", goldDeep: "#8a6a25" };
  function flames(g, x0, x1, y, h, dir, seed) {        // a row of licking flames along x, tips pointing dir (+1/-1) in y
    const r = rng(seed), n = Math.max(3, Math.round((x1 - x0) / 0.3)), step = (x1 - x0) / n;
    g.beginPath(); g.moveTo(x0, y);
    for (let i = 0; i < n; i++) {
      const xa = x0 + i * step, tip = h * (0.55 + 0.45 * r()), lean = step * (0.15 + 0.5 * r());
      g.quadraticCurveTo(xa + step * 0.2, y + dir * tip * 0.35, xa + step * 0.35 + lean, y + dir * tip);
      g.quadraticCurveTo(xa + step * 0.62, y + dir * tip * 0.3, xa + step, y);
    }
    g.closePath();
    g.fillStyle = steel(g, 0, y, 0, y + dir * h, [NEM.redDeep, NEM.red, NEM.redHi]); g.fill();
  }
  function sunburst(g, cx, cy, r0, r1, a0, a1, n) {    // thin gold rays fanning out from a point
    g.fillStyle = steel(g, cx, cy, cx + r1, cy, [NEM.goldHi, NEM.gold, NEM.goldDeep]);
    for (let i = 0; i < n; i++) {
      const a = lerp(a0, a1, i / (n - 1)), w = 0.035, len = r1 * (i % 2 ? 0.72 : 1);
      g.beginPath();
      g.moveTo(cx + Math.cos(a - w) * r0, cy + Math.sin(a - w) * r0);
      g.lineTo(cx + Math.cos(a) * len, cy + Math.sin(a) * len);
      g.lineTo(cx + Math.cos(a + w) * r0, cy + Math.sin(a + w) * r0);
      g.closePath(); g.fill();
    }
  }
  function paintUs(g, o, th) {
    const L = o.half_len, W = o.half_wid, wheels = usWheels(o);
    // four forks out of the front bar, gold at the tips. Decoration only: nothing in the sim touches them.
    for (const fy of [-0.86 * W, -0.3 * W, 0.3 * W, 0.86 * W]) {
      const x0 = L - 0.25, x1 = L + FORK_FT - 0.04, hw = 0.075, tip = x1 - 0.42;
      g.fillStyle = "rgba(0,0,0,.35)"; rr(g, x0 + 0.03, fy - hw + 0.04, x1 - x0 - 0.1, 2 * hw, 0.03); g.fill();
      g.fillStyle = steel(g, 0, fy - hw, 0, fy + hw, ["#3a3d42", "#141518", "#08090a"]); rr(g, x0, fy - hw, tip - x0 + 0.02, 2 * hw, 0.03); g.fill();
      g.fillStyle = steel(g, tip - 0.25, 0, x1, 0, ["#141518", NEM.goldDeep, NEM.gold, NEM.goldHi]);
      poly(g, [[tip - 0.25, fy - hw], [x1 - 0.2, fy - hw * 1.5], [x1, fy], [x1 - 0.2, fy + hw * 1.5], [tip - 0.25, fy + hw]]); g.fill();
      g.fillStyle = "#0c0d0f"; rr(g, x0 + 0.1, fy - hw * 1.6, 0.26, hw * 3.2, 0.03); g.fill();
    }
    g.fillStyle = steel(g, 0, -W, 0, W, ["#2a2c30", "#17181b", "#0c0d0f"]);
    rr(g, -L, -W, 2 * L, 2 * W, 0.2); g.fill();
    g.strokeStyle = "rgba(255,255,255,.14)"; g.lineWidth = 0.035; rr(g, -L + 0.03, -W + 0.03, 2 * L - 0.06, 2 * W - 0.06, 0.18); g.stroke();
    cutWells(g, wheels);
    const ww = wheels[0].w;
    // top lids: matte black plate
    const px0 = -L + 0.2, px1 = L - 0.5, py = W - ww - 0.1;
    g.fillStyle = steel(g, px0, -py, px1, py, ["#34373c", "#1e2024", "#292b30", "#18191c"]);
    rr(g, px0, -py, px1 - px0, 2 * py, 0.1); g.fill();
    brushed(g, px0, -py, px1 - px0, 2 * py, 31, "x");
    g.strokeStyle = "rgba(255,255,255,.2)"; g.lineWidth = 0.025; rr(g, px0 + 0.02, -py + 0.02, px1 - px0 - 0.04, 2 * py - 0.04, 0.09); g.stroke();
    g.strokeStyle = "rgba(0,0,0,.7)"; g.lineWidth = 0.035; rr(g, px0, -py, px1 - px0, 2 * py, 0.1); g.stroke();
    // gold sunburst on each lid, fanning away from the arm
    g.save(); rr(g, px0 + 0.04, -py + 0.04, px1 - px0 - 0.08, 2 * py - 0.08, 0.08); g.clip();
    const rr1 = Math.min(py - 0.3, 1.0) + 0.25;
    sunburst(g, 0.05, 0.34, 0.1, rr1, 0.12 * Math.PI, 0.88 * Math.PI, 13);
    sunburst(g, 0.05, -0.34, 0.1, rr1, -0.88 * Math.PI, -0.12 * Math.PI, 13);
    g.restore();
    // red flames licking in from both flanks
    g.save(); rr(g, px0, -py, px1 - px0, 2 * py, 0.1); g.clip();
    flames(g, px0 + 0.05, px1 - 0.05, -py, 0.26, 1, 7);
    flames(g, px0 + 0.05, px1 - 0.05, py, 0.26, -1, 11);
    g.restore();
    // the arm slot: the arm lies in it when parked
    g.fillStyle = steel(g, 0, -0.3, 0, 0.3, ["#050607", "#15171a", "#050607"]);
    rr(g, -L - 0.01, -0.26, L + 0.25, 0.52, 0.06); g.fill();
    g.strokeStyle = "rgba(217,178,90,.45)"; g.lineWidth = 0.02; rr(g, -L - 0.01, -0.26, L + 0.25, 0.52, 0.06); g.stroke();
    // the front bar: black, flames running across it, a gold leading edge
    g.fillStyle = steel(g, L - 0.5, 0, L, 0, ["#141518", "#26282c", "#3a3d42"]);
    const wedge = [[L - 0.5, -W + 0.05], [L - 0.04, -W + 0.12], [L, -W + 0.2], [L, W - 0.2], [L - 0.04, W - 0.12], [L - 0.5, W - 0.05]];
    poly(g, wedge); g.fill();
    g.save(); poly(g, wedge); g.clip();
    g.translate(L - 0.5, 0); g.rotate(Math.PI / 2); flames(g, -W + 0.1, W - 0.1, 0, 0.36, -1, 23);
    g.restore();
    g.strokeStyle = NEM.goldHi; g.lineWidth = 0.035; g.beginPath(); g.moveTo(L - 0.015, -W + 0.2); g.lineTo(L - 0.015, W - 0.2); g.stroke();
    // the name on the starboard lid
    g.save(); g.translate(px0 + 0.28, py - 0.42); g.scale(0.01, 0.01);
    g.font = "italic 900 17px 'Arial Black', Impact, sans-serif"; g.textBaseline = "middle";
    g.fillStyle = NEM.red; g.fillText("NEMESIS", 0, 0);
    g.restore();
    // rear bumper and the removable power link
    g.fillStyle = steel(g, -L, 0, -L + 0.22, 0, ["#0b0c0e", "#2a2d31"]); rr(g, -L, -W + 0.08, 0.2, 2 * W - 0.16, 0.06); g.fill();
    g.fillStyle = "#111"; rr(g, -L + 0.32, -W + ww + 0.2, 0.22, 0.22, 0.04); g.fill();
    g.fillStyle = NEM.red; g.beginPath(); g.arc(-L + 0.43, -W + ww + 0.31, 0.06, 0, TAU); g.fill();
    // pivot blocks and the gold clutch either side of the slot
    for (const s of [-1, 1]) {
      g.fillStyle = steel(g, -0.24, 0, 0.24, 0, ["#15171a", "#3b3f45", "#15171a"]);
      rr(g, -0.24, s > 0 ? 0.28 : -0.62, 0.48, 0.34, 0.05); g.fill();
      g.fillStyle = steel(g, 0, s * 0.3, 0, s * 0.6, [NEM.goldHi, NEM.goldDeep]);
      poly(g, [[-0.08, s * 0.3], [0.08, s * 0.3], [0, s * 0.52]]); g.fill();
      bolt(g, -0.16, s * 0.47); bolt(g, 0.16, s * 0.47);
    }
    // perimeter bolts
    boltRow(g, px0 + 0.12, -py + 0.1, px1 - 0.12, -py + 0.1, 0.42);
    boltRow(g, px0 + 0.12, py - 0.1, px1 - 0.12, py - 0.1, 0.42);
  }

  // Theirs: blackened diamond-plate armour, orange trim, the weapon mount at the nose.
  function paintThem(g, e, th) {
    const L = e.half_len, W = e.half_wid, wheels = themWheels(e), ch = Math.min(0.42, W * 0.35);
    g.fillStyle = steel(g, 0, -W, 0, W, ["#3a3532", "#211d1b", "#141210"]);
    poly(g, [[-L, -W + 0.18], [-L + 0.18, -W], [L - ch, -W], [L, -W + ch], [L, W - ch], [L - ch, W], [-L + 0.18, W], [-L, W - 0.18]]); g.fill();
    g.strokeStyle = "rgba(255,255,255,.12)"; g.lineWidth = 0.035; g.stroke();
    cutWells(g, wheels);
    const ww = wheels[0].w, px0 = -L + 0.22, px1 = L - 0.3, py = W - ww - 0.1;
    // diamond plate
    g.fillStyle = steel(g, px0, -py, px1, py, ["#6b7078", "#454a51", "#5a6067", "#3a3f45"]);
    poly(g, [[px0, -py], [px1 - ch * 0.6, -py], [px1, -py + ch * 0.6], [px1, py - ch * 0.6], [px1 - ch * 0.6, py], [px0, py]]); g.fill();
    g.save(); g.clip();
    let k = 0;
    for (let x = px0 + 0.1; x < px1; x += 0.24) for (let y = -py + 0.08; y < py; y += 0.16) {
      const a = (k++ % 2 ? 1 : -1) * 0.8;
      g.save(); g.translate(x + ((y * 6.25 | 0) % 2) * 0.12, y); g.rotate(a);
      g.fillStyle = "rgba(0,0,0,.35)"; g.fillRect(-0.06 + 0.01, -0.016 + 0.012, 0.12, 0.032);
      g.fillStyle = "rgba(255,255,255,.16)"; g.fillRect(-0.06, -0.016, 0.12, 0.032);
      g.restore();
    }
    g.restore();
    g.strokeStyle = th.them; g.lineWidth = 0.07;
    poly(g, [[px0, -py], [px1 - ch * 0.6, -py], [px1, -py + ch * 0.6], [px1, py - ch * 0.6], [px1 - ch * 0.6, py], [px0, py]]); g.stroke();
    // orange racing slashes at the rear
    g.save(); g.beginPath(); g.rect(px0, -py, 0.9, 2 * py); g.clip();
    g.fillStyle = th.them;
    for (let i = 0; i < 3; i++) { const x = px0 + 0.12 + i * 0.26; poly(g, [[x, -py], [x + 0.12, -py], [x + 0.12 + 2 * py * 0.35, py], [x + 2 * py * 0.35, py]]); g.fill(); }
    g.restore();
    // rear bumper
    g.fillStyle = steel(g, -L, 0, -L + 0.22, 0, ["#100e0d", "#36302c"]); rr(g, -L, -W + 0.2, 0.2, 2 * W - 0.4, 0.05); g.fill();
    // weapon mount
    if (e.weapon === "horizontal" || e.rad > 0.8) {
      const x0 = Math.max(-L * 0.2, L - 1.1), x1 = e.fore + 0.15, hw = Math.min(0.46, W * 0.45);
      g.fillStyle = "rgba(0,0,0,.45)"; rr(g, x0 + 0.04, -hw + 0.06, x1 - x0, 2 * hw, 0.12); g.fill();
      g.fillStyle = steel(g, 0, -hw, 0, hw, ["#7e858d", "#3f454c", "#2a2e33"]); rr(g, x0, -hw, x1 - x0, 2 * hw, 0.12); g.fill();
      g.strokeStyle = th.them; g.lineWidth = 0.04; rr(g, x0, -hw, x1 - x0, 2 * hw, 0.12); g.stroke();
      boltRow(g, x0 + 0.12, -hw + 0.12, e.fore - 0.4, -hw + 0.12, 0.3);
      boltRow(g, x0 + 0.12, hw - 0.12, e.fore - 0.4, hw - 0.12, 0.3);
    } else if (e.weapon === "vert" || e.weapon === "drum") {
      for (const s of [-1, 1]) {
        const y = s * (e.wide + 0.14), x0 = L - 0.6, x1 = e.fore + e.rad * 0.55;
        g.fillStyle = "rgba(0,0,0,.45)"; rr(g, x0 + 0.04, y - 0.1 + 0.06, x1 - x0, 0.2, 0.06); g.fill();
        g.fillStyle = steel(g, 0, y - 0.1, 0, y + 0.1, ["#9aa1a9", "#4c535b"]); rr(g, x0, y - 0.1, x1 - x0, 0.2, 0.06); g.fill();
        bolt(g, x0 + 0.15, y); bolt(g, x1 - 0.12, y);
      }
    } else {                                                 // no weapon: a fixed plough
      const hw = Math.max(W, e.wide || 0) + 0.05, x0 = L - 0.25, x1 = L + 0.32;
      g.fillStyle = steel(g, x0, 0, x1, 0, ["#2c3137", "#7d858e", "#c6ccd2"]);
      poly(g, [[x0, -hw], [x1, -hw * 0.82], [x1, hw * 0.82], [x0, hw]]); g.fill();
      g.strokeStyle = "rgba(255,255,255,.6)"; g.lineWidth = 0.03; g.beginPath(); g.moveTo(x1 - 0.01, -hw * 0.82); g.lineTo(x1 - 0.01, hw * 0.82); g.stroke();
    }
    boltRow(g, px0 + 0.1, -py + 0.09, px1 - 0.5, -py + 0.09, 0.4);
    boltRow(g, px0 + 0.1, py - 0.09, px1 - 0.5, py - 0.09, 0.4);
  }

  function robotSprites(doc, th, themeKey, ppf) {
    return cached(`bots|${themeKey}|${ppf}`, () => {
      const o = doc.our, e = doc.enemy;
      const eFront = Math.max(e.half_len + 0.4, e.fore + e.rad + 0.3);
      const usSpr = sprite(-o.half_len - 0.1, -o.half_wid - 0.1, 2 * o.half_len + 0.2 + FORK_FT, 2 * o.half_wid + 0.2, ppf, g => paintUs(g, o, th));
      const eW = Math.max(e.half_wid, (e.wide || 0) + 0.4, e.rad > 0.8 ? 0 : 0) + 0.15;
      const themSpr = sprite(-e.half_len - 0.1, -eW, e.half_len + 0.1 + eFront, 2 * eW, ppf, g => paintThem(g, e, th));
      const blur = (x0, y0, w, h, path) => sprite(x0 - 0.5, y0 - 0.5, w + 1, h + 1, Math.min(ppf, 48), (g, p) => {
        g.filter = `blur(${0.13 * p}px)`; g.fillStyle = "#000"; path(g); g.fill();
      });
      const usShadow = blur(-o.half_len, -o.half_wid, 2 * o.half_len, 2 * o.half_wid, g => rr(g, -o.half_len, -o.half_wid, 2 * o.half_len, 2 * o.half_wid, 0.25));
      const themShadow = blur(-e.half_len, -e.half_wid, 2 * e.half_len, 2 * e.half_wid, g => rr(g, -e.half_len, -e.half_wid, 2 * e.half_len, 2 * e.half_wid, 0.25));
      return { usSpr, themSpr, usShadow, themShadow };
    });
  }

  function drawWheels(g, wheels, odoA, odoB) {
    for (const w of wheels) {
      const off = w.side < 0 ? odoA : odoB, x0 = w.x - w.l / 2, y0 = w.y - w.w / 2;
      g.fillStyle = "#101214"; rr(g, x0, y0, w.l, w.w, 0.07); g.fill();
      g.strokeStyle = "rgba(255,255,255,.13)"; g.lineWidth = 0.035;
      const pitch = 0.13, ph = ((off % pitch) + pitch) % pitch;
      g.beginPath();                                   // tread bars, rolling with the distance driven
      for (let x = x0 + ph; x < x0 + w.l - 0.04; x += pitch) { g.moveTo(x, y0 + 0.02); g.lineTo(x + 0.03, y0 + w.w - 0.02); }
      g.stroke();
      g.fillStyle = steel(g, x0, 0, x0 + w.l, 0, ["rgba(0,0,0,.75)", "rgba(255,255,255,.07)", "rgba(0,0,0,.75)"]);
      rr(g, x0, y0, w.l, w.w, 0.07); g.fill();
    }
  }

  // The opponent's weapon, in its own frame. phase = how far it has spun, sp = blade speed 0..1.
  function drawEnemyWeapon(g, e, phase, sp, th, cssPx) {
    const horizontal = e.weapon === "horizontal" || e.rad > 0.8;
    if (e.weapon === "none") return;
    if (horizontal) {
      const hx = e.fore, R = e.rad;
      if (sp > 0.03) {                           // the disc the bar sweeps, and its motion blur
        const dg = g.createRadialGradient(hx, 0, R * 0.15, hx, 0, R);
        dg.addColorStop(0, "rgba(200,208,216,0)"); dg.addColorStop(0.75, `rgba(200,208,216,${0.08 * sp})`); dg.addColorStop(1, `rgba(220,226,232,${0.2 * sp})`);
        g.fillStyle = dg; g.beginPath(); g.arc(hx, 0, R, 0, TAU); g.fill();
        const cg = g.createConicGradient(phase, hx, 0), d = clamp(0.08 + 0.3 * sp, 0, 0.42), pk = 0.55 * sp;
        cg.addColorStop(0, `rgba(235,238,242,${pk})`); cg.addColorStop(0.004, "rgba(235,238,242,0)");
        cg.addColorStop(0.5 - d, "rgba(235,238,242,0)"); cg.addColorStop(0.496, `rgba(235,238,242,${pk})`); cg.addColorStop(0.504, "rgba(235,238,242,0)");
        cg.addColorStop(1 - d, "rgba(235,238,242,0)"); cg.addColorStop(1, `rgba(235,238,242,${pk})`);
        g.fillStyle = cg; g.beginPath(); g.arc(hx, 0, R, 0, TAU); g.arc(hx, 0, R * 0.3, 0, TAU, true); g.fill();
        g.strokeStyle = th.them; g.globalAlpha = 0.35 + 0.45 * sp; g.lineWidth = Math.max(0.035, 1.2 * cssPx);
        g.setLineDash([0.18, 0.12]); g.lineDashOffset = -phase * R * 0.5;
        g.beginPath(); g.arc(hx, 0, R, 0, TAU); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
      }
      g.save(); g.translate(hx, 0);
      for (let j = 3; j >= 0; j--) {             // ghosts behind the bar, then the bar
        g.save(); g.rotate(phase - j * 0.09 * sp);
        g.globalAlpha = j ? 0.16 * sp * (1 - j / 4) : 1 - 0.25 * sp;
        if (j && sp < 0.15) { g.restore(); continue; }
        const Rb = R * 0.97;
        g.fillStyle = "rgba(0,0,0,.35)"; poly(g, [[-Rb, -0.13 + 0.05], [0, -0.22 + 0.05], [Rb, -0.13 + 0.05], [Rb, 0.13 + 0.05], [0, 0.22 + 0.05], [-Rb, 0.13 + 0.05]]); g.fill();
        g.fillStyle = steel(g, 0, -0.22, 0, 0.22, ["#e8ecf0", "#9aa3ad", "#59616a"]);
        poly(g, [[-Rb, -0.13], [0, -0.22], [Rb, -0.13], [Rb, 0.13], [0, 0.22], [-Rb, 0.13]]); g.fill();
        for (const s of [-1, 1]) {               // hardened impactor tips
          g.fillStyle = steel(g, 0, -0.18, 0, 0.18, ["#f5f7f9", "#7a838d", "#363b41"]);
          rr(g, s > 0 ? Rb - 0.32 : -Rb, -0.18, 0.32, 0.36, 0.05); g.fill();
          g.fillStyle = th.them; g.fillRect(s > 0 ? Rb - 0.36 : -Rb + 0.32, -0.15, 0.04, 0.3);
        }
        g.restore();
      }
      g.globalAlpha = 1;
      g.fillStyle = "#1a1d21"; g.beginPath(); g.arc(0, 0, 0.27, 0, TAU); g.fill();
      g.fillStyle = steel(g, -0.22, -0.22, 0.22, 0.22, ["#c9cfd6", "#59616a"]); g.beginPath(); g.arc(0, 0, 0.22, 0, TAU); g.fill();
      for (let k = 0; k < 6; k++) bolt(g, Math.cos(phase + k * TAU / 6) * 0.15, Math.sin(phase + k * TAU / 6) * 0.15, 0.028);
      g.fillStyle = th.them; g.beginPath(); g.arc(0, 0, 0.07, 0, TAU); g.fill();
      g.restore();
      return;
    }
    const x0 = e.fore, R = e.rad, Wd = Math.max(0.25, e.wide);
    if (e.weapon === "drum") {
      g.fillStyle = "rgba(0,0,0,.4)"; rr(g, x0 - R + 0.06, -Wd + 0.08, 2 * R, 2 * Wd, 0.08); g.fill();
      g.fillStyle = steel(g, x0 - R, 0, x0 + R, 0, ["#2b2f34", "#8c949d", "#d8dde2", "#8c949d", "#2b2f34"]);
      rr(g, x0 - R, -Wd, 2 * R, 2 * Wd, 0.08); g.fill();
      g.save(); rr(g, x0 - R, -Wd, 2 * R, 2 * Wd, 0.08); g.clip();
      const rows = Math.max(3, Math.round(Wd / 0.28));
      for (let j = 0; j < rows; j++) for (let k = 0; k < 2; k++) {
        const phi = phase + k * Math.PI + (j % 2) * Math.PI / 2, c = Math.cos(phi);
        if (c <= 0) continue;
        const tx = x0 + R * Math.sin(phi), ty = -Wd + (j + 0.5) * 2 * Wd / rows;
        g.fillStyle = `rgba(30,32,36,${0.4 + 0.6 * c * (1 - 0.6 * sp)})`; g.fillRect(tx - 0.07 * c - 0.02, ty - 0.11, 0.14 * c + 0.04, 0.22);
      }
      if (sp > 0.05) { g.fillStyle = `rgba(230,234,238,${0.22 * sp})`; g.fillRect(x0 - R, -Wd, 2 * R, 2 * Wd); }
      g.restore();
      g.strokeStyle = th.them; g.lineWidth = 0.04; g.globalAlpha = 0.6 + 0.4 * sp; rr(g, x0 - R, -Wd, 2 * R, 2 * Wd, 0.08); g.stroke(); g.globalAlpha = 1;
      return;
    }
    // vert: an eggbeater across the forks. From above each blade is a bar across the row,
    // sliding fore and aft as it turns; the swept band is what the sim collides.
    if (sp > 0.03) {
      const bg = g.createLinearGradient(x0 - R, 0, x0 + R, 0);
      bg.addColorStop(0, "rgba(225,230,236,0)"); bg.addColorStop(0.5, `rgba(225,230,236,${0.3 * sp})`); bg.addColorStop(1, `rgba(225,230,236,${0.12 * sp})`);
      g.fillStyle = bg; g.fillRect(x0 - R, -Wd, 2 * R, 2 * Wd);
    }
    g.strokeStyle = th.them; g.globalAlpha = 0.35 + 0.5 * sp; g.lineWidth = Math.max(0.03, cssPx * 1.2);
    g.setLineDash([0.16, 0.1]); g.lineDashOffset = -phase * 0.3; g.strokeRect(x0 - R, -Wd, 2 * R, 2 * Wd); g.setLineDash([]); g.globalAlpha = 1;
    g.fillStyle = steel(g, 0, -0.08, 0, 0.08, ["#a0a8b0", "#3a4047"]); g.fillRect(x0 - 0.09, -Wd - 0.12, 0.18, 2 * Wd + 0.24);
    for (let j = 2; j >= 0; j--) for (let k = 0; k < 2; k++) {
      const phi = phase + k * Math.PI - j * 0.22 * sp, c = Math.cos(phi), tx = x0 + R * Math.sin(phi);
      if (c < -0.1) continue;
      const a = (j ? 0.3 * sp : 1 - 0.3 * sp) * (0.6 + 0.4 * Math.max(0, c));
      if (a < 0.02) continue;
      g.globalAlpha = a;
      const bw = 0.13;
      g.fillStyle = "rgba(0,0,0,.45)"; rr(g, tx - bw + 0.05, -Wd + 0.07, 2 * bw, 2 * Wd, 0.05); g.fill();
      g.fillStyle = steel(g, tx - bw, 0, tx + bw, 0, ["#5a626b", "#f4f6f8", "#7c858f"]); rr(g, tx - bw, -Wd, 2 * bw, 2 * Wd, 0.05); g.fill();
      g.fillStyle = th.them; g.fillRect(tx - bw, -Wd, 2 * bw, 0.16); g.fillRect(tx - bw, Wd - 0.16, 2 * bw, 0.16);
    }
    g.globalAlpha = 1;
  }

  // Our arm and saw, in our frame. h = height of the saw above the floor (0..1).
  function drawSaw(g, doc, s, th, cssPx, oa, P, shadowOnly) {
    const o = doc.our, R = o.rad, along = -o.aft + (o.aft + o.fore) * s.arm;
    const h = Math.pow(Math.sin(Math.PI * clamp(s.arm, 0, 1)), 0.85), sc = 1 + 0.5 * h;
    if (shadowOnly) {                                   // cast onto floor and body; offset grows with height
      const k = 0.4 + 5.2 * h, dx = SHADOW.x * k, dy = SHADOW.y * k;
      const lx = Math.cos(-oa) * dx - Math.sin(-oa) * dy, ly = Math.sin(-oa) * dx + Math.cos(-oa) * dy;
      g.save(); g.translate(lx, ly); g.globalAlpha = th.shadow * (0.55 - 0.25 * h); g.fillStyle = "#000"; g.strokeStyle = "#000";
      g.lineWidth = 0.15; g.lineCap = "round";
      g.beginPath(); for (const y of [-0.34, 0.34]) { g.moveTo(0, y); g.lineTo(along, y * 0.5); } g.stroke();
      g.beginPath(); g.ellipse(along, 0, R, R * 0.4, 0, 0, TAU); g.fill();
      g.restore(); return;
    }
    // where the saw was over the last frames: a fading swing arc
    const flying = s.arm > doc.arm_parked && s.arm < doc.arm_down;
    if (flying) {                                       // where it will come down
      const q = clamp((s.arm - doc.arm_parked) / (doc.arm_down - doc.arm_parked), 0, 1), lw = Math.max(0.035, 1.3 * cssPx);
      g.strokeStyle = th.usHi; g.globalAlpha = 0.25 + 0.55 * q; g.lineWidth = lw;
      g.setLineDash([0.12, 0.09]); g.beginPath(); g.arc(o.fore, 0, R, 0, TAU); g.stroke(); g.setLineDash([]);
      g.globalAlpha = 0.12 + 0.25 * q; g.fillStyle = th.us; g.beginPath(); g.arc(o.fore, 0, R * (0.3 + 0.7 * q), 0, TAU); g.fill();
      g.globalAlpha = 1;
    }
    if (flying || s.arm >= doc.arm_down) {
      for (let j = 6; j >= 1; j--) {
        const idx = s.i0 - j; if (idx < 0) continue;
        const aj = doc.frames[idx][6]; if (Math.abs(aj - s.arm) < 0.02) continue;
        const al = -o.aft + (o.aft + o.fore) * aj, hj = Math.pow(Math.sin(Math.PI * clamp(aj, 0, 1)), 0.85), sj = 1 + 0.5 * hj;
        g.globalAlpha = 0.2 * (1 - j / 7); g.fillStyle = th.usHi;
        g.beginPath(); g.ellipse(al, 0, R * sj, R * 0.42 * sj, 0, 0, TAU); g.fill();
      }
      g.globalAlpha = 1;
    }
    // arm beams from the pivot to the hub
    g.lineCap = "round";
    for (const y of [-0.36, 0.36]) {
      const y1 = y * 0.55 * sc;
      g.strokeStyle = "#050607"; g.lineWidth = 0.17 * Math.sqrt(sc); g.beginPath(); g.moveTo(0, y); g.lineTo(along, y1); g.stroke();
      g.strokeStyle = "#2c2f34"; g.lineWidth = 0.11 * Math.sqrt(sc); g.beginPath(); g.moveTo(0, y); g.lineTo(along, y1); g.stroke();
      g.strokeStyle = NEM.red; g.lineWidth = 0.03 * sc; g.beginPath(); g.moveTo(0, y); g.lineTo(along, y1); g.stroke();
    }
    g.fillStyle = steel(g, -0.13, 0, 0.13, 0, ["#15171a", "#4a4e55", "#15171a"]); rr(g, -0.13, -0.6, 0.26, 1.2, 0.08); g.fill();
    // the saw: a disc spinning in the fore-aft plane, drawn with a little perspective
    g.save(); g.translate(along, 0); g.scale(sc, sc);
    const rx = R, ry = R * 0.42, sp = clamp(s.blade, 0, 1), ph = s.spinO;
    g.fillStyle = steel(g, 0, -ry, 0, ry, [NEM.redHi, NEM.red, NEM.redDeep]);
    g.beginPath(); g.ellipse(0, 0, rx, ry, 0, 0, TAU); g.fill();
    g.strokeStyle = "rgba(0,0,0,.35)"; g.lineWidth = 0.02; g.beginPath(); g.ellipse(0, 0, rx * 0.7, ry * 0.7, 0, 0, TAU); g.stroke();
    const N = 2;
    g.fillStyle = `rgba(227,32,43,${1 - 0.6 * sp})`;
    g.beginPath();
    for (let k = 0; k < N; k++) {
      const a = ph + k * TAU / N, a2 = a + 0.55;
      g.moveTo(Math.cos(a) * rx, Math.sin(a) * ry);
      g.lineTo(Math.cos(a) * rx * 1.24, Math.sin(a) * ry * 1.24);
      g.lineTo(Math.cos(a2) * rx, Math.sin(a2) * ry);
    }
    g.fill();
    if (sp > 0.05) {
      g.strokeStyle = `rgba(255,120,120,${0.55 * sp})`; g.lineWidth = 0.06;
      g.beginPath(); g.ellipse(0, 0, rx * 1.05, ry * 1.05, 0, 0, TAU); g.stroke();
      g.strokeStyle = th.us; g.globalAlpha = 0.5 * sp; g.lineWidth = Math.max(0.03, cssPx * 1.2);
      g.beginPath(); g.ellipse(0, 0, rx * 1.16, ry * 1.16, 0, 0, TAU); g.stroke(); g.globalAlpha = 1;
    }
    g.fillStyle = NEM.gold; g.beginPath(); g.ellipse(0, 0, 0.2, 0.1, 0, 0, TAU); g.fill();
    g.fillStyle = "#1c2026"; g.beginPath(); g.ellipse(0, 0, 0.08, 0.045, 0, 0, TAU); g.fill();
    g.restore();
    // the strike circle the sim checks, while the saw is down
    if (s.arm >= doc.arm_down) {
      g.strokeStyle = th.us; g.globalAlpha = 0.6; g.lineWidth = Math.max(0.03, cssPx * 1.2);
      g.setLineDash([0.14, 0.1]); g.beginPath(); g.arc(along, 0, R, 0, TAU); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
  }

  // ───────────────────────────── effects (pure functions of age) ─────────────────────────────
  function sparks(g, x, y, age, seed, n, vmin, vmax, lmin, lmax, dir, spread, cols, wpx) {
    const r = rng(seed), drag = 4.2;
    g.lineCap = "round";
    for (let i = 0; i < n; i++) {
      const a = dir + (r() - 0.5) * spread, v = lerp(vmin, vmax, r()), life = lerp(lmin, lmax, r()), w = r();
      if (age > life) continue;
      const q = age / life, d = v * (1 - Math.exp(-drag * age)) / drag, vel = v * Math.exp(-drag * age);
      const tail = Math.min(d, vel * 0.05 + 0.04), c = Math.cos(a), s = Math.sin(a);
      g.strokeStyle = cols[Math.min(cols.length - 1, Math.floor(q * cols.length))];
      g.globalAlpha = 1 - q * q;
      g.lineWidth = wpx * (0.7 + w) * (1 - 0.5 * q);
      g.beginPath(); g.moveTo(x + c * (d - tail), y + s * (d - tail)); g.lineTo(x + c * d, y + s * d); g.stroke();
    }
    g.globalAlpha = 1;
  }
  function ring(g, x, y, r, lw, color, alpha, dash) {
    if (alpha <= 0.002) return;
    g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = lw;
    if (dash) g.setLineDash(dash);
    g.beginPath(); g.arc(x, y, Math.max(0.01, r), 0, TAU); g.stroke();
    if (dash) g.setLineDash([]);
    g.globalAlpha = 1;
  }
  function glow(g, x, y, r, color, alpha) {
    if (alpha <= 0.002) return;
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, color); gr.addColorStop(1, "rgba(0,0,0,0)");
    g.globalAlpha = alpha; g.fillStyle = gr; g.fillRect(x - r, y - r, 2 * r, 2 * r); g.globalAlpha = 1;
  }
  const LIFE = { solid: 1.1, weak: 0.8, side: 1.1, floor: 1.0 };

  function decal(g, L, age, th) {
    const a = clamp(age * 10, 0, 1) * (0.35 + 0.65 * Math.exp(-age / 30));
    const r = rng(L.seed ^ 0x55), c = Math.cos(L.a), s = Math.sin(L.a);
    g.save(); g.globalAlpha = a; g.lineCap = "round";
    if (L.kind === "floor") {                         // the saw bit the floor: a cut along our heading
      g.strokeStyle = th.scorch; g.lineWidth = 0.13;
      g.beginPath(); g.moveTo(L.x - c * 0.55, L.y - s * 0.55); g.lineTo(L.x + c * 0.55, L.y + s * 0.55); g.stroke();
      g.strokeStyle = th.cut; g.lineWidth = 0.03;
      g.beginPath(); g.moveTo(L.x - c * 0.45, L.y - s * 0.45); g.lineTo(L.x + c * 0.45, L.y + s * 0.45); g.stroke();
    } else {                                          // a scorch where the strike landed
      const rad = L.kind === "solid" ? 0.85 : 0.55;
      glow(g, L.x, L.y, rad, th.scorch, a * 0.7);
      g.globalAlpha = a * 0.35; g.strokeStyle = th.scorch; g.lineWidth = 0.03;
      g.beginPath();
      for (let i = 0; i < 9; i++) { const q = r() * TAU, l = rad * (0.6 + r() * 0.9); g.moveTo(L.x + Math.cos(q) * rad * 0.25, L.y + Math.sin(q) * rad * 0.25); g.lineTo(L.x + Math.cos(q) * l, L.y + Math.sin(q) * l); }
      g.stroke();
    }
    g.restore();
  }

  function landFx(g, L, age, th, cssPx, layer) {
    const add = th.additive ? "lighter" : "source-over", wpx = Math.max(0.03, 1.6 * cssPx);
    if (L.kind === "solid") {
      if (layer === "under") {
        g.globalCompositeOperation = add; glow(g, L.x, L.y, 2.6, th.gold, 0.55 * clamp(1 - age / 0.35, 0, 1)); g.globalCompositeOperation = "source-over";
      } else {
        const u = clamp(age / 0.6, 0, 1);
        ring(g, L.x, L.y, 0.4 + 3.6 * ease3(u), 0.05 + 0.24 * (1 - u), th.gold, 0.9 * Math.pow(1 - u, 1.5));
        const u2 = clamp((age - 0.07) / 0.5, 0, 1);
        if (age > 0.07) ring(g, L.x, L.y, 0.3 + 2.4 * ease3(u2), 0.03 + 0.08 * (1 - u2), th.hot, 0.8 * Math.pow(1 - u2, 2));
        g.globalCompositeOperation = add;
        if (age < 0.16) glow(g, L.x, L.y, 0.4 + 1.6 * (age / 0.16), th.hot, 1 - age / 0.16);
        sparks(g, L.x, L.y, age, L.seed, 46, 8, 22, 0.35, 0.85, L.a, TAU, [th.hot, th.gold, "#ff9a2e", "#c8401a"], wpx);
        sparks(g, L.x, L.y, age, L.seed + 1, 18, 10, 26, 0.2, 0.5, L.a + Math.PI / 2, 0.9, [th.hot, th.gold], wpx * 1.2);
        sparks(g, L.x, L.y, age, L.seed + 2, 18, 10, 26, 0.2, 0.5, L.a - Math.PI / 2, 0.9, [th.hot, th.gold], wpx * 1.2);
        g.globalCompositeOperation = "source-over";
        const r = rng(L.seed + 3);                        // chunks of armour tumbling away
        for (let i = 0; i < 7; i++) {
          const a = r() * TAU, v = 3 + r() * 5, life = 0.6 + r() * 0.5, rot = (r() - 0.5) * 30;
          if (age > life) continue;
          const d = v * (1 - Math.exp(-3 * age)) / 3;
          g.save(); g.translate(L.x + Math.cos(a) * d, L.y + Math.sin(a) * d); g.rotate(rot * age);
          g.globalAlpha = 1 - age / life; g.fillStyle = "#9aa2ab"; g.fillRect(-0.06, -0.035, 0.12, 0.07);
          g.restore();
        }
        g.globalAlpha = 1;
      }
    } else if (L.kind === "weak") {
      if (layer === "over") {
        const u = clamp(age / 0.45, 0, 1);
        ring(g, L.x, L.y, 0.35 + 1.9 * ease3(u), 0.03 + 0.1 * (1 - u), th.usHi, 0.85 * Math.pow(1 - u, 1.5));
        g.globalCompositeOperation = add;
        if (age < 0.1) glow(g, L.x, L.y, 1.0, th.hot, 0.8 * (1 - age / 0.1));
        sparks(g, L.x, L.y, age, L.seed, 18, 5, 13, 0.25, 0.6, L.a, TAU, [th.hot, th.gold, "#ff9a2e"], wpx);
        g.globalCompositeOperation = "source-over";
      }
    } else if (L.kind === "side") {
      if (layer === "over") {
        const u = clamp(age / 0.7, 0, 1);
        g.save(); g.translate(L.x, L.y); g.rotate(age * 3);
        ring(g, 0, 0, 0.5 + 2.6 * ease3(u), 0.1 + 0.12 * (1 - u), th.amber, Math.pow(1 - u, 1.3), [0.35, 0.22]);
        g.restore();
        ring(g, L.x, L.y, 0.4 + 1.5 * ease3(u), 0.05, th.amber, 0.7 * Math.pow(1 - u, 2));
        g.globalCompositeOperation = add;
        if (age < 0.14) glow(g, L.x, L.y, 1.4, th.amber, 0.9 * (1 - age / 0.14));
        sparks(g, L.x, L.y, age, L.seed, 30, 7, 18, 0.3, 0.7, L.a + Math.PI, TAU, [th.hot, th.amber, "#ff7b1c"], wpx);
        g.globalCompositeOperation = "source-over";
      }
    } else {                                              // floor: dust and grit
      if (layer === "under") {
        const r = rng(L.seed);
        for (let i = 0; i < 20; i++) {
          const a = L.a + (r() - 0.5) * 4.5, dist = 0.4 + r() * 1.6, size = 0.25 + r() * 0.5, life = 0.6 + r() * 0.5;
          if (age > life) continue;
          const u = age / life, d = dist * ease3(u);
          g.fillStyle = `rgba(${th.dust},${0.5 * (1 - u) * (1 - u * 0.3)})`;
          g.beginPath(); g.arc(L.x + Math.cos(a) * d, L.y + Math.sin(a) * d, size * (0.5 + u), 0, TAU); g.fill();
        }
        const u = clamp(age / 0.5, 0, 1);
        ring(g, L.x, L.y, 0.3 + 1.4 * ease3(u), 0.05, `rgb(${th.dust})`, 0.5 * (1 - u));
      } else {
        g.globalCompositeOperation = add;
        sparks(g, L.x, L.y, age, L.seed, 10, 3, 8, 0.15, 0.4, L.a, 2.2, [th.gold, "#c8601a"], wpx * 0.8);
        g.globalCompositeOperation = "source-over";
      }
    }
  }

  function hitFx(g, H, age, th, cssPx) {
    const add = th.additive ? "lighter" : "source-over";
    g.globalCompositeOperation = add;
    if (age < 0.15) glow(g, H.x, H.y, 1.0, th.red, 0.9 * (1 - age / 0.15));
    sparks(g, H.x, H.y, age, H.seed, 16, 5, 14, 0.18, 0.45, H.a + Math.PI, 2.6, [th.hot, "#ffb347", th.red], Math.max(0.03, 1.4 * cssPx));
    g.globalCompositeOperation = "source-over";
  }

  // ───────────────────────────── screen-space overlays ─────────────────────────────
  function setFont(g, weight, size) {
    g.font = `${weight} ${size}px ${FONT}`;
    try { g.fontStretch = "condensed"; } catch (e) { /* older canvas */ }
  }
  function plate(g, x, y, text, fill, ink, size, pop, alpha) {   // a slanted broadcast caption centred on x
    setFont(g, 700, size);
    try { g.letterSpacing = (size * 0.08).toFixed(1) + "px"; } catch (e) { /* older canvas */ }
    const w = g.measureText(text).width + size * 1.3, h = size * 1.55, sk = h * 0.28;
    g.save(); g.globalAlpha = alpha; g.translate(x, y); g.scale(pop, pop);
    g.fillStyle = "rgba(0,0,0,.45)"; poly(g, [[-w / 2 + sk + 2, -h / 2 + 3], [w / 2 + sk + 2, -h / 2 + 3], [w / 2 - sk + 2, h / 2 + 3], [-w / 2 - sk + 2, h / 2 + 3]]); g.fill();
    g.fillStyle = fill; poly(g, [[-w / 2 + sk, -h / 2], [w / 2 + sk, -h / 2], [w / 2 - sk, h / 2], [-w / 2 - sk, h / 2]]); g.fill();
    g.fillStyle = ink; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, size * 0.04, size * 0.06);
    g.restore();
    try { g.letterSpacing = "0px"; } catch (e) { /* older canvas */ }
  }
  const CAPTION = { solid: ["SOLID HIT", "gold", "#141008"], weak: ["LIGHT HIT", "usHi", "#06121f"], side: ["WEAPON SIDE", "amber", "#1a1200"], floor: ["MISS", "miss", "#0d0f12"] };

  // How far a robot reaches up and down the screen, for placing its name tag.
  function extent(a, pts) {
    let up = 0, dn = 0, side = 0;
    const sa = Math.sin(a), ca = Math.cos(a);
    for (const [x, y, r] of pts) {
      const wy = x * sa + y * ca, wx = x * ca - y * sa;
      up = Math.max(up, -wy + r); dn = Math.max(dn, wy + r); side = Math.max(side, Math.abs(wx) + r);
    }
    return [up, dn, side];
  }

  // ───────────────────────────── the frame ─────────────────────────────
  function draw(canvas, doc, view) {
    if (!canvas || !doc || !doc.frames || !doc.frames.length) return;
    view = view || {};
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width, cssH = canvas.clientHeight || cssW;
    const cw = Math.max(1, Math.round(cssW * dpr)), ch = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    if (S.doc !== doc) { S.doc = doc; S.P = prep(doc); S.cam = null; S.cache.clear(); }
    const P = S.P, A = P.A, o = doc.our, e = doc.enemy, F = doc.frames;
    const themeKey = view.theme === "light" ? "light" : "dark", th = THEMES[themeKey];
    const t = clamp(+view.t || 0, 0, P.total), s = sample(doc, P, t);

    // camera: frame both robots, or the whole arena; ease only while playing
    const now = performance.now(), dtR = clamp((now - S.lastNow) / 1000, 0, 0.1); S.lastNow = now;
    const full = A + 2 * RIM, minDim = Math.min(cw, ch);
    const want = view.follow
      ? { span: clamp(Math.hypot(s.ox - s.ex, s.oy - s.ey) * 1.3 + 10, 16, full), x: (s.ox + s.ex) / 2, y: (s.oy + s.ey) / 2 }
      : { span: full, x: A / 2, y: A / 2 };
    const jump = S.lastT < 0 || Math.abs(t - S.lastT) > 1.0; S.lastT = t;
    if (!S.cam || !view.playing || jump) S.cam = { span: want.span, x: want.x, y: want.y };
    else {
      const k = 1 - Math.exp(-dtR * 3.2 * Math.sqrt(Math.max(1, view.speed || 1)));
      S.cam.span = lerp(S.cam.span, want.span, k); S.cam.x = lerp(S.cam.x, want.x, k); S.cam.y = lerp(S.cam.y, want.y, k);
    }
    const kpx = minDim / S.cam.span, vw = cw / kpx, vh = ch / kpx;
    const cx = vw >= full ? A / 2 : clamp(S.cam.x, -RIM + vw / 2, A + RIM - vw / 2);
    const cy = vh >= full ? A / 2 : clamp(S.cam.y, -RIM + vh / 2, A + RIM - vh / 2);
    // a small kick on solid strikes
    let shx = 0, shy = 0;
    if (!REDUCED) for (const L of P.lands) {
      if (L.kind !== "solid" && L.kind !== "side") continue;
      const age = t - L.t; if (age < 0 || age > 0.45) continue;
      const amp = (L.kind === "solid" ? 0.2 : 0.08) * Math.pow(1 - age / 0.45, 2) * (S.cam.span / 24);
      shx += amp * Math.sin(age * 92 + (L.seed % 7)); shy += amp * Math.cos(age * 71 + (L.seed % 5));
    }
    const ox0 = cw / 2 - (cx + shx) * kpx, oy0 = ch / 2 - (cy + shy) * kpx;
    const toScreen = (x, y) => [x * kpx + ox0, y * kpx + oy0];
    const cssPx = dpr / kpx;                              // one CSS pixel, in feet

    const g = canvas.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
    if (shx || shy || vw > full || vh > full) { g.fillStyle = th.void; g.fillRect(0, 0, cw, ch); }
    g.setTransform(kpx, 0, 0, kpx, ox0, oy0);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "medium";

    // floor
    const fppf = kpx > 36 ? 48 : kpx > 18 ? 32 : 16;
    const floor = cached(`floor|${themeKey}|${fppf}`, () => buildFloor(P, th, fppf));
    g.imageSmoothingQuality = kpx >= fppf ? "low" : "medium";      // bilinear is enough when enlarging
    g.drawImage(floor.c, -RIM, -RIM, floor.c.width / fppf, floor.c.height / fppf);
    g.imageSmoothingQuality = "medium";

    // marks left on the floor so far
    for (const L of P.lands) { const age = t - L.t; if (age < 0) break; decal(g, L, age, th); }

    // tyre tracks: the last five seconds
    const back = Math.round(5 / P.dt);
    g.lineCap = "round"; g.lineJoin = "round";
    for (const [c, hw, wl] of [[0, o.half_wid * 0.85, 0.22], [3, e.half_wid * 0.85, 0.2]]) {
      const start = Math.max(0, s.i0 - back), bands = 6, per = Math.ceil((s.i0 - start + 1) / bands);
      g.strokeStyle = th.track; g.lineWidth = wl;
      for (let b = 0; b < bands; b++) {
        const i0 = start + b * per, i1 = Math.min(s.i0, i0 + per);
        if (i1 <= i0) continue;
        g.globalAlpha = 0.4 * ((b + 1) / bands) ** 1.8;
        g.beginPath();
        for (const side of [-1, 1]) {
          for (let i = i0; i <= i1; i = i < i1 && i + 2 > i1 ? i1 : i + 2) {
            const r = F[i], a = r[c + 2], x = r[c] - Math.sin(a) * hw * side, y = r[c + 1] + Math.cos(a) * hw * side;
            if (i === i0) g.moveTo(x, y); else g.lineTo(x, y);
          }
        }
        g.stroke();
      }
    }
    g.globalAlpha = 1;

    // effects that sit on the floor
    for (const L of P.lands) { const age = t - L.t; if (age < 0) break; if (age < LIFE[L.kind]) landFx(g, L, age, th, cssPx, "under"); }

    // robots
    const spr = robotSprites(doc, th, themeKey, clamp(2 ** Math.ceil(Math.log2(Math.max(1, kpx))), 16, 128));
    const usThrown = s.flags & 1, themThrown = s.flags & 2;
    const wob = (on, seed) => on ? 0.11 * Math.sin(t * 19 + seed) : 0;
    const lift = on => on ? 2.4 : 1;
    const solidAge = (() => { let best = 9; for (const L of P.lands) { const d = t - L.t; if (d < 0) break; if (L.kind === "solid" || L.kind === "weak") best = d; } return best; })();
    const hitAge = (() => { let best = 9; for (const H of P.hitFx) { const d = t - H.t; if (d < 0) break; best = d; } return best; })();

    // zoomed out, a team-coloured ring under each robot helps the eye find it
    const halo = clamp((S.cam.span - 18) / 26, 0, 1);
    if (halo > 0.01) for (const [x, y, col, rad] of [[s.ex, s.ey, th.them, Math.hypot(e.half_len, e.half_wid) + 0.9], [s.ox, s.oy, th.us, Math.hypot(o.half_len, o.half_wid) + 0.9]]) {
      g.globalCompositeOperation = th.additive ? "lighter" : "source-over";
      glow(g, x, y, rad * 1.25, col, 0.3 * halo);
      g.globalCompositeOperation = "source-over";
      ring(g, x, y, rad, Math.max(0.05, 1.5 * cssPx), col, 0.75 * halo);
    }

    // opponent
    {
      const a = s.ea + wob(themThrown, 1), l = lift(themThrown);
      g.save(); g.translate(s.ex + SHADOW.x * l, s.ey + SHADOW.y * l); g.rotate(a); g.globalAlpha = th.shadow; blit(g, spr.themShadow); g.restore();
      if (e.weapon === "horizontal" || e.rad > 0.8) {     // the bar rides high: its shadow falls further out
        g.save(); g.translate(s.ex + SHADOW.x * 2.2 * l, s.ey + SHADOW.y * 2.2 * l); g.rotate(a); g.translate(e.fore, 0); g.rotate(s.spinE);
        g.globalAlpha = th.shadow * 0.5; g.fillStyle = "#000"; rr(g, -e.rad, -0.17, 2 * e.rad, 0.34, 0.1); g.fill(); g.restore();
      }
      g.save(); g.translate(s.ex, s.ey); g.rotate(a); if (themThrown) g.scale(1.06, 1.06);
      drawWheels(g, themWheels(e), s.odo[2], s.odo[3]);
      blit(g, spr.themSpr);
      drawEnemyWeapon(g, e, s.spinE, clamp(s.eblade, 0, 1), th, cssPx);
      if (solidAge < 0.12) {                             // our strike lights them up
        g.globalCompositeOperation = th.additive ? "lighter" : "source-over";
        g.globalAlpha = 0.22 * (1 - solidAge / 0.12); g.fillStyle = th.gold;
        rr(g, -e.half_len, -e.half_wid, 2 * e.half_len, 2 * e.half_wid, 0.2); g.fill();
        g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
      }
      g.restore();
    }
    // us
    {
      const a = s.oa + wob(usThrown, 2), l = lift(usThrown);
      g.save(); g.translate(s.ox + SHADOW.x * l, s.oy + SHADOW.y * l); g.rotate(a); g.globalAlpha = th.shadow; blit(g, spr.usShadow); g.restore();
      g.save(); g.translate(s.ox, s.oy); g.rotate(a); if (usThrown) g.scale(1.06, 1.06);
      drawWheels(g, usWheels(o), s.odo[0], s.odo[1]);
      blit(g, spr.usSpr);
      drawSaw(g, doc, s, th, cssPx, a, P, true);
      drawSaw(g, doc, s, th, cssPx, a, P, false);
      if (hitAge < 0.25) {                               // we took a hit: flash red
        const q = 1 - hitAge / 0.25;
        g.globalAlpha = 0.28 * q * q; g.fillStyle = th.red;
        rr(g, -o.half_len, -o.half_wid, 2 * o.half_len, 2 * o.half_wid, 0.2); g.fill();
        g.globalAlpha = 0.9 * q; g.strokeStyle = th.red; g.lineWidth = Math.max(0.05, 2 * cssPx);
        rr(g, -o.half_len - 0.08, -o.half_wid - 0.08, 2 * o.half_len + 0.16, 2 * o.half_wid + 0.16, 0.25); g.stroke();
        g.globalAlpha = 1;
      }
      g.restore();
    }
    // thrown: a turning hazard ring and the robot rocks
    for (const [on, x, y, rad] of [[usThrown, s.ox, s.oy, Math.hypot(o.half_len, o.half_wid) + 0.35], [themThrown, s.ex, s.ey, Math.hypot(e.half_len, e.half_wid) + 0.35]]) {
      if (!on) continue;
      g.save(); g.translate(x, y); g.rotate(t * 2.5);
      ring(g, 0, 0, rad, Math.max(0.06, 2 * cssPx), th.amber, 0.65 + 0.3 * Math.sin(t * 12), [0.4, 0.25]);
      g.fillStyle = th.amber;
      for (let k = 0; k < 3; k++) {
        const q = k * TAU / 3, sx = Math.cos(q) * rad, sy = Math.sin(q) * rad, z = 0.16;
        poly(g, [[sx, sy - z], [sx + z * 0.3, sy - z * 0.3], [sx + z, sy], [sx + z * 0.3, sy + z * 0.3], [sx, sy + z], [sx - z * 0.3, sy + z * 0.3], [sx - z, sy], [sx - z * 0.3, sy - z * 0.3]]); g.fill();
      }
      g.restore();
    }

    // effects above the robots
    for (const L of P.lands) { const age = t - L.t; if (age < 0) break; if (age < LIFE[L.kind]) landFx(g, L, age, th, cssPx, "over"); }
    for (const H of P.hitFx) { const age = t - H.t; if (age < 0) break; if (age < 0.5) hitFx(g, H, age, th, cssPx); }

    // ── screen space ──
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
    {                                                     // vignette: four edge bands touch fewer pixels than a full-screen layer
      const bw = Math.round(Math.min(cw, ch) * 0.2);
      for (const [x0, y0, x1, y1, rx, ry, rw, rh] of [[0, 0, 0, bw, 0, 0, cw, bw], [0, ch, 0, ch - bw, 0, ch - bw, cw, bw], [0, 0, bw, 0, 0, 0, bw, ch], [cw, 0, cw - bw, 0, cw - bw, 0, bw, ch]]) {
        const vg = g.createLinearGradient(x0, y0, x1, y1);
        vg.addColorStop(0, th.vignette); vg.addColorStop(0.45, th.vignette.replace(/[\d.]+\)$/, m => (parseFloat(m) * 0.35).toFixed(3) + ")")); vg.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = vg; g.fillRect(rx, ry, rw, rh);
      }
    }
    if (hitAge < 0.4) {                                   // red pulse at the edges when we are hit
      const q = 1 - hitAge / 0.4, R = Math.hypot(cw, ch) / 2;
      const gr = g.createRadialGradient(cw / 2, ch / 2, R * 0.55, cw / 2, ch / 2, R);
      gr.addColorStop(0, "rgba(255,0,0,0)"); gr.addColorStop(1, th.red);
      g.globalAlpha = 0.28 * q * q; g.fillStyle = gr; g.fillRect(0, 0, cw, ch); g.globalAlpha = 1;
    }

    // name tags
    const fs = Math.round(clamp(minDim / dpr / 52, 11, 15) * dpr), pad = fs * 0.55, gapPx = 10 * dpr;
    const usPts = [[o.half_len, o.half_wid, 0], [o.half_len, -o.half_wid, 0], [-o.half_len, o.half_wid, 0], [-o.half_len, -o.half_wid, 0], [-o.aft + (o.aft + o.fore) * s.arm, 0, o.rad]];
    const enPts = [[e.half_len, e.half_wid, 0], [e.half_len, -e.half_wid, 0], [-e.half_len, e.half_wid, 0], [-e.half_len, -e.half_wid, 0]];
    if (e.weapon !== "none") enPts.push(...(e.rad > 0.8 ? [[e.fore, 0, e.rad]] : [[e.fore, e.wide, e.rad], [e.fore, -e.wide, e.rad]]));
    const tags = [
      { name: "US", sub: usThrown ? "THROWN" : "", col: th.us, x: s.ox, y: s.oy, ext: extent(s.oa, usPts) },
      { name: String(e.tactic || "Opponent").toUpperCase(), sub: themThrown ? "THROWN" : "", col: th.them, x: s.ex, y: s.ey, ext: extent(s.ea, enPts) },
    ];
    setFont(g, 700, fs);
    for (const T of tags) {
      try { g.letterSpacing = (fs * 0.1).toFixed(1) + "px"; } catch (err) { /* older canvas */ }
      const wName = g.measureText(T.name).width;
      setFont(g, 600, Math.round(fs * 0.82));
      const wSub = T.sub ? g.measureText(T.sub).width + pad : 0;
      setFont(g, 700, fs);
      T.w = wName + wSub + pad * 2 + 5 * dpr; T.h = Math.round(fs * 1.75); T.wName = wName;
      [T.sx, T.sy] = toScreen(T.x, T.y);
      const rx = Math.max(T.ext[2], 0.5) * kpx;
      T.box = [T.sx - rx, T.sy - T.ext[0] * kpx, T.sx + rx, T.sy + T.ext[1] * kpx];   // the robot on screen
      T.cands = [                                          // above, below, then beside
        { tx: T.sx - T.w / 2, ty: T.box[1] - gapPx - T.h, dir: 1, cost: 0 },
        { tx: T.sx - T.w / 2, ty: T.box[3] + gapPx, dir: -1, cost: 0.3 },
        { tx: T.box[2] + gapPx, ty: T.sy - T.h / 2, dir: 0, cost: 0.6 },
        { tx: T.box[0] - gapPx - T.w, ty: T.sy - T.h / 2, dir: 0, cost: 0.6 },
      ];
    }
    const area = (x0, y0, x1, y1, b) => Math.max(0, Math.min(x1, b[2]) - Math.max(x0, b[0])) * Math.max(0, Math.min(y1, b[3]) - Math.max(y0, b[1]));
    const margin = 6 * dpr;
    const place = (T, c) => ({ ...c, tx: clamp(c.tx, margin, cw - T.w - margin), ty: clamp(c.ty, margin, ch - T.h - margin), moved: Math.abs(clamp(c.tx, margin, cw - T.w - margin) - c.tx) + Math.abs(clamp(c.ty, margin, ch - T.h - margin) - c.ty) });
    let bestPair = null, bestCost = Infinity;
    for (const c0 of tags[0].cands) for (const c1 of tags[1].cands) {
      const a = place(tags[0], c0), b = place(tags[1], c1), pad2 = 3 * dpr;
      const ra = [a.tx - pad2, a.ty - pad2, a.tx + tags[0].w + pad2, a.ty + tags[0].h + pad2];
      const cost = (a.cost + b.cost) * tags[0].h * tags[0].h
        + 4 * area(ra[0], ra[1], ra[2], ra[3], [b.tx, b.ty, b.tx + tags[1].w, b.ty + tags[1].h])
        + area(a.tx, a.ty, a.tx + tags[0].w, a.ty + tags[0].h, tags[1].box) + area(a.tx, a.ty, a.tx + tags[0].w, a.ty + tags[0].h, tags[0].box)
        + area(b.tx, b.ty, b.tx + tags[1].w, b.ty + tags[1].h, tags[0].box) + area(b.tx, b.ty, b.tx + tags[1].w, b.ty + tags[1].h, tags[1].box)
        + 2 * (a.moved + b.moved) * tags[0].h;
      if (cost < bestCost) { bestCost = cost; bestPair = [a, b]; }
    }
    tags.forEach((T, i) => Object.assign(T, { tx: bestPair[i].tx, ty: bestPair[i].ty, dir: bestPair[i].dir }));
    for (const T of tags) {
      g.fillStyle = "rgba(0,0,0,.28)"; rr(g, T.tx + dpr, T.ty + 2 * dpr, T.w, T.h, 3 * dpr); g.fill();
      g.fillStyle = th.tagBg; rr(g, T.tx, T.ty, T.w, T.h, 3 * dpr); g.fill();
      g.strokeStyle = th.tagLine; g.lineWidth = dpr; rr(g, T.tx + 0.5 * dpr, T.ty + 0.5 * dpr, T.w - dpr, T.h - dpr, 3 * dpr); g.stroke();
      g.fillStyle = T.col; g.fillRect(T.tx, T.ty, 4 * dpr, T.h);
      // pointer toward the robot
      g.fillStyle = th.tagBg;
      if (T.dir) {
        const px = clamp(T.sx, T.tx + 8 * dpr, T.tx + T.w - 8 * dpr), py = T.dir > 0 ? T.ty + T.h : T.ty;
        poly(g, [[px - 5 * dpr, py], [px + 5 * dpr, py], [px, py + T.dir * 5 * dpr]]); g.fill();
      } else {
        const right = T.tx > T.sx, px = right ? T.tx : T.tx + T.w, py = T.ty + T.h / 2, d = right ? -1 : 1;
        poly(g, [[px, py - 5 * dpr], [px, py + 5 * dpr], [px + d * 5 * dpr, py]]); g.fill();
      }
      setFont(g, 700, fs);
      try { g.letterSpacing = (fs * 0.1).toFixed(1) + "px"; } catch (err) { /* older canvas */ }
      g.fillStyle = th.tagInk; g.textAlign = "left"; g.textBaseline = "middle";
      g.fillText(T.name, T.tx + 4 * dpr + pad, T.ty + T.h / 2 + fs * 0.05);
      if (T.sub) {
        setFont(g, 600, Math.round(fs * 0.82)); g.fillStyle = th.amber;
        g.fillText(T.sub, T.tx + 4 * dpr + pad + T.wName + pad, T.ty + T.h / 2 + fs * 0.05);
      }
    }
    try { g.letterSpacing = "0px"; } catch (err) { /* older canvas */ }

    // captions at the impacts
    const cs = Math.round(clamp(minDim / dpr / 40, 12, 19) * dpr);
    for (const L of P.lands) {
      const age = t - L.t; if (age < 0) break; if (age > 1.6) continue;
      const [x, y] = toScreen(L.x, L.y), [text, colKey, ink] = CAPTION[L.kind];
      const pop = age < 0.12 ? lerp(1.45, 1, ease3(age / 0.12)) : 1;
      const alpha = age > 1.2 ? 1 - (age - 1.2) / 0.4 : 1, rise = 22 * dpr * ease3(Math.min(1, age / 1.6));
      setFont(g, 700, cs);
      const w = g.measureText(text).width * 1.1 + cs * 1.8, h = cs * 1.7, px = clamp(x, w / 2 + 4 * dpr, cw - w / 2 - 4 * dpr);
      const hitsTag = yy => tags.some(T => px - w / 2 < T.tx + T.w && T.tx < px + w / 2 && yy - h / 2 < T.ty + T.h && T.ty < yy + h / 2);
      // keep clear of the name tags and, where possible, of the robots
      const boxes = tags.map(T => [T.tx, T.ty, T.tx + T.w, T.ty + T.h]).concat(tags.map(T => T.box));
      const cost = yy => boxes.reduce((s, b, i) => s + (i < 2 ? 4 : 1) * area(px - w / 2, yy - h / 2, px + w / 2, yy + h / 2, b), 0);
      let py = y - cs * 2.4 - rise, best = cost(py);
      for (const yy of [y - cs * 4 - rise, y + cs * 2.4 + rise * 0.3, y + cs * 4 + rise * 0.3]) {
        const c = cost(yy) + 0.05 * w * h; if (c < best) { best = c; py = yy; }
      }
      plate(g, px, clamp(py, h, ch - h), text, th[colKey], ink, cs, pop, alpha);
    }
  }

  // a live fight grows frame by frame: index the new frames without throwing the floor and sprites away
  function grow(doc) { if (S.doc === doc) S.P = prep(doc); }
  window.ArenaRenderer = { reset, draw, grow };
})();
