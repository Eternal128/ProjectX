// Seen or Solved? — site behaviour and charts. Data comes from site/data/*.json, written by
// `python -m evals.pythia_published` (real published results) and `python -m figures.site_methods`
// (simulated method checks). Without script the page still reads; charts need it.

const NS = 'http://www.w3.org/2000/svg';
const COLORS = { '410m': 'var(--series-1)', '1.4b': 'var(--series-2)', '2.8b': 'var(--series-3)' };
const SIZE_LABEL = { '410m': '410M', '1.4b': '1.4B', '2.8b': '2.8B' };
const BENCH_NAME = { lambada: 'LAMBADA', piqa: 'PIQA', sciq: 'SciQ', arc_easy: 'ARC-Easy', mmlu: 'MMLU' };
const DESIGN = {
  total: 143000,
  checkpoints: [512, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000, 16000, 20000, 24000, 28000, 33000, 38000,
    43000, 49000, 55000, 61000, 68000, 75000, 82000, 90000, 98000, 106000, 114000, 122000, 130000, 137000, 143000],
  filter: { min_before: 3, lo: 0.1, hi: 0.9 },
};
const fmt = (n) => n.toLocaleString('en-US');
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const sizeOf = (model) => model.replace('pythia-', '').replace('-deduped', '');

function el(name, attrs = {}, text) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}
const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const symlog = (max, r0, r1) => { const m = Math.log10(1 + max); return (v) => r0 + (Math.log10(1 + v) / m) * (r1 - r0); };
const pathOf = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('');
const stepLabel = (s) => (s >= 1000 ? `${Math.round(s / 1000)}k` : String(s));

function frame(container, height) {
  container.querySelectorAll('svg').forEach((s) => s.remove());
  const w = Math.max(240, container.clientWidth);
  const h = height || container.clientHeight || 240;
  const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, 'aria-hidden': 'true' });
  container.prepend(svg);
  return { svg, w, h };
}

function tooltip(container) {
  let t = container.querySelector('.tooltip');
  if (!t) { t = document.createElement('div'); t.className = 'tooltip'; container.append(t); }
  return {
    show(html, x, y) {
      t.innerHTML = html;
      const cw = container.clientWidth;
      const tw = t.offsetWidth || 170;
      t.style.left = `${Math.min(Math.max(x + 14, 0), cw - tw)}px`;
      t.style.top = `${Math.max(y - 20, 0)}px`;
      t.classList.add('is-on');
    },
    hide() { t.classList.remove('is-on'); },
  };
}

function srTable(container, head, rows, caption) {
  container.querySelector('.sr-table')?.remove();
  const t = document.createElement('div');
  t.className = 'visually-hidden sr-table';
  t.innerHTML = `<table><caption>${caption}</caption><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  container.append(t);
}

function gridY(svg, y, ticks, x0, x1, fmtT) {
  for (const v of ticks) {
    svg.append(el('line', { x1: x0, x2: x1, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('text', { x: x0 - 8, y: y(v) + 4, 'text-anchor': 'end' }, fmtT(v)));
  }
}

function drawLine(svg, pts, color, extra = {}) {
  const p = el('path', { d: pathOf(pts), fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...extra });
  svg.append(p);
  if (extra.animate !== false && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const len = p.getTotalLength();
    p.style.setProperty('--len', len);
    p.classList.add('line-draw');
  }
  return p;
}

// ---------- hero: when did Pythia read the question? ----------
function timing(t0) {
  const { checkpoints, total, filter } = DESIGN;
  const before = checkpoints.filter((s) => s < t0).length;
  const frac = t0 / total;
  let note = 'Kept: read between 10% and 90% of training, with at least three checkpoints before it.';
  let kept = true;
  if (frac < filter.lo) { kept = false; note = 'Dropped: read in the first 10% of training, too early to see a baseline.'; }
  else if (frac > filter.hi) { kept = false; note = 'Dropped: read in the last 10% of training, too late to see the effect.'; }
  else if (before < filter.min_before) { kept = false; note = 'Dropped: fewer than three checkpoints before it.'; }
  return { before, after: checkpoints.length - before, frac, kept, note };
}

function drawProbe(t0) {
  const box = document.getElementById('probe-chart');
  if (!box) return;
  const { svg, w, h } = frame(box);
  const x = linear(0, DESIGN.total, 6, w - 6);
  const r = timing(t0);
  svg.append(el('rect', { x: x(DESIGN.filter.lo * DESIGN.total), y: 30, width: x(DESIGN.filter.hi * DESIGN.total) - x(DESIGN.filter.lo * DESIGN.total), height: 66, rx: 6, fill: '#e9f8ef' }));
  svg.append(el('text', { x: x(0.5 * DESIGN.total), y: 48, 'text-anchor': 'middle' }, 'timing window, 10% to 90%'));
  svg.append(el('line', { x1: 6, x2: w - 6, y1: 78, y2: 78, stroke: 'var(--grid)', 'stroke-width': 2 }));
  for (const s of DESIGN.checkpoints) {
    const seen = s < t0;
    svg.append(el('circle', { cx: x(s), cy: 78, r: 4.5, fill: seen ? 'var(--ink)' : 'var(--bg)', stroke: seen ? 'var(--bg)' : 'var(--axis)', 'stroke-width': seen ? 1.5 : 1.5 }));
  }
  const mx = x(t0);
  svg.append(el('line', { x1: mx, x2: mx, y1: 18, y2: 104, stroke: r.kept ? '#0ca30c' : '#d03b3b', 'stroke-width': 2 }));
  svg.append(el('text', { x: 6, y: h - 8 }, '0'));
  svg.append(el('text', { x: x(DESIGN.total / 2), y: h - 8, 'text-anchor': 'middle' }, '71.5k'));
  svg.append(el('text', { x: w - 6, y: h - 8, 'text-anchor': 'end' }, '143k steps'));
  svg.append(el('text', { x: 6, y: 124, class: 't-ink' }, `● ${r.before} evaluated before`));
  svg.append(el('text', { x: w - 6, y: 124, 'text-anchor': 'end' }, `○ ${r.after} after`));
  return r;
}

function setupProbe() {
  const input = document.getElementById('t0');
  if (!input) return;
  const update = () => {
    const t0 = Number(input.value);
    const r = drawProbe(t0);
    document.getElementById('p-step-head').textContent = fmt(t0);
    document.getElementById('p-frac').textContent = pct(r.frac, 0);
    document.getElementById('p-around').textContent = `${r.before} · ${r.after}`;
    document.getElementById('p-batch').textContent = t0 > 0 ? `batch ${fmt(t0 - 1)}` : 'before training';
    const v = document.getElementById('p-verdict');
    v.textContent = r.kept ? 'Kept' : 'Dropped';
    v.dataset.kept = String(r.kept);
    document.getElementById('p-note').textContent = r.note;
    input.setAttribute('aria-valuetext', `step ${fmt(t0)}, ${r.kept ? 'kept' : 'dropped'}`);
  };
  input.addEventListener('input', update);
  new ResizeObserver(() => drawProbe(Number(input.value))).observe(document.getElementById('probe-chart'));
  update();
}

// ---------- accuracy over training ----------
function curveState() {
  const f = document.getElementById('curve-controls');
  const d = new FormData(f);
  return { bench: d.get('bench'), data: d.get('data'), chance: d.get('chance') === 'on', band: d.get('band') === 'on' };
}

function drawCurves(B, opts = {}) {
  const box = document.getElementById('curve-chart');
  const st = curveState();
  const { svg, w, h } = frame(box);
  const m = { l: 44, r: 64, t: 12, b: 34 };
  const models = B.models.filter((mm) => mm.data_version === st.data).map((mm) => mm.name);
  const chance = B.benchmarks.find((b) => b.key === st.bench).chance;
  const all = models.flatMap((mm) => B.series[mm][st.bench].map((p) => p[1]));
  const lo = Math.max(0, Math.floor((Math.min(...all, chance) - 0.04) * 10) / 10);
  const hi = Math.min(1, Math.ceil((Math.max(...all) + 0.04) * 10) / 10);
  const x = symlog(DESIGN.total, m.l, w - m.r);
  const y = linear(lo, hi, h - m.b, m.t);
  const ticks = [];
  for (let v = lo; v <= hi + 1e-9; v += hi - lo > 0.5 ? 0.2 : 0.1) ticks.push(+v.toFixed(2));
  gridY(svg, y, ticks, m.l, w - m.r, (v) => `${Math.round(v * 100)}%`);
  for (const s of [0, 10, 100, 1000, 10000, 143000]) {
    svg.append(el('text', { x: x(s), y: h - 10, 'text-anchor': s === 143000 ? 'end' : 'middle' }, s === 143000 ? '143k' : stepLabel(s)));
  }
  svg.append(el('text', { x: w - m.r, y: h - 10 + 0, 'text-anchor': 'start', dx: 6 }, 'step'));
  if (st.chance) {
    svg.append(el('line', { x1: m.l, x2: w - m.r, y1: y(chance), y2: y(chance), stroke: 'var(--axis)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }));
    svg.append(el('text', { x: w - m.r + 6, y: y(chance) + 4 }, `chance ${Math.round(chance * 100)}%`));
  }
  const ends = [];
  for (const mm of models) {
    const size = sizeOf(mm);
    const pts = B.series[mm][st.bench];
    if (st.band) {
      const up = pts.map((p) => [x(p[0]), y(Math.min(1, p[1] + 2 * p[2]))]);
      const dn = pts.map((p) => [x(p[0]), y(Math.max(0, p[1] - 2 * p[2]))]).reverse();
      svg.append(el('path', { d: `${pathOf(up)}L${dn.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join('L')}Z`, fill: COLORS[size], 'fill-opacity': 0.14 }));
    }
    drawLine(svg, pts.map((p) => [x(p[0]), y(p[1])]), COLORS[size], { animate: opts.animate });
    const last = pts[pts.length - 1];
    ends.push({ size, y: y(last[1]), v: last[1] });
  }
  // direct end labels, nudged apart
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
  for (const e of ends) {
    svg.append(el('circle', { cx: x(DESIGN.total), cy: y(e.v), r: 4, fill: COLORS[e.size], stroke: 'var(--bg)', 'stroke-width': 2 }));
    svg.append(el('text', { x: w - m.r + 8, y: e.y + 4, class: 't-ink' }, `${SIZE_LABEL[e.size]} ${pct(e.v, 0)}`));
  }
  // hover crosshair
  const steps = B.steps;
  const hair = el('line', { y1: m.t, y2: h - m.b, stroke: 'var(--ink)', 'stroke-width': 1, opacity: 0 });
  svg.append(hair);
  const dots = models.map((mm) => { const c = el('circle', { r: 5, fill: COLORS[sizeOf(mm)], stroke: 'var(--bg)', 'stroke-width': 2, opacity: 0 }); svg.append(c); return c; });
  const hit = el('rect', { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, fill: 'transparent', style: 'cursor:crosshair' });
  svg.append(hit);
  const tip = tooltip(box);
  const move = (evt) => {
    const rect = svg.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * w;
    let best = 0;
    steps.forEach((s, i) => { if (Math.abs(x(s) - px) < Math.abs(x(steps[best]) - px)) best = i; });
    const s = steps[best];
    hair.setAttribute('x1', x(s)); hair.setAttribute('x2', x(s)); hair.setAttribute('opacity', 0.35);
    let rows = '';
    models.forEach((mm, i) => {
      const p = B.series[mm][st.bench][best];
      dots[i].setAttribute('cx', x(s)); dots[i].setAttribute('cy', y(p[1])); dots[i].setAttribute('opacity', 1);
      rows += `<div class="tt-row"><span class="tt-dot" style="background:${COLORS[sizeOf(mm)]}"></span><span>Pythia ${SIZE_LABEL[sizeOf(mm)]}</span><span>${pct(p[1])} ± ${pct(2 * p[2])}</span></div>`;
    });
    tip.show(`<div class="tt-h">Step ${fmt(s)}</div>${rows}`, (x(s) / w) * rect.width, ((evt.clientY - rect.top)));
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerleave', () => { tip.hide(); hair.setAttribute('opacity', 0); dots.forEach((d) => d.setAttribute('opacity', 0)); });

  document.getElementById('curve-title').textContent = `${BENCH_NAME[st.bench]} accuracy, ${st.data === 'standard' ? 'standard' : 'deduplicated'} Pile`;
  document.getElementById('curve-legend').innerHTML = models.map((mm) => `<span><i style="background:${COLORS[sizeOf(mm)]}"></i>Pythia ${SIZE_LABEL[sizeOf(mm)]}</span>`).join('') + (st.chance ? '<span><i class="dash"></i>Chance</span>' : '');
  const tv = document.getElementById('curve-table');
  if (!tv.closest('details').open) { tv.innerHTML = ''; return; }
  const head = ['Step', ...models.map((mm) => `Pythia ${SIZE_LABEL[sizeOf(mm)]}`)];
  const rows = steps.map((s, i) => [fmt(s), ...models.map((mm) => { const p = B.series[mm][st.bench][i]; return `${pct(p[1])} ± ${pct(p[2])}`; })]);
  tv.innerHTML = `<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

// ---------- event study stage (simulated) ----------
function drawStage(M, state, animate = true) {
  const box = document.getElementById('stage-chart');
  if (!box) return;
  const { svg, w, h } = frame(box);
  const es = M.event_study;
  const m = { l: 44, r: 16, t: 12, b: 34 };
  const ks = es.effects.map((e) => e[0]);
  const x = linear(ks[0], ks[ks.length - 1], m.l, w - m.r);
  const y = linear(-0.5, 2.0, h - m.b, m.t);
  gridY(svg, y, [-0.5, 0, 0.5, 1, 1.5, 2], m.l, w - m.r, (v) => v.toFixed(1));
  for (const k of ks) if (k % 2 === 0 || k === -1) svg.append(el('text', { x: x(k), y: h - 12, 'text-anchor': 'middle' }, k > 0 ? `+${k}` : String(k)));
  svg.append(el('line', { x1: x(-0.5), x2: x(-0.5), y1: m.t, y2: h - m.b, stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-dasharray': '3 4' }));
  svg.append(el('text', { x: x(-0.5) + 6, y: m.t + 12, class: 't-ink' }, 'read at t0'));
  svg.append(el('line', { x1: m.l, x2: w - m.r, y1: y(0), y2: y(0), stroke: 'var(--axis)', 'stroke-width': 1 }));
  const legend = document.getElementById('stage-legend');
  const g = es.groups;
  if (state === 0) {
    drawLine(svg, g.map((r) => [x(r[0]), y(r[1])]), 'var(--series-1)', { animate });
    drawLine(svg, g.map((r) => [x(r[0]), y(r[2])]), 'var(--series-2)', { animate });
    svg.append(el('text', { x: w - m.r, y: y(g[g.length - 1][1]) - 10, 'text-anchor': 'end', class: 't-ink' }, 'seen'));
    svg.append(el('text', { x: w - m.r, y: y(g[g.length - 1][2]) - 10, 'text-anchor': 'end', class: 't-ink' }, 'unseen twins'));
    legend.innerHTML = '<span><i style="background:var(--series-1)"></i>Seen</span><span><i style="background:var(--series-2)"></i>Unseen twins</span>';
  } else {
    if (state === 1) {
      drawLine(svg, g.map((r) => [x(r[0]), y(r[1])]), 'var(--series-1)', { animate: false, 'stroke-opacity': 0.25 });
      drawLine(svg, g.map((r) => [x(r[0]), y(r[2])]), 'var(--series-2)', { animate: false, 'stroke-opacity': 0.25 });
    }
    if (state === 2) {
      const up = es.effects.map((e) => [x(e[0]), y(e[3])]);
      const dn = es.effects.map((e) => [x(e[0]), y(e[2])]).reverse();
      svg.append(el('path', { d: `${pathOf(up)}L${dn.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join('L')}Z`, fill: 'var(--ink)', 'fill-opacity': 0.12 }));
      svg.append(el('line', { x1: m.l, x2: w - m.r, y1: y(es.planted_jump), y2: y(es.planted_jump), stroke: '#0ca30c', 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }));
      svg.append(el('text', { x: m.l + 6, y: y(es.planted_jump) - 8 }, `planted ${es.planted_jump}`));
    }
    drawLine(svg, es.effects.map((e) => [x(e[0]), y(e[1])]), 'var(--ink)', { animate });
    for (const e of es.effects) svg.append(el('circle', { cx: x(e[0]), cy: y(e[1]), r: 4, fill: 'var(--ink)', stroke: 'var(--bg)', 'stroke-width': 2 }));
    legend.innerHTML = state === 2 ? '<span><i style="background:var(--ink)"></i>Seen minus twins, 95% interval</span>' : '<span><i style="background:var(--ink)"></i>Seen minus twins</span>';
  }
  srTable(box, ['Checkpoints from t0', 'Seen', 'Unseen twins', 'Difference', '95% low', '95% high'],
    es.effects.map((e, i) => [e[0], g[i][1], g[i][2], e[1], e[2], e[3]]), 'Simulated event study');
}

function setupStage(M) {
  const steps = [...document.querySelectorAll('.how__step')];
  let current = 0;
  drawStage(M, 0, false);
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const s = Number(e.target.dataset.state);
      if (s === current) continue;
      current = s;
      steps.forEach((x) => x.classList.toggle('is-active', x === e.target));
      drawStage(M, s);
    }
  }, { rootMargin: '-45% 0px -45% 0px' });
  steps.forEach((s) => io.observe(s));
  new ResizeObserver(() => drawStage(M, current, false)).observe(document.getElementById('stage-chart'));
}

// ---------- proofs ----------
function drawChance(B) {
  const box = document.getElementById('chance-chart');
  const { svg, w, h } = frame(box);
  const m = { l: 74, r: 12, t: 8, b: 28 };
  const order = ['sciq', 'piqa', 'lambada', 'arc_easy', 'mmlu'];
  const x = linear(0, 1, m.l, w - m.r);
  const rowH = (h - m.t - m.b) / order.length;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el('line', { x1: x(v), x2: x(v), y1: m.t, y2: h - m.b, stroke: 'var(--grid)', 'stroke-width': 1 }));
    svg.append(el('text', { x: x(v), y: h - 8, 'text-anchor': 'middle' }, `${v * 100}%`));
  }
  const tip = tooltip(box);
  order.forEach((k, i) => {
    const cy = m.t + rowH * (i + 0.5);
    const b = B.benchmarks.find((bb) => bb.key === k);
    svg.append(el('text', { x: m.l - 10, y: cy + 4, 'text-anchor': 'end', class: 't-ink' }, b.name));
    svg.append(el('rect', { x: x(b.chance), y: cy - 10, width: x(b.chance + B.rule.min_margin) - x(b.chance), height: 20, fill: '#fde4dc' }));
    svg.append(el('line', { x1: x(b.chance), x2: x(b.chance), y1: cy - 12, y2: cy + 12, stroke: 'var(--ink)', 'stroke-width': 2 }));
    for (const f of B.final.filter((r) => r.benchmark === k)) {
      const size = sizeOf(f.model);
      const dedup = f.model.endsWith('deduped');
      const c = el('circle', { cx: x(f.acc), cy: cy + (dedup ? 5 : -5), r: 4.5, fill: dedup ? 'var(--bg)' : COLORS[size], stroke: COLORS[size], 'stroke-width': 2 });
      c.addEventListener('pointerenter', () => tip.show(`<div class="tt-h">${b.name}, Pythia ${SIZE_LABEL[size]}${dedup ? ' deduped' : ''}</div>${pct(f.acc)} vs chance ${pct(b.chance, 0)}<br>${f.passes ? 'Passes' : 'Fails'} the rule`, Number(c.getAttribute('cx')), cy));
      c.addEventListener('pointerleave', () => tip.hide());
      svg.append(c);
    }
  });
  srTable(box, ['Model', 'Benchmark', 'Accuracy', 'Chance', 'Passes'], B.final.map((f) => [f.model, BENCH_NAME[f.benchmark], pct(f.acc), pct(f.chance, 0), f.passes ? 'yes' : 'no']), 'Final-checkpoint accuracy against chance');
}

function drawSchedule(B) {
  const box = document.getElementById('sched-chart');
  const { svg, w, h } = frame(box);
  const m = { l: 92, r: 14, t: 10, b: 26 };
  const x = symlog(DESIGN.total, m.l, w - m.r);
  const saved = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  for (let s = 1000; s <= DESIGN.total; s += 1000) saved.push(s);
  const rows = [
    ['Saved, 154', saved, 'var(--axis)', 2.2],
    ['Published, 27', B.steps, 'var(--series-2)', 4],
    ['Evaluated, 30', DESIGN.checkpoints, 'var(--series-1)', 4],
  ];
  const rowH = (h - m.t - m.b) / rows.length;
  rows.forEach(([label, pts, color, r], i) => {
    const cy = m.t + rowH * (i + 0.5);
    svg.append(el('text', { x: m.l - 10, y: cy + 4, 'text-anchor': 'end', class: 't-ink' }, label));
    for (const s of pts) svg.append(el('circle', { cx: x(s), cy, r, fill: color, stroke: r > 3 ? 'var(--surface)' : 'none', 'stroke-width': 1.5 }));
  });
  for (const s of [0, 10, 100, 1000, 10000, 143000]) svg.append(el('text', { x: x(s), y: h - 6, 'text-anchor': s === 143000 ? 'end' : 'middle' }, s === 143000 ? '143k' : stepLabel(s)));
  srTable(box, ['Set', 'Steps'], rows.map(([l, p]) => [l, p.map(fmt).join(', ')]), 'Checkpoint schedule');
}

function drawIrt(M) {
  const box = document.getElementById('irt-chart');
  const { svg, w, h } = frame(box);
  const pts = M.irt.difficulty;
  const m = { l: 40, r: 12, t: 12, b: 34 };
  const lo = Math.floor(Math.min(...pts.flat())), hi = Math.ceil(Math.max(...pts.flat()));
  const side = Math.min(w - m.l - m.r, h - m.t - m.b);
  const x = linear(lo, hi, m.l, m.l + side);
  const y = linear(lo, hi, m.t + side, m.t);
  for (let v = lo; v <= hi; v++) {
    svg.append(el('line', { x1: x(lo), x2: x(hi), y1: y(v), y2: y(v), stroke: 'var(--grid)' }));
    svg.append(el('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, String(v)));
    svg.append(el('text', { x: x(v), y: m.t + side + 18, 'text-anchor': 'middle' }, String(v)));
  }
  svg.append(el('line', { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }));
  for (const [t, e] of pts) svg.append(el('circle', { cx: x(t), cy: y(e), r: 4, fill: 'var(--series-1)', 'fill-opacity': 0.75, stroke: 'var(--surface)', 'stroke-width': 1.5 }));
  const lx = m.l + side + 12;
  if (w - lx > 90) {
    svg.append(el('text', { x: lx, y: m.t + 14, class: 't-ink' }, 'True difficulty →'));
    svg.append(el('text', { x: lx, y: m.t + 32 }, 'across'));
    svg.append(el('text', { x: lx, y: m.t + 58, class: 't-ink' }, 'Estimated ↑'));
    svg.append(el('text', { x: lx, y: m.t + 76 }, 'up the side'));
    svg.append(el('text', { x: lx, y: m.t + 102 }, 'Dashed: perfect'));
    svg.append(el('text', { x: lx, y: m.t + 120 }, 'recovery'));
  }
  srTable(box, ['True difficulty', 'Estimated'], pts, 'Simulated IRT recovery');
  document.getElementById('irt-delta').textContent = M.irt.recovered_delta_0.toFixed(3);
  document.getElementById('irt-cb').textContent = M.irt.corr_difficulty.toFixed(2);
  document.getElementById('irt-ca').textContent = M.irt.corr_ability.toFixed(2);
}

function drawPretrend(M) {
  const box = document.getElementById('pre-chart');
  const { svg, w, h } = frame(box);
  const es = M.event_study_pretrend;
  const m = { l: 40, r: 12, t: 12, b: 28 };
  const ks = es.effects.map((e) => e[0]);
  const vals = es.effects.flatMap((e) => [e[2], e[3]]);
  const lo = Math.floor(Math.min(...vals) * 2) / 2, hi = Math.ceil(Math.max(...vals) * 2) / 2;
  const x = linear(ks[0], ks[ks.length - 1], m.l, w - m.r);
  const y = linear(lo, hi, h - m.b, m.t);
  svg.append(el('rect', { x: x(ks[0]) - 4, y: m.t, width: x(-1.5) - x(ks[0]) + 4, height: h - m.t - m.b, fill: '#fde4dc' }));
  svg.append(el('text', { x: x(ks[0]) + 2, y: m.t + 14, class: 't-ink' }, 'pre-trend'));
  const ticks = []; for (let v = lo; v <= hi + 1e-9; v += 0.5) ticks.push(v);
  gridY(svg, y, ticks, m.l, w - m.r, (v) => v.toFixed(1));
  for (const k of ks) if (k % 2 === 0) svg.append(el('text', { x: x(k), y: h - 8, 'text-anchor': 'middle' }, k > 0 ? `+${k}` : String(k)));
  const up = es.effects.map((e) => [x(e[0]), y(e[3])]);
  const dn = es.effects.map((e) => [x(e[0]), y(e[2])]).reverse();
  svg.append(el('path', { d: `${pathOf(up)}L${dn.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join('L')}Z`, fill: 'var(--ink)', 'fill-opacity': 0.12 }));
  drawLine(svg, es.effects.map((e) => [x(e[0]), y(e[1])]), 'var(--ink)', { animate: false });
  for (const e of es.effects) svg.append(el('circle', { cx: x(e[0]), cy: y(e[1]), r: 4, fill: e[0] < -1 ? '#d03b3b' : 'var(--ink)', stroke: 'var(--surface)', 'stroke-width': 2 }));
  srTable(box, ['Checkpoints from t0', 'Effect', '95% low', '95% high'], es.effects.map((e) => [e[0], e[1], e[2], e[3]]), 'Simulated event study with a pre-trend');
  document.getElementById('pre-val').textContent = es.pretrend.toFixed(2).replace('-', '−');
  document.getElementById('pre-ci').textContent = `${es.pretrend_ci[0].toFixed(2)} to ${es.pretrend_ci[1].toFixed(2)}`.replaceAll('-', '−');
}

// ---------- proposal index ----------
function setupPaper() {
  const track = document.querySelector('.paper__track');
  const sheets = [...document.querySelectorAll('.sheet')];
  const items = [...document.querySelectorAll('#paper-index li')];
  if (!track || !sheets.length) return;
  const n = sheets.length;
  let cur = -1;
  const set = (i) => {
    if (i === cur) return;
    cur = i;
    sheets.forEach((s, j) => { s.classList.toggle('is-current', j === i); s.classList.toggle('is-past', j < i); });
    items.forEach((li, j) => li.classList.toggle('is-current', j === i));
  };
  const onScroll = () => {
    if (getComputedStyle(document.querySelector('.paper__stage')).position !== 'sticky') return;
    const r = track.getBoundingClientRect();
    const p = Math.min(Math.max(-r.top / (r.height - innerHeight), 0), 0.999);
    set(Math.floor(p * n));
  };
  addEventListener('scroll', () => requestAnimationFrame(onScroll), { passive: true });
  items.forEach((li, j) => li.querySelector('button').addEventListener('click', () => {
    const top = track.getBoundingClientRect().top + scrollY + ((j + 0.5) / n) * (track.offsetHeight - innerHeight);
    scrollTo({ top, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }));
  set(0);
}

// ---------- reveal, clock, results ----------
(() => {
  const targets = document.querySelectorAll('[data-reveal]');
  const show = (e) => e.classList.add('is-in');
  if (!('IntersectionObserver' in window)) { targets.forEach(show); return; }
  const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) { show(e.target); io.unobserve(e.target); } }, { rootMargin: '0px 0px -10% 0px', threshold: 0.05 });
  targets.forEach((t) => io.observe(t));
  addEventListener('load', () => setTimeout(() => targets.forEach((t) => { const r = t.getBoundingClientRect(); if (r.top < innerHeight && r.bottom > 0) show(t); }), 2500));
})();

(() => {
  const out = document.getElementById('toronto-time');
  if (!out) return;
  const f = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Toronto', hour: '2-digit', minute: '2-digit', hour12: false });
  const tick = () => { out.textContent = `${f.format(new Date())} local time`; };
  tick(); setInterval(tick, 30000);
})();

async function getJSON(path) {
  if (location.protocol === 'file:') return null;
  try { const r = await fetch(path, { cache: 'no-cache' }); return r.ok ? r.json() : null; } catch { return null; }
}

(async () => {
  setupProbe();
  setupPaper();
  const [B, M, R] = await Promise.all([getJSON('data/benchmarks.json'), getJSON('data/methods.json'), getJSON('data/results.json')]);
  if (B) {
    drawCurves(B, { animate: false });
    document.getElementById('curve-controls').addEventListener('change', () => drawCurves(B));
    document.getElementById('curve-table').closest('details').addEventListener('toggle', () => drawCurves(B, { animate: false }));
    new ResizeObserver(() => drawCurves(B, { animate: false })).observe(document.getElementById('curve-chart'));
    drawChance(B);
    drawSchedule(B);
    new ResizeObserver(() => { drawChance(B); drawSchedule(B); }).observe(document.getElementById('sched-chart'));
  }
  if (M) {
    setupStage(M);
    drawIrt(M);
    drawPretrend(M);
    new ResizeObserver(() => { drawIrt(M); drawPretrend(M); }).observe(document.getElementById('irt-chart'));
  }
  if (R) {
    for (const [key, block] of Object.entries(R.blocks || {})) {
      if (block.status !== 'available') continue;
      const v = document.querySelector(`.row__val[data-status="${key}"]`);
      if (v) { v.textContent = 'Result in'; v.dataset.state = 'available'; v.previousElementSibling.previousElementSibling.className = 'row__dot row__dot--done'; }
    }
  }
})();
