// Seen or Solved: the five charts on the home page, drawn as SVG from data/pilot.json.
// Each chart redraws on resize; nothing animates or moves with the scroll.
const NS = 'http://www.w3.org/2000/svg';
const TOTAL = 143000;
const C = { accent: '#5ab9fc', ink: '#e9edf3', grey: '#8891a1', amber: '#f0c27a', band: 'rgba(90,185,252,0.12)' };
const NAME = { arc_easy: 'ARC-Easy', sciq: 'SciQ', piqa: 'PIQA', lambada: 'LAMBADA', triviaqa: 'TriviaQA' };

const el = (name, attrs = {}, text) => {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};
const fmtK = (s) => (s >= 1000 ? `${Math.round(s / 1000)}k` : String(s));
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;

function svgFor(box) {
  box.querySelector('svg')?.remove();
  const w = box.clientWidth, h = box.clientHeight;
  const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: w, height: h, focusable: 'false' });
  box.append(svg);
  return { svg, w, h };
}
const line = (svg, pts, color, width = 2, dash) => svg.append(el('polyline', {
  points: pts.map((p) => p.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': width,
  'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...(dash ? { 'stroke-dasharray': dash } : {}),
}));
const logX = (x0, x1) => (s) => x0 + ((Math.log10(s) - Math.log10(500)) / (Math.log10(TOTAL) - Math.log10(500))) * (x1 - x0);

function yAxis(svg, y, ticks, x0, x1) {
  for (const t of ticks) {
    svg.append(el('line', { x1: x0, x2: x1, y1: y(t), y2: y(t), class: 'grid-l' }));
    svg.append(el('text', { x: x0 - 8, y: y(t) + 4, 'text-anchor': 'end' }, pct(t)));
  }
}
function stepAxis(svg, x, y0, ticks) {
  for (const t of ticks) svg.append(el('text', { x: x(t), y: y0 + 18, 'text-anchor': 'middle' }, fmtK(t)));
}

function drawAcc(P) {
  const box = document.getElementById('chart-acc');
  const { svg, w, h } = svgFor(box);
  const m = { l: 40, r: w < 420 ? 64 : 76, t: 6, b: 24 };
  const x = logX(m.l, w - m.r), y = (v) => m.t + (1 - v) * (h - m.t - m.b);
  yAxis(svg, y, [0, 0.5, 1], m.l, w - m.r);
  stepAxis(svg, x, h - m.b, [1000, 10000, 100000]);
  const series = [['sciq', C.ink], ['piqa', C.amber], ['arc_easy', C.grey], ['lambada', C.accent]];
  const ends = [];
  for (const [b, col] of series) {
    const rows = P.accuracy[b];
    line(svg, rows.map((r) => [x(r[0]), y(r[1])]), col);
    for (const r of rows) if (r[2] != null) svg.append(el('circle', { cx: x(r[0]), cy: y(r[2]), r: 2.5, fill: '#0a0d14', stroke: col, 'stroke-width': 1.5 }));
    ends.push([b, col, y(rows[rows.length - 1][1])]);
  }
  ends.sort((a, b) => a[2] - b[2]);
  for (let i = 1; i < ends.length; i++) ends[i][2] = Math.max(ends[i][2], ends[i - 1][2] + 16);
  for (const [b, col, yy] of ends) svg.append(el('text', { x: w - m.r + 8, y: yy + 4, fill: col, style: `fill:${col}` }, NAME[b]));
}

function drawCov(P) {
  const box = document.getElementById('chart-cov');
  const { svg, w, h } = svgFor(box);
  const pick = (b, check) => P.coverage.find((c) => c.benchmark === b && c.check === check);
  const rows = [pick('lambada', '13-word window found'), pick('sciq', '13-word window found'), pick('piqa', '13-word window found'),
    pick('arc_easy', '13-word window found'), pick('triviaqa', 'question found')];
  const m = { l: 84, r: 56, t: 4, b: 24 };
  const x = (v) => m.l + (v / 0.3) * (w - m.l - m.r);
  const band = (h - m.t - m.b) / rows.length;
  for (const t of [0, 0.1, 0.2, 0.3]) {
    svg.append(el('line', { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, class: 'grid-l' }));
    svg.append(el('text', { x: x(t), y: h - m.b + 18, 'text-anchor': 'middle' }, pct(t)));
  }
  rows.forEach((r, i) => {
    const cy = m.t + band * (i + 0.5), bh = Math.min(22, band * 0.55);
    const col = r.benchmark === 'lambada' ? C.accent : C.grey;
    svg.append(el('text', { x: m.l - 10, y: cy + 4, 'text-anchor': 'end', class: 't-ink' }, NAME[r.benchmark]));
    svg.append(el('rect', { x: x(0), y: cy - bh / 2, width: Math.max(2, x(r.share) - x(0)), height: bh, rx: 3, fill: col }));
    svg.append(el('line', { x1: x(r.ci_low), x2: x(r.ci_high), y1: cy, y2: cy, stroke: C.ink, 'stroke-width': 1.5 }));
    svg.append(el('text', { x: Math.max(x(r.share), x(r.ci_high)) + 8, y: cy + 4, class: 't-ink' }, pct(r.share, 1)));
  });
}

function drawDates(P) {
  const box = document.getElementById('chart-dates');
  const { svg, w, h } = svgFor(box);
  const t0 = P.dates.lambada.t0;
  const m = { l: 8, r: 8, t: 22, b: 24 };
  const bins = w < 420 ? 24 : 40, bw = TOTAL / bins, counts = new Array(bins).fill(0);
  for (const t of t0) counts[Math.min(bins - 1, Math.floor(t / bw))]++;
  const max = Math.max(...counts);
  const x = (s) => m.l + (s / TOTAL) * (w - m.l - m.r), y = (c) => h - m.b - (c / max) * (h - m.t - m.b);
  svg.append(el('rect', { x: x(0.1 * TOTAL), y: m.t - 18, width: x(0.9 * TOTAL) - x(0.1 * TOTAL), height: h - m.b - m.t + 18, fill: C.band, rx: 6 }));
  svg.append(el('text', { x: x(0.5 * TOTAL), y: m.t - 4, 'text-anchor': 'middle', class: 't-ink' }, 'Usable: 10% to 90% of training'));
  counts.forEach((c, i) => {
    const mid = (i + 0.5) * bw, inside = mid >= 0.1 * TOTAL && mid <= 0.9 * TOTAL;
    svg.append(el('rect', { x: x(i * bw) + 1, y: y(c), width: Math.max(1, x(bw) - x(0) - 2), height: h - m.b - y(c), rx: 2, fill: inside ? C.accent : C.grey }));
  });
  for (const t of [0, 72000, TOTAL]) svg.append(el('text', { x: x(t), y: h - m.b + 18, 'text-anchor': t === 0 ? 'start' : t === TOTAL ? 'end' : 'middle' }, t === TOTAL ? '143k steps' : fmtK(t)));
}

function drawSeeds(P) {
  const box = document.getElementById('chart-seeds');
  const { svg, w, h } = svgFor(box);
  const ex = P.seeds.examples;
  const m = { l: 8, r: 8, t: 4, b: 24 };
  const x = (s) => m.l + (s / TOTAL) * (w - m.l - m.r), band = (h - m.t - m.b) / ex.length;
  ex.forEach((it, i) => {
    const cy = m.t + band * (i + 0.5);
    const lo = Math.min(...it.t0), hi = Math.max(...it.t0);
    svg.append(el('line', { x1: x(lo), x2: x(hi), y1: cy, y2: cy, stroke: 'rgba(255,255,255,0.14)', 'stroke-width': 2 }));
    it.t0.forEach((t, s) => svg.append(el('circle', { cx: x(t), cy, r: s === 0 ? 4.5 : 3.5, fill: s === 0 ? C.accent : C.grey })));
  });
  for (const t of [0, 72000, TOTAL]) svg.append(el('text', { x: x(t), y: h - m.b + 18, 'text-anchor': t === 0 ? 'start' : t === TOTAL ? 'end' : 'middle' }, t === TOTAL ? '143k steps' : fmtK(t)));
}

function drawGroups(P) {
  const box = document.getElementById('chart-groups');
  const { svg, w, h } = svgFor(box);
  const g = P.groups.lambada;
  const m = { l: 40, r: 12, t: 6, b: 24 };
  const x = logX(m.l, w - m.r), y = (v) => m.t + (1 - v / 0.7) * (h - m.t - m.b);
  yAxis(svg, y, [0, 0.35, 0.7], m.l, w - m.r);
  stepAxis(svg, x, h - m.b, [1000, 10000, 100000]);
  const f = g.confirmed.series, c = g.clean.series;
  const up = f.map((r) => [x(r[0]), y(r[3])]), dn = f.map((r) => [x(r[0]), y(r[2])]).reverse();
  svg.append(el('polygon', { points: [...up, ...dn].map((p) => p.join(',')).join(' '), fill: C.band }));
  line(svg, c.map((r) => [x(r[0]), y(r[1])]), C.grey);
  line(svg, f.map((r) => [x(r[0]), y(r[1])]), C.accent, 2.5);
}

async function main() {
  let P;
  try { P = await (await fetch('data/pilot.json')).json(); } catch { return; }
  const charts = [['chart-acc', drawAcc], ['chart-cov', drawCov], ['chart-dates', drawDates], ['chart-seeds', drawSeeds], ['chart-groups', drawGroups]];
  for (const [id, fn] of charts) {
    const box = document.getElementById(id);
    if (!box) continue;
    const draw = () => { if (box.clientWidth > 0 && box.clientHeight > 0) fn(P); };
    draw();
    new ResizeObserver(draw).observe(box);
  }
}
main();
