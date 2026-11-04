// Seen or Solved? — site behaviour. Everything here is progressive: without it the page
// shows every section, all plan phases, the default timing example and "pending" results.

// Checkpoints and timing filter. Defaults mirror configs/models.yaml and configs/exposure.yaml;
// data/results.json (written by `python -m figures.site_data`) overrides them.
const DEFAULTS = {
  total_steps: 143000,
  checkpoints: [512, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000, 16000,
    20000, 24000, 28000, 33000, 38000, 43000, 49000, 55000, 61000, 68000,
    75000, 82000, 90000, 98000, 106000, 114000, 122000, 130000, 137000, 143000],
  timing_filter: { min_checkpoints_before_t0: 3, t0_min_fraction: 0.1, t0_max_fraction: 0.9 },
};

const fmt = (n) => n.toLocaleString('en-US');

// ---------- reveal once ----------
(() => {
  const targets = document.querySelectorAll('[data-reveal], .chapter, .result');
  const show = (el) => el.classList.add('is-in');
  if (!('IntersectionObserver' in window)) { targets.forEach(show); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { show(e.target); io.unobserve(e.target); }
  }, { rootMargin: '0px 0px -12% 0px', threshold: 0.08 });
  targets.forEach((el) => io.observe(el));
  window.addEventListener('load', () => setTimeout(() => {
    targets.forEach((el) => { const r = el.getBoundingClientRect(); if (r.top < innerHeight && r.bottom > 0) show(el); });
  }, 2500));
})();

// ---------- plan tabs ----------
(() => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  if (!tabs.length) return;
  const select = (tab, focus) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    if (focus) tab.focus();
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t, false));
    t.addEventListener('keydown', (e) => {
      const keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
      if (e.key in keys) { e.preventDefault(); select(tabs[(i + keys[e.key] + tabs.length) % tabs.length], true); }
      if (e.key === 'Home') { e.preventDefault(); select(tabs[0], true); }
      if (e.key === 'End') { e.preventDefault(); select(tabs[tabs.length - 1], true); }
    });
  });
  select(tabs.find((t) => t.getAttribute('aria-selected') === 'true') || tabs[0], false);
})();

// ---------- timing: when was it read? ----------
function timing(t0, cfg) {
  const { checkpoints, total_steps: total, timing_filter: tf } = cfg;
  const before = checkpoints.filter((s) => s < t0).length;
  const after = checkpoints.length - before;
  const frac = t0 / total;
  let verdict = 'Yes';
  if (t0 < tf.t0_min_fraction * total) verdict = 'No: read in the first 10% of training';
  else if (t0 > tf.t0_max_fraction * total) verdict = 'No: read in the last 10% of training';
  else if (before < tf.min_checkpoints_before_t0) verdict = 'No: fewer than three checkpoints before';
  return { before, after, frac, verdict, kept: verdict === 'Yes' };
}

function drawTimeline(svg, t0, cfg) {
  const NS = 'http://www.w3.org/2000/svg';
  const { checkpoints, total_steps: total, timing_filter: tf } = cfg;
  const x = (s) => 12 + (s / total) * 376;
  const el = (name, attrs, text) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text) n.textContent = text;
    return n;
  };
  [...svg.querySelectorAll(':scope > :not(desc)')].forEach((n) => n.remove());
  const r = timing(t0, cfg);
  svg.append(
    el('rect', { x: x(tf.t0_min_fraction * total), y: 70, width: x(tf.t0_max_fraction * total) - x(tf.t0_min_fraction * total), height: 40, fill: '#46531a' }),
    el('rect', { x: 12, y: 87, width: 376, height: 6, fill: '#5d683a' }),
  );
  for (const s of checkpoints) {
    svg.append(el('circle', { cx: x(s), cy: 90, r: 5, fill: s < t0 ? '#ecebe7' : '#9aa27a' }));
  }
  const mx = x(t0);
  svg.append(
    el('rect', { class: 'timeline__marker', x: mx - 2, y: 34, width: 4, height: 50, fill: r.kept ? '#ecebe7' : '#9aa27a' }),
    el('circle', { cx: mx, cy: 34, r: 7, fill: r.kept ? '#ecebe7' : '#9aa27a' }),
    el('text', { x: Math.min(Math.max(mx, 60), 340), y: 18, 'text-anchor': 'middle', 'font-size': 17, fill: '#ecebe7', 'font-family': 'EB Garamond, serif' }, `read at ${fmt(t0)}`),
    el('text', { x: 12, y: 134, 'font-size': 17, fill: '#ecebe7', 'font-family': 'EB Garamond, serif' }, '0'),
    el('text', { x: 388, y: 134, 'text-anchor': 'end', 'font-size': 17, fill: '#ecebe7', 'font-family': 'EB Garamond, serif' }, `${fmt(total)} steps`),
    el('text', { x: 200, y: 134, 'text-anchor': 'middle', 'font-size': 17, fill: '#c9cdb5', 'font-family': 'EB Garamond, serif' }, 'kept if read in the band'),
  );
  return r;
}

function setupTiming(cfg) {
  const svg = document.getElementById('timeline-svg');
  const input = document.getElementById('t0');
  if (!svg || !input) return;
  input.max = String(cfg.total_steps);
  const update = () => {
    const t0 = Number(input.value);
    const r = drawTimeline(svg, t0, cfg);
    document.getElementById('f-step').textContent = fmt(t0);
    document.getElementById('f-frac').textContent = `${Math.round(r.frac * 100)}%`;
    document.getElementById('f-around').textContent = `${r.before} and ${r.after}`;
    document.getElementById('t0-verdict').textContent = r.verdict;
    input.setAttribute('aria-valuetext', `step ${fmt(t0)}, ${r.verdict === 'Yes' ? 'kept' : 'dropped'}`);
  };
  input.addEventListener('input', update);
  update();
}

// ---------- results ----------
function applyResults(data) {
  for (const [key, block] of Object.entries(data.blocks || {})) {
    const status = document.querySelector(`[data-status="${key}"]`);
    if (status && block.status === 'available' && block.summary) {
      status.dataset.state = 'available';
      status.textContent = block.summary;
    }
    const slot = document.querySelector(`[data-figure="${key}"]`);
    if (slot && block.figure) {
      const img = new Image();
      img.src = block.figure;
      img.alt = block.figure_alt || '';
      img.loading = 'lazy';
      img.onload = () => slot.replaceChildren(img);
    }
  }
}

(async () => {
  let cfg = DEFAULTS;
  try {
    if (location.protocol === 'file:') throw new Error('opened from disk');
    const res = await fetch('data/results.json', { cache: 'no-cache' });
    if (res.ok) {
      const data = await res.json();
      cfg = { ...DEFAULTS, ...(data.design || {}) };
      applyResults(data);
    }
  } catch (_) { /* opened from disk or offline: keep the static, pending page */ }
  setupTiming(cfg);
})();
