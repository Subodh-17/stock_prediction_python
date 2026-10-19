/* Workstation UI. No framework, no CDN (CSP: script-src 'self'). All dynamic text goes through
   textContent / createTextNode -- never innerHTML -- so upstream strings cannot inject markup. */
(() => {
'use strict';
// Element.append() stringifies arrays ("[object HTMLSpanElement]"): make every insertion flatten arrays, drop null/false and
// turn plain values into text nodes, so a call site can never render an object by accident.
for (const m of ['append', 'prepend', 'replaceChildren', 'after', 'before']) {
  const orig = Element.prototype[m];
  Element.prototype[m] = function (...kids) { return orig.apply(this, kids.flat(9).filter(k => k != null && k !== false).map(k => (k && k.nodeType) ? k : document.createTextNode(String(k)))); };
}
const $ = (s, r = document) => r.querySelector(s);
const CSRF = () => ($('meta[name="csrf-token"]') || {}).content || '';
const view = $('#view');

/* ---------------------------------------------------------------- DOM + format helpers */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'text') e.textContent = v; else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(9)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const nf = (v, d = 2) => v == null || !isFinite(v) ? '—' : Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const money = (v, cur) => v == null ? '—' : (cur ? cur + ' ' : '') + nf(v, Math.abs(v) < 1 ? 4 : 2);
const pct = (v, d = 2) => v == null || !isFinite(v) ? '—' : (v > 0 ? '+' : '') + nf(v, d) + '%';
const dir = v => v > 0 ? ['up', '▲'] : v < 0 ? ['dn', '▼'] : ['mu', '■'];
const big = v => v == null ? '—' : Math.abs(v) >= 1e9 ? nf(v / 1e9, 2) + 'B' : Math.abs(v) >= 1e6 ? nf(v / 1e6, 2) + 'M' : Math.abs(v) >= 1e3 ? nf(v / 1e3, 1) + 'K' : nf(v, 0);
const tsLocal = iso => { if (!iso) return '—'; const d = new Date(iso); return isNaN(d) ? '—' : d.toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'medium'}); };
const age = s => s == null ? '' : s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)}m` : s < 172800 ? `${(s / 3600).toFixed(1)}h` : `${Math.round(s / 86400)}d`;
const safeUrl = u => { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.href : null; } catch { return null; } };
const store = { get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } } };

function getTheme() {
  try { return localStorage.getItem('ws.theme') || 'auto'; } catch { return 'auto'; }
}
function setTheme(t) {
  try { localStorage.setItem('ws.theme', t); } catch { /* private mode */ }
  applyTheme(t);
}
function applyTheme(t) {
  const effective = (t === 'auto' || !t) ? 'auto' : t;
  document.documentElement.setAttribute('data-theme', effective);
  const sel = $('#theme-sel');
  if (sel && sel.value !== t) sel.value = t;
}

async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': CSRF() }, body: JSON.stringify(body) };
  let r; try { r = await fetch('/api/v2' + path, opt); } catch (e) { throw new Error('Network error: the server is unreachable.'); }
  let j = null; try { j = await r.json(); } catch { /* non-JSON */ }
  if (!r.ok) { const e = new Error((j && j.message) || `Request failed (${r.status})`); e.status = r.status; e.code = j && j.error; throw e; }
  return j;
}

/* ---------------------------------------------------------------- state */
const prefs = Object.assign({ period: '1y', refresh: 0, fperiod: '2y', epochs: 15 }, store.get('ws.prefs', {}));
const S = { symbol: store.get('ws.symbol', ''), quote: null, hist: null, fkey: null, forecast: null, timers: {}, retrieved: 0, nextRefresh: 0 };
const watch = () => store.get('ws.watch', []);

/* ---------------------------------------------------------------- freshness badge (never colour-only) */
const FRESH = { live: ['●', 'Live'], delayed: ['◐', 'Delayed'], end_of_day: ['◼', 'End of day'], historical: ['▤', 'Historical'],
  stale: ['▲', 'Stale'], unverified: ['?', 'Freshness unverified'], unavailable: ['✕', 'Unavailable'] };
function badge(q) {
  const f = q.freshness || 'unavailable', [g, label] = FRESH[f] || FRESH.unavailable;
  let t = label; if (f === 'delayed' && q.delay_seconds) t += ` ~${Math.round(q.delay_seconds / 60)}m`;
  const el = h('span', { class: `badge b-${f}`, title: q.age_seconds != null ? `Source timestamp is ${age(q.age_seconds)} old` : 'No source timestamp' }, g, ' ', t);
  return h('span', { class: 'badges' }, el, q.from_cache ? [' ', h('span', { class: 'badge b-cached', title: `Served from cache, ${age(q.cache_age_seconds)} old` }, 'Cached')] : '');
}
function provenance(q) {
  if (!q) return '';
  return `Source: ${q.provider || '—'} · exchange ${q.exchange || '—'} · source time ${tsLocal(q.source_ts)} · retrieved ${tsLocal(q.retrieved_ts)}` +
    (q.downgraded_from ? ` · FALLBACK from ${q.downgraded_from}` : '');
}

/* ---------------------------------------------------------------- accessible table equivalent for charts */
function dataTable(box, caption, headers, rows) {
  const wrap = h('div', { class: 'dt-scroll', tabindex: '0', role: 'region', 'aria-label': `${caption} (scrollable table)` },
    h('table', {}, h('caption', { class: 'sr', text: caption }), h('thead', {}, h('tr', {}, headers.map(x => h('th', { scope: 'col', text: x })))),
      h('tbody', {}, rows.map(r => h('tr', {}, r.map((c, i) => h(i === 0 ? 'th' : 'td', i === 0 ? { scope: 'row', text: c } : { class: 'num', text: c })))))));
  box.append(h('details', { class: 'dt' }, h('summary', { text: `Data table: ${caption} (${rows.length} rows)` }), wrap));
}

/* ---------------------------------------------------------------- charts (canvas) */
function mount(box, height) {
  box.replaceChildren();
  const c = h('canvas', { role: 'img', tabindex: '0' }), tip = h('div', { class: 'tip' });
  box.append(c, tip); box.classList.add('chart');
  const fit = () => { const dpr = window.devicePixelRatio || 1, w = box.clientWidth || 600; c.width = w * dpr; c.height = height * dpr; c.style.height = height + 'px';
    const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); return [x, w, height]; };
  return { c, tip, fit };
}
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), r = step0 / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}
function axes(x, w, ht, L, R, T, B, lo, hi, labels, yf) {
  x.strokeStyle = css('--line'); x.fillStyle = css('--mut'); x.font = '11px ' + css('--num'); x.lineWidth = 1; x.textAlign = 'right';
  const y = v => T + (hi - v) / (hi - lo) * (ht - T - B);
  for (const t of niceTicks(lo, hi)) { const yy = Math.round(y(t)) + .5; x.beginPath(); x.moveTo(L, yy); x.lineTo(w - R, yy); x.stroke(); x.fillText(yf(t), L - 6, yy + 4); }
  const n = labels.length, k = Math.min(6, n);
  for (let i = 0; i < k; i++) { const idx = k === 1 ? 0 : Math.round(i * (n - 1) / (k - 1)); x.textAlign = i === 0 ? 'left' : i === k - 1 ? 'right' : 'center';   // edge labels stay inside the canvas
    x.fillText(String(labels[idx]).slice(0, 10), L + (w - L - R) * (n === 1 ? .5 : idx / (n - 1)), ht - 6); }
  return y;
}
function hover(m, labels, textFor, xOf) {
  const move = ev => { const r = m.c.getBoundingClientRect(), px = ev.clientX - r.left, i = xOf(px);
    if (i == null) { m.tip.style.display = 'none'; return; }
    m.tip.textContent = textFor(i); m.tip.style.display = 'block';
    m.tip.style.left = Math.min(px + 12, r.width - 170) + 'px'; m.tip.style.top = '8px'; };
  m.c.onmousemove = move; m.c.onmouseleave = () => { m.tip.style.display = 'none'; };
}
function lineChart(box, o) {   // o: {labels, series:[{name,color,data,dash,width,area}], yFmt, markers:[{i,y,glyph,color,title}], height}
  const m = mount(box, o.height || 260); m.c.setAttribute('aria-label', o.aria || 'Line chart');
  const yf = o.yFmt || (v => nf(v, 2)); let L = 62; const R = 12, T = 10, B = 24, n = o.labels.length;
  const draw = () => {
    const [x, w, ht] = m.fit(); x.clearRect(0, 0, w, ht);
    let lo = Infinity, hi = -Infinity; for (const s of o.series) for (const v of s.data) if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!isFinite(lo)) return; const pad = (hi - lo) * .06 || 1; lo -= pad; hi += pad; if (o.zero != null) { hi = Math.max(hi, o.zero); lo = Math.min(lo, o.zero); }
    x.font = '11px ' + css('--num'); L = Math.max(62, Math.ceil(Math.max(...niceTicks(lo, hi).map(t => x.measureText(yf(t)).width))) + 16);   // margin fits the widest tick label
    const y = axes(x, w, ht, L, R, T, B, lo, hi, o.labels, yf), xs = i => L + (w - L - R) * (n <= 1 ? .5 : i / (n - 1));
    if (o.zero != null) { x.strokeStyle = css('--line2'); x.setLineDash([4, 3]); x.beginPath(); x.moveTo(L, y(o.zero)); x.lineTo(w - R, y(o.zero)); x.stroke(); x.setLineDash([]); }
    for (const s of o.series) {
      x.strokeStyle = s.color; x.lineWidth = s.width || 1.6; x.setLineDash(s.dash || []); x.beginPath(); let pen = false;
      s.data.forEach((v, i) => { if (v == null || !isFinite(v)) { pen = false; return; } pen ? x.lineTo(xs(i), y(v)) : x.moveTo(xs(i), y(v)); pen = true; }); x.stroke(); x.setLineDash([]);
      if (s.area) { x.lineTo(xs(n - 1), y(o.zero ?? lo)); x.lineTo(xs(0), y(o.zero ?? lo)); x.closePath(); x.globalAlpha = .18; x.fillStyle = s.color; x.fill(); x.globalAlpha = 1; }
    }
    for (const mk of o.markers || []) { x.fillStyle = mk.color; x.font = '12px ' + css('--sans'); x.textAlign = 'center'; x.fillText(mk.glyph, xs(mk.i), y(mk.y) + (mk.glyph === '▲' ? 14 : -6)); }
    hover(m, o.labels, i => o.labels[i] + '\n' + o.series.map(s => `${s.name}: ${s.data[i] == null ? '—' : yf(s.data[i])}`).join('\n'),
      px => { const i = Math.round((px - L) / (w - L - R) * (n - 1)); return i >= 0 && i < n ? i : null; });
  };
  draw(); new ResizeObserver(draw).observe(box);
  box.prepend(h('div', { class: 'legend' }, o.series.map(s => h('span', {}, h('i', { class: 'sw', style: `border-color:${s.color};${s.dash ? 'border-top-style:dashed' : ''}` }), s.name))));
  dataTable(box, o.aria || 'Chart data', ['Date', ...o.series.map(s => s.name)], o.labels.map((l, i) => [l, ...o.series.map(s => s.data[i] == null ? 'n/a' : yf(s.data[i]))]));
}
function histChart(box, values, o = {}) {
  const m = mount(box, o.height || 200); m.c.setAttribute('aria-label', o.aria || 'Histogram');
  const nb = 24, lo = Math.min(...values), hi = Math.max(...values), bw = (hi - lo) / nb || 1, bins = Array(nb).fill(0);
  values.forEach(v => bins[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
  const draw = () => { const [x, w, ht] = m.fit(); x.clearRect(0, 0, w, ht);
    const mx = Math.max(...bins), y = axes(x, w, ht, 40, 10, 8, 24, 0, mx * 1.05, [nf(lo, 2), '', '', '', '', nf(hi, 2)], v => nf(v, 0));
    const bwp = (w - 50) / nb; bins.forEach((c, i) => { x.fillStyle = css('--info'); x.fillRect(40 + i * bwp + 1, y(c), bwp - 2, y(0) - y(c)); });
    if (lo < 0 && hi > 0) { const zx = 40 + (0 - lo) / (hi - lo) * (w - 50); x.strokeStyle = css('--acc'); x.beginPath(); x.moveTo(zx, 8); x.lineTo(zx, ht - 24); x.stroke(); }
    hover(m, [], i => `${nf(lo + i * bw, 2)} to ${nf(lo + (i + 1) * bw, 2)}\ncount ${bins[i]}`, px => { const i = Math.floor((px - 40) / bwp); return i >= 0 && i < nb ? i : null; }); };
  draw(); new ResizeObserver(draw).observe(box);
  dataTable(box, o.aria || 'Histogram', ['Bin (from to)', 'Count'], bins.map((c, i) => [`${nf(lo + i * bw, 3)} to ${nf(lo + (i + 1) * bw, 3)}`, String(c)]));
}
function candleChart(box, hist, opt) {   // real OHLCV only; up = hollow, down = filled
  const bars = hist.bars, ind = hist.indicators.series, m = mount(box, opt.rsi ? 470 : 400);
  m.c.setAttribute('aria-label', `Candlestick chart, ${bars.length} daily bars from ${bars[0].t} to ${bars[bars.length - 1].t}${opt.forecast ? '; a dashed marker to the right is a model estimate for the next trading day, not an observation' : ''}`);
  const draw = () => {
    const [x, w, ht] = m.fit(); x.clearRect(0, 0, w, ht); const L = 62, R = 12, T = 8, B = 24, n = bars.length;
    const rsiH = opt.rsi ? 70 : 0, volH = 64, pH = ht - T - B - volH - rsiH - 8, pB = ht - B - volH - rsiH - 8;
    let lo = Infinity, hi = -Infinity; bars.forEach(b => { lo = Math.min(lo, b.l); hi = Math.max(hi, b.h); });
    const fc = opt.forecast, FW = fc ? 56 : 0;   // fixed-pixel zone for the model estimate, independent of how many bars there are
    if (fc) { lo = Math.min(lo, fc.low ?? fc.price); hi = Math.max(hi, fc.high ?? fc.price); }
    if (opt.bb) ind.bb_upper.concat(ind.bb_lower).forEach(v => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    const pad = (hi - lo) * .04; lo -= pad; hi += pad;
    const y = axes(x, w, ht, L, R + FW, T, ht - pB, lo, hi, bars.map(b => b.t), v => nf(v, 2)); // price pane ends at pB; gridlines stop before the forecast zone
    const step = (w - L - R - FW) / n, cw = Math.max(1, Math.min(10, step * .68)), xs = i => L + step * (i + .5);
    const green = css('--gain'), red = css('--loss');
    bars.forEach((b, i) => { const up = b.c >= b.o, col = up ? green : red, cx = Math.round(xs(i)) + .5;
      x.strokeStyle = col; x.fillStyle = col; x.lineWidth = 1; x.beginPath(); x.moveTo(cx, y(b.h)); x.lineTo(cx, y(b.l)); x.stroke();
      const top = y(Math.max(b.o, b.c)), hh = Math.max(1, Math.abs(y(b.o) - y(b.c)));
      if (up && cw > 2) { x.fillStyle = css('--bg'); x.fillRect(cx - cw / 2, top, cw, hh); x.strokeRect(cx - cw / 2, top, cw, hh); } else x.fillRect(cx - cw / 2, top, cw, hh); });
    if (fc) {   // MODEL ESTIMATE, not an observation: dashed whisker + open diamond, right of a divider, never a candle
      const fx = Math.round(L + n * step + FW * .6) + .5, dv = Math.round(L + n * step + 6) + .5;
      x.save(); x.strokeStyle = css('--mut'); x.setLineDash([2, 4]); x.beginPath(); x.moveTo(dv, T); x.lineTo(dv, pB); x.stroke();
      x.strokeStyle = css('--acc'); x.lineWidth = 1.6; x.setLineDash([5, 3]);
      if (fc.low != null) { x.beginPath(); x.moveTo(fx, y(fc.high)); x.lineTo(fx, y(fc.low)); x.stroke(); [fc.high, fc.low].forEach(v => { x.beginPath(); x.moveTo(fx - 6, y(v)); x.lineTo(fx + 6, y(v)); x.stroke(); }); }
      x.setLineDash([]); x.fillStyle = css('--acc'); x.font = '11px ' + css('--num'); x.textAlign = 'center'; x.fillText('next day', fx, ht - 6); const py = y(fc.price); x.beginPath(); x.moveTo(fx, py - 7); x.lineTo(fx + 7, py); x.lineTo(fx, py + 7); x.lineTo(fx - 7, py); x.closePath(); x.fillStyle = css('--bg'); x.fill(); x.stroke(); x.restore();
    }
    const line = (arr, col, dash) => { x.strokeStyle = col; x.lineWidth = 1.2; x.setLineDash(dash || []); x.beginPath(); let pen = false;
      arr.forEach((v, i) => { if (v == null) { pen = false; return; } pen ? x.lineTo(xs(i), y(v)) : x.moveTo(xs(i), y(v)); pen = true; }); x.stroke(); x.setLineDash([]); };
    if (opt.sma20) line(ind.sma20, css('--acc')); if (opt.sma50) line(ind.sma50, css('--info'));
    if (opt.bb) { line(ind.bb_upper, css('--mut'), [3, 3]); line(ind.bb_lower, css('--mut'), [3, 3]); }
    const vmax = Math.max(...bars.map(b => b.v)) || 1, vy0 = pB + 8 + volH;
    x.fillStyle = css('--mut'); x.font = '11px ' + css('--num'); x.textAlign = 'left'; x.fillText('Volume (max ' + big(vmax) + ')', L + 4, pB + 20);
    bars.forEach((b, i) => { x.globalAlpha = .55; x.fillStyle = b.c >= b.o ? green : red; const hh = b.v / vmax * (volH - 6); x.fillRect(Math.round(xs(i) - cw / 2), vy0 - hh, cw, hh); x.globalAlpha = 1; });
    if (opt.rsi) { const r0 = ht - B - rsiH, ry = v => r0 + (100 - v) / 100 * rsiH;
      x.strokeStyle = css('--line'); [30, 70].forEach(v => { x.beginPath(); x.moveTo(L, ry(v)); x.lineTo(w - R, ry(v)); x.stroke(); x.fillText(v, L - 6, ry(v) + 4); });
      x.strokeStyle = css('--info'); x.beginPath(); let pen = false; ind.rsi14.forEach((v, i) => { if (v == null) { pen = false; return; } pen ? x.lineTo(xs(i), ry(v)) : x.moveTo(xs(i), ry(v)); pen = true; }); x.stroke(); x.fillText('RSI14', L - 6, r0 + 10); }
    hover(m, [], i => { if (i >= n) return fc ? `MODEL ESTIMATE (not observed)\nnext trading day after ${fc.asOf}\n${nf(fc.price)}${fc.low != null ? `\ninterval ${nf(fc.low)} – ${nf(fc.high)} (nominal ${Math.round(fc.level * 100)}%)` : '\ninterval: insufficient evaluation data'}` : '';
      const b = bars[i], f = k => ind[k][i] == null ? '—' : nf(ind[k][i], 2);
      return `${b.t}\nO ${nf(b.o)}  H ${nf(b.h)}\nL ${nf(b.l)}  C ${nf(b.c)}\nVol ${big(b.v)}\nSMA20 ${f('sma20')}  SMA50 ${f('sma50')}\nRSI14 ${f('rsi14')}`; },
      px => { if (fc && px > L + n * step + 10) return n; const i = Math.floor((px - L) / step); return i >= 0 && i < n ? i : null; });
  };
  draw(); new ResizeObserver(draw).observe(box);
  const g = (k, i) => ind[k][i] == null ? 'n/a' : nf(ind[k][i], 2);
  dataTable(box, `OHLC bars with indicators for ${hist.instrument.key}`, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI14'],
    bars.map((b, i) => [b.t, nf(b.o), nf(b.h), nf(b.l), nf(b.c), big(b.v), g('sma20', i), g('sma50', i), g('rsi14', i)]));
}

/* ---------------------------------------------------------------- shared UI pieces */
const panel = (title, body, foot, right, cls) => h('section', { class: 'pan' + (cls ? ' ' + cls : '') }, h('header', {}, h('h2', { text: title }), right || ''), h('div', { class: 'bd' }, body), foot ? h('footer', { text: foot }) : '');
const bento = (...tiles) => h('div', { class: 'bento' }, ...tiles);
const empty = (t, msg, action) => h('div', { class: 'empty' }, h('b', { text: t }), msg, action ? h('div', { style: 'margin-top:10px' }, action) : '');
const note = (kind, ...c) => h('div', { class: 'note ' + kind }, c);
const metric = (l, v, cls, title) => h('div', { class: 'metric', title }, h('div', { class: 'l', text: l }), h('div', { class: 'v ' + (cls || ''), text: v }));
const loading = t => h('div', { class: 'empty' }, h('span', { class: 'spin' }), t || 'Loading…');
function table(cols, rows, o = {}) {
  // Scrollable region must be keyboard-focusable (WCAG 2.1.1) and named, so arrow keys can scroll a wide table.
  return h('div', { class: 'tw', tabindex: '0', role: 'region', 'aria-label': o.label || 'Data table (scrollable)' }, h('table', {}, h('thead', {}, h('tr', {}, cols.map(c => h('th', { scope: 'col', class: c.num ? 'num' : '', text: c.h })))),
    h('tbody', {}, rows.map(r => h('tr', {}, cols.map(c => { const v = c.f(r); return h('td', { class: (c.num ? 'num ' : '') + (c.cls ? c.cls(r) : '') }, v); }))))));
}
const errBox = e => note('err', h('b', { text: 'Could not load. ' }), e.message);
function go(route) { location.hash = '#/' + route; }
function needSymbol() {
  return empty('Choose an instrument', 'Type a symbol in the bar above. Use NSE:RELIANCE or BSE:RELIANCE for Indian listings, VOD.L for London, or a bare symbol like AAPL for US.');
}

/* ---------------------------------------------------------------- quote strip */
async function loadQuote(force) {
  if (!S.symbol) return;
  try { S.quote = await api(`/quote?symbol=${encodeURIComponent(S.symbol)}${force ? '&force=1' : ''}`); S.retrieved = Date.now(); }
  catch (e) { S.quote = { symbol: S.symbol, freshness: 'unavailable', error: e.message }; }
  renderStrip(); schedule();
}
function updateTopPills() {
  document.querySelectorAll('#top-pills .pill').forEach(b => {
    b.classList.toggle('act', b.dataset.sym === S.symbol);
  });
}
function renderStrip() {
  const q = S.quote, box = $('#quote'); box.replaceChildren();
  updateTopPills();
  if (!q) { box.append(h('span', { class: 'mu', text: 'No instrument selected' })); return; }
  const chg = q.price != null && q.prev_close ? (q.price / q.prev_close - 1) * 100 : null, [cls, g] = dir(chg || 0);
  box.append(h('b', { text: `${q.exchange || ''}:${q.symbol || S.symbol}` }),
    h('span', { class: 'px', text: q.price != null ? nf(q.price, 2) : '—' }),
    h('span', { class: 'mu', text: q.currency || '' }),
    chg != null ? h('span', { class: cls, title: 'Change vs previous close' }, g, ' ', pct(chg)) : '', badge(q),
    h('span', { class: 'mu', text: q.session ? `session: ${q.session}` : '' }));
  $('#prov').textContent = q.error && q.price == null ? `No quote: ${q.error}` : provenance(q);
  $('#prov').className = 'dim' + (q.is_stale ? ' dn' : '');
}
function schedule() {
  clearTimeout(S.timers.q); const sec = Number(prefs.refresh);
  if (!sec) { S.nextRefresh = 0; return; }
  S.nextRefresh = Date.now() + sec * 1000;
  S.timers.q = setTimeout(() => { if (!document.hidden) { loadQuote(true); if (cur() === 'overview') loadWatchQuotes(); } else schedule(); }, sec * 1000);
}
setInterval(() => { const el = $('#age'); if (!el) return;
  const t = S.retrieved ? Math.round((Date.now() - S.retrieved) / 1000) : null;
  el.textContent = t == null ? '' : `updated ${age(t)} ago` + (S.nextRefresh ? ` · next in ${Math.max(0, Math.round((S.nextRefresh - Date.now()) / 1000))}s` : ' · auto-refresh off'); }, 1000);
function setSymbol(sym) {
  S.symbol = sym.trim().toUpperCase(); store.set('ws.symbol', S.symbol); S.hist = null; S.quote = null; S.forecast = null; S.fkey = null; try { sessionStorage.removeItem('ws.fkey'); } catch { /* ignore */ }
  $('#sym').value = S.symbol; updateTopPills(); loadQuote(); render();
}

/* ---------------------------------------------------------------- views */
const VIEWS = {};

VIEWS.overview = async root => {
  root.append(h('h1', { text: 'Market Overview' }), h('p', { class: 'sub', text: 'Your watchlist with the freshness of every quote. A price is only shown as Live when the provider is entitled to real-time data and its own timestamp is current.' }));
  const q = S.quote;
  let heroCard;
  if (S.symbol && q) {
    const chg = q.price != null && q.prev_close ? (q.price / q.prev_close - 1) * 100 : null, [cls, g] = dir(chg || 0);
    heroCard = h('div', { class: 'hero-card c12' },
      h('div', { class: 'hero-main' },
        h('div', { class: 'row', style: 'align-items:center;gap:8px' },
          h('span', { class: 'hero-sym', text: `${q.exchange || ''}:${q.symbol || S.symbol}` }),
          badge(q),
          q.session ? h('span', { class: 'tag', text: q.session }) : ''
        ),
        h('div', { class: 'hero-price-row' },
          h('span', { class: 'hero-px', text: q.price != null ? money(q.price, q.currency) : '—' }),
          chg != null ? h('span', { class: 'hero-chg ' + cls }, g, ' ', pct(chg)) : ''
        ),
        h('div', { class: 'dim', style: 'font-size:11px;margin-top:2px', text: provenance(q) })
      ),
      h('div', { class: 'hero-stats' },
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Prev Close' }), h('div', { class: 'v', text: q.prev_close != null ? nf(q.prev_close, 2) : '—' })),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Provider' }), h('div', { class: 'v', text: q.provider || '—' })),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Freshness' }), h('div', { class: 'v', text: (q.freshness || 'unavailable').replace(/_/g, ' ') })),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Data Age' }), h('div', { class: 'v', text: q.age_seconds != null ? age(q.age_seconds) : '—' }))
      ),
      h('div', { class: 'row', style: 'gap:8px' },
        h('a', { class: 'btn pri sm', href: '#/forecast', text: '⚡ Forecast Lab' }),
        h('a', { class: 'btn sm', href: '#/research', text: '⌕ Full Chart' }),
        h('a', { class: 'btn sm', href: '#/portfolio', text: '⊞ Paper Order' })
      )
    );
  } else {
    heroCard = h('div', { class: 'hero-card c12' },
      h('div', { class: 'hero-main' },
        h('span', { class: 'hero-sym', text: 'Quant Workstation Terminal' }),
        h('div', { class: 'dim', text: 'Select an instrument from the command bar above or choose a quick pick (AAPL, MSFT, NSE:TCS, NVDA) to stream quotes, analyze price history, and run quantitative forecasts.' })
      ),
      h('div', { class: 'row', style: 'gap:8px' },
        h('button', { class: 'btn pri sm', type: 'button', text: 'Load AAPL', onclick: () => setSymbol('AAPL') }),
        h('button', { class: 'btn sm', type: 'button', text: 'Load MSFT', onclick: () => setSymbol('MSFT') }),
        h('button', { class: 'btn sm', type: 'button', text: 'Load NSE:TCS', onclick: () => setSymbol('NSE:TCS') })
      )
    );
  }

  const wlBody = h('div', { id: 'wl' });
  const addForm = h('form', { class: 'row', onsubmit: e => {
    e.preventDefault(); const v = $('#wl-in').value.trim().toUpperCase(); if (!v) return;
    const w = watch(); if (!w.includes(v)) { w.push(v); store.set('ws.watch', w); }
    $('#wl-in').value = ''; loadWatchQuotes();
  } },
    h('div', { class: 'fld' }, h('label', { for: 'wl-in', text: 'Add to watchlist' }), h('input', { class: 'inp', id: 'wl-in', placeholder: 'e.g. NSE:TCS', autocomplete: 'off' })),
    h('button', { class: 'btn', type: 'submit', text: 'Add' }),
    h('button', { class: 'btn', type: 'button', text: 'Refresh now', onclick: () => loadWatchQuotes(true) })
  );
  const wlPanel = panel('Watchlist', [addForm, h('div', { style: 'margin-top:10px' }, wlBody)], 'Stored in this browser only. Quotes are cached briefly and de-duplicated server-side.', null, S.symbol ? 'c4' : 'c8');

  let centerTile;
  if (S.symbol) {
    const chartBox = h('div', {});
    const chartStats = h('div', { class: 'grid g3', style: 'margin-top:10px' });
    centerTile = panel(`${S.symbol} — Price Action & Forecast`, [chartBox, chartStats], 'Historical bars with next-day model forecast diamond if computed.', null, 'c8');
    chartBox.append(loading('Loading price action…'));
    api(`/history?symbol=${encodeURIComponent(S.symbol)}&period=${prefs.period}`).then(hs => {
      if (cur() !== 'overview') return;
      S.hist = hs;
      const b = hs.bars, last = b[b.length - 1], hi = Math.max(...b.map(x => x.h)), lo = Math.min(...b.map(x => x.l)), avgv = b.reduce((a, x) => a + x.v, 0) / b.length;
      const fcPayload = S.forecast && S.forecast.instrument && S.forecast.instrument.yahoo_symbol === hs.instrument.yahoo_symbol ? S.forecast : null;
      const fcOpt = () => fcPayload ? { price: fcPayload.final_price, asOf: fcPayload.as_of_bar, level: (fcPayload.interval || {}).nominal_coverage || .8,
        low: fcPayload.interval && fcPayload.interval.status === 'ok' ? fcPayload.interval.low : null, high: fcPayload.interval && fcPayload.interval.status === 'ok' ? fcPayload.interval.high : null } : null;
      chartBox.replaceChildren();
      candleChart(chartBox, hs, { sma20: true, sma50: true, forecast: fcOpt() });
      chartStats.replaceChildren(
        metric('Last Close', `${nf(last.c)} (${last.t})`),
        metric('Range High / Low', `${nf(hi)} / ${nf(lo)}`),
        metric('Avg Volume', big(avgv))
      );
    }).catch(e => {
      chartBox.replaceChildren(errBox(e));
    });
  } else {
    centerTile = panel('Quick Instrument Launcher', [
      empty('No Active Instrument', 'Click an instrument below or enter one in the top bar to initialize real-time charts and quantitative evaluation.'),
      h('div', { class: 'grid g3', style: 'margin-top:14px' },
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('AAPL') }, h('b', { text: 'AAPL' }), h('div', { class: 'dim', text: 'Apple Inc. (NASDAQ)' })),
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('MSFT') }, h('b', { text: 'MSFT' }), h('div', { class: 'dim', text: 'Microsoft Corp. (NASDAQ)' })),
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('NVDA') }, h('b', { text: 'NVDA' }), h('div', { class: 'dim', text: 'NVIDIA Corp. (NASDAQ)' })),
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('NSE:TCS') }, h('b', { text: 'NSE:TCS' }), h('div', { class: 'dim', text: 'Tata Consultancy (NSE)' })),
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('NSE:RELIANCE') }, h('b', { text: 'NSE:RELIANCE' }), h('div', { class: 'dim', text: 'Reliance Ind. (NSE)' })),
        h('button', { class: 'btn', type: 'button', onclick: () => setSymbol('VOD.L') }, h('b', { text: 'VOD.L' }), h('div', { class: 'dim', text: 'Vodafone Group (LSE)' }))
      )
    ], null, null, 'c4');
  }

  let fcTile;
  if (S.forecast && S.forecast.instrument && S.symbol && S.forecast.instrument.yahoo_symbol === (S.quote ? S.quote.yahoo_symbol : S.symbol)) {
    const fc = S.forecast, [k, g] = dir(fc.pct_change);
    fcTile = panel('Model Forecast Snapshot', [
      h('div', { class: 'grid g2' },
        metric('Next Close Estimate', money(fc.final_price, fc.target.currency)),
        metric('Implied Move', `${g} ${pct(fc.pct_change)}`, k)
      ),
      h('div', { style: 'margin-top:10px' },
        h('div', { class: 'dim', text: `As of bar ${fc.as_of_bar} · horizon ${fc.horizon}` }),
        fc.interval && fc.interval.status === 'ok' ? h('div', { class: 'dim', text: `80% interval: ${nf(fc.interval.low)} – ${nf(fc.interval.high)}` }) : '',
        h('div', { style: 'margin-top:8px' }, h('a', { class: 'btn sm', href: '#/evaluation', text: 'View Model Evaluation →' }))
      )
    ], 'Linear Regression + LSTM ensemble. Model agreement is not probability.', null, 'c4');
  } else {
    fcTile = panel('Model Forecast Snapshot', empty('No active forecast', S.symbol ? `Run a next-day forecast for ${S.symbol} in the Forecast Lab.` : 'Select an instrument and run a forecast.', h('a', { class: 'btn pri sm', href: '#/forecast', text: 'Open Forecast Lab' })), null, null, 'c4');
  }

  const telemTile = panel('Feed Telemetry & Health', q ? h('dl', { class: 'kv' },
    h('dt', { text: 'Provider' }), h('dd', { text: q.provider || '—' }),
    h('dt', { text: 'Exchange' }), h('dd', { text: q.exchange || '—' }),
    h('dt', { text: 'Freshness' }), h('dd', { text: (q.freshness || 'unavailable').replace(/_/g, ' ') }),
    h('dt', { text: 'Quote age' }), h('dd', { text: q.age_seconds != null ? age(q.age_seconds) : '—' }),
    h('dt', { text: 'Cache state' }), h('dd', { text: q.from_cache ? `Cached (${age(q.cache_age_seconds)} ago)` : 'Direct upstream' }),
    h('dt', { text: 'Retrieved' }), h('dd', { text: tsLocal(q.retrieved_ts) })
  ) : empty('No telemetry', 'Select an instrument to view feed status.'), null, null, 'c4');

  const breadthTile = panel('Market Breadth', empty('Not available', 'No breadth data source (advance/decline counts) is configured, so none is shown. Nothing is estimated.'), null, null, 'c4');

  root.append(bento(heroCard, centerTile, wlPanel, fcTile, telemTile, breadthTile));
  loadWatchQuotes();
};
async function loadWatchQuotes() {
  const box = $('#wl'); if (!box) return; const w = watch();
  if (!w.length) { box.replaceChildren(empty('Your watchlist is empty', 'Add a symbol above. No example symbols are pre-loaded.')); return; }
  try {
    const { quotes } = await api('/quotes?symbols=' + encodeURIComponent(w.join(','))); if (!$('#wl')) return;
    box.replaceChildren(table([
      { h: 'Symbol', f: q => h('a', { href: '#/research', onclick: () => setSymbol(q.symbol && q.exchange && q.exchange !== 'US/UNSPECIFIED' ? `${q.exchange}:${q.symbol}` : (q.symbol || '')), text: `${q.exchange || ''}:${q.symbol}` }) },
      { h: 'Last', num: 1, f: q => nf(q.price, 2) }, { h: 'Chg', num: 1, f: q => { const c = q.price != null && q.prev_close ? (q.price / q.prev_close - 1) * 100 : null; const [k, g] = dir(c || 0); return c == null ? '—' : h('span', { class: k }, g + ' ' + pct(c)); } },
      { h: 'Freshness', f: q => badge(q) }, { h: 'Provider', f: q => q.provider || '—' }, { h: 'Source time', f: q => q.source_ts ? `${tsLocal(q.source_ts)} (${age(q.age_seconds)} old)` : (q.error ? h('span', { class: 'dn', text: q.error.slice(0, 90) }) : '—') },
      { h: '', f: q => h('button', { class: 'btn sm', 'aria-label': 'Remove ' + q.symbol, text: 'Remove', onclick: () => { const key = (q.exchange && q.exchange !== 'US/UNSPECIFIED' ? q.exchange + ':' : '') + q.symbol; store.set('ws.watch', watch().filter(s => s !== key && s !== q.symbol && s.replace(/\.(NS|BO)$/, '') !== q.symbol)); loadWatchQuotes(); } }) }], quotes));
  } catch (e) { box.replaceChildren(errBox(e)); }
}

VIEWS.research = async root => {
  root.append(h('h1', { text: 'Instrument Research' }), h('p', { class: 'sub', text: `Deep-dive candlestick price action, technical overlays, range statistics, and news feeds for ${S.symbol || 'the selected instrument'}.` }));
  if (!S.symbol) { root.append(needSymbol()); return; }
  const opts = Object.assign({ sma20: true, sma50: true, bb: false, rsi: false }, store.get('ws.chart', {}));
  const periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y'], chart = h('div', {}), info = h('div', {}), news = h('div', {});
  const ctl = h('div', { class: 'row' }, h('div', { class: 'fld' }, h('label', { for: 'per', text: 'Range' }), h('select', { class: 'sel', id: 'per', onchange: e => { prefs.period = e.target.value; store.set('ws.prefs', prefs); S.hist = null; render(); } }, periods.map(p => h('option', { value: p, selected: p === prefs.period, text: p })))),
    [['sma20', 'SMA 20'], ['sma50', 'SMA 50'], ['bb', 'Bollinger (20, 2)'], ['rsi', 'RSI 14']].map(([k, l]) => h('label', { class: 'row', style: 'align-items:center;gap:4px' }, h('input', { type: 'checkbox', checked: opts[k], onchange: e => { opts[k] = e.target.checked; store.set('ws.chart', opts); if (S.hist) candleChart(chart, S.hist, S.chartOpts ? S.chartOpts() : opts); } }), l)));
  root.append(bento(panel(`${S.symbol} — daily OHLC`, [ctl, chart], null, null, 'c12'), panel('Range statistics & data quality', info, null, null, 'c6'), panel('News', news, null, null, 'c6')));
  chart.append(loading('Loading daily bars…')); news.append(loading('Checking configured news feeds…'));
  api(`/news?symbol=${encodeURIComponent(S.symbol)}`).then(n => { news.replaceChildren(...newsList(n)); }).catch(e => news.replaceChildren(errBox(e)));
  try {
    S.hist = S.hist || await api(`/history?symbol=${encodeURIComponent(S.symbol)}&period=${prefs.period}`); if (cur() !== 'research') return;
    const hs = S.hist, b = hs.bars, last = b[b.length - 1], hi = Math.max(...b.map(x => x.h)), lo = Math.min(...b.map(x => x.l)), avgv = b.reduce((a, x) => a + x.v, 0) / b.length;
    const fcPayload = S.forecast && S.forecast.instrument && S.forecast.instrument.yahoo_symbol === hs.instrument.yahoo_symbol ? S.forecast : null;
    const fcOpt = () => fcPayload ? { price: fcPayload.final_price, asOf: fcPayload.as_of_bar, level: (fcPayload.interval || {}).nominal_coverage || .8,
      low: fcPayload.interval && fcPayload.interval.status === 'ok' ? fcPayload.interval.low : null, high: fcPayload.interval && fcPayload.interval.status === 'ok' ? fcPayload.interval.high : null } : null;
    S.chartOpts = () => Object.assign({}, opts, { forecast: fcOpt() });
    chart.replaceChildren(); candleChart(chart, hs, S.chartOpts());
    chart.after(h('div', { class: 'dim', style: 'padding:6px 0 0' },
      `Daily bars · dates are exchange-local trading days (${hs.instrument.timezone}) · adjusted prices · source ${hs.provider}, retrieved ${tsLocal(hs.retrieved_ts)}.`,
      fcPayload ? h('div', {}, '◇ Dashed whisker/diamond = model estimate for the next trading day after ', fcPayload.as_of_bar, ' (',
        nf(fcPayload.final_price), fcPayload.interval && fcPayload.interval.status === 'ok' ? `, interval ${nf(fcPayload.interval.low)} – ${nf(fcPayload.interval.high)}` : ', interval unavailable: insufficient evaluation data',
        '), generated ', tsLocal(fcPayload.generated_ts), '. It is not an observation.') : h('div', {}, 'No forecast shown: run one in the Forecast Lab and it will appear here, visually separated from observed bars.')));
    const dq = hs.data_quality, flags = [...(dq.issues || []).map(t => ['err', t]), ...(dq.warnings || []).map(t => ['warn', t])];
    info.append(h('dl', { class: 'kv' }, h('dt', { text: 'Last completed bar' }), h('dd', { text: `${last.t} · close ${nf(last.c)}` }), h('dt', { text: 'Range high / low' }), h('dd', { class: 'num', text: `${nf(hi)} / ${nf(lo)}` }),
      h('dt', { text: 'Average volume' }), h('dd', { class: 'num', text: big(avgv) }), h('dt', { text: 'Bars' }), h('dd', { text: `${hs.rows} (${hs.first_bar} → ${hs.last_bar})` }),
      h('dt', { text: 'Adjustment' }), h('dd', { text: 'split/dividend adjusted' }), h('dt', { text: 'Provider' }), h('dd', { text: hs.provider + (hs.from_cache ? ` (cached ${age(hs.cache_age_seconds)})` : '') }),
      h('dt', { text: 'RSI14 / SMA20' }), h('dd', { class: 'num', text: `${nf(hs.indicators.latest.rsi14, 1)} / ${nf(hs.indicators.latest.sma20)}` })),
      flags.length ? flags.map(([k, t]) => note(k, t)) : note('ok', 'No data-quality issues found in these bars.'),
      note('', 'The last bar may be an unfinished session if the market is open. The Forecast Lab removes it before modelling.'));
  } catch (e) { chart.replaceChildren(errBox(e), note('', 'If this is a rate limit or outage, wait and press Refresh. The app never substitutes sample data.')); }
};
function newsList(n) {
  if (n.status === 'not_configured') return [empty('No news source configured', n.message)];
  const out = []; if (n.errors.length) out.push(note('warn', 'Some feeds failed: ' + n.errors.map(e => `${e.feed} (${e.error})`).join('; ')));
  if (!n.items.length) return [...out, empty('No headlines', 'The configured feeds returned nothing for this symbol.')];
  out.push(h('ul', { style: 'list-style:none;padding:0;margin:0' }, n.items.map(i => h('li', { style: 'padding:6px 0;border-bottom:1px solid var(--line)' },
    (u => u ? h('a', { href: u, target: '_blank', rel: 'noopener noreferrer', text: i.title }) : h('span', { text: i.title }))(safeUrl(i.url)),
    h('div', { class: 'dim', text: `${i.publisher} · ${i.published_ts ? tsLocal(i.published_ts) : 'publication time not provided'} · retrieved ${tsLocal(i.retrieved_ts)}` })))),
    h('div', { class: 'dim', text: n.coverage_note }));
  return out;
}

VIEWS.forecast = async root => {
  root.append(h('h1', { text: 'Forecast Lab' }), h('p', { class: 'sub', text: 'Next-trading-day close from Linear Regression and an LSTM, compared against naive baselines on the same out-of-sample days. Training runs in the background; quote refreshes never retrain.' }));
  if (!S.symbol) { root.append(needSymbol()); return; }
  const out = h('div', {}), status = h('div', {});
  const form = h('form', { class: 'row', onsubmit: async e => { e.preventDefault(); run(); } },
    h('div', { class: 'fld' }, h('label', { for: 'fp', text: 'History for training' }), h('select', { class: 'sel', id: 'fp' }, ['1y', '2y', '5y'].map(p => h('option', { value: p, selected: p === prefs.fperiod, text: p })))),
    h('div', { class: 'fld' }, h('label', { for: 'fe', text: 'LSTM epochs (if training)' }), h('input', { class: 'inp', id: 'fe', type: 'number', min: 1, max: 200, value: prefs.epochs, style: 'width:90px' })),
    h('label', { class: 'row', style: 'align-items:center;gap:4px' }, h('input', { type: 'checkbox', id: 'fw' }), 'Also run walk-forward (slower)'),
    h('button', { class: 'btn pri', id: 'run', type: 'submit', text: 'Run Prediction' }), h('button', { class: 'btn', id: 'cancel', type: 'button', text: 'Cancel', disabled: true }));
  const classic = $('#classic-form'); const classicBox = classic ? h('details', {}, h('summary', { text: 'Classic full report (server-rendered; includes CSV/PDF export and the gated trade flow)' }), classic.content.cloneNode(true)) : '';
  root.append(bento(panel(`Run for ${S.symbol}`, [form, status], 'Enter a different instrument in the top bar. Results are cached by last completed bar.', null, 'c12')), out, classicBox);
  const tk = $('input[name=ticker]', root); if (tk) tk.value = S.symbol;
  if (S.forecast) renderForecast(out, S.forecast);
  async function run() {
    prefs.fperiod = $('#fp').value; prefs.epochs = Number($('#fe').value) || 15; store.set('ws.prefs', prefs);
    $('#run').disabled = true; out.replaceChildren(); status.replaceChildren(loading('Submitting…'));
    try {
      const job = await api('/forecast', { symbol: S.symbol, period: prefs.fperiod, epochs: prefs.epochs, walk_forward: $('#fw').checked });
      $('#cancel').disabled = false; $('#cancel').onclick = () => api(`/jobs/${job.id}/cancel`, {}).then(r => status.append(note('warn', r.note)));
      await poll(job.id, status);
      const j = await api(`/jobs/${job.id}`); if (j.status !== 'done') throw new Error(j.status === 'cancelled' ? 'Cancelled.' : (j.error || 'Forecast failed.'));
      S.fkey = j.result_key; S.forecast = await api(`/forecast/result?key=${encodeURIComponent(S.fkey)}`); status.replaceChildren();
      try { sessionStorage.setItem('ws.fkey', JSON.stringify({ symbol: S.symbol, key: S.fkey })); } catch { /* private mode */ }
      if (cur() === 'forecast') renderForecast(out, S.forecast);
    } catch (e) { status.replaceChildren(errBox(e)); }
    finally { const r = $('#run'); if (r) r.disabled = false; const c = $('#cancel'); if (c) c.disabled = true; }
  }
};
async function poll(id, box) {
  const t0 = Date.now();
  for (;;) { const j = await api(`/jobs/${id}`); box.replaceChildren(h('div', {}, h('span', { class: 'spin' }), `${j.status}: ${j.stage} · ${Math.round((Date.now() - t0) / 1000)}s`)); if (['done', 'failed', 'cancelled'].includes(j.status)) return j; await new Promise(r => setTimeout(r, 1500)); }
}
function renderForecast(root, f) {
  root.replaceChildren(); const cur0 = f.target.currency || '', [k, g] = dir(f.pct_change);
  const iv = f.interval, e = iv.evaluation || {};
  const ivEl = iv.status === 'ok'
    ? h('div', {}, h('div', { class: 'v num', text: `${nf(iv.low)} – ${nf(iv.high)}` }),
        h('div', { class: 'dim', text: `Nominal ${Math.round(iv.nominal_coverage * 100)}%. Calibrated on the ${iv.calibration.n} most recent out-of-sample residuals (ending ${iv.calibration.end}); every one is dated before the forecast target.` }),
        h('div', { class: 'dim', text: `Historical coverage of this same procedure: ${nf(e.observed_coverage * 100, 0)}% observed over ${e.n_evaluated} past days (${e.evaluation_period.start} → ${e.evaluation_period.end}) — ${e.assessment}. This describes the method; it does not validate this particular interval.` }),
        h('div', { class: 'dim', text: iv.caveat }))
    : h('div', {}, h('b', { text: 'Insufficient evaluation data' }), h('div', { class: 'dim', text: iv.message || 'An interval cannot be reliably estimated.' }));
  root.append(bento(
    h('div', { class: 'hero-card c12' },
      h('div', { class: 'hero-main' },
        h('div', { class: 'row', style: 'align-items:center;gap:8px' },
          h('span', { class: 'hero-sym', text: `${S.symbol} Next-Trading-Day Forecast` }),
          h('span', { class: 'tag', text: f.horizon })
        ),
        h('div', { class: 'hero-price-row' },
          h('span', { class: 'hero-px', text: money(f.final_price, cur0) }),
          h('span', { class: 'hero-chg ' + k }, g, ' ', pct(f.pct_change))
        ),
        h('div', { class: 'dim', style: 'font-size:11px' }, `Numerical ensemble: next trading day · Model estimate generated ${f.generated_ts}; LR+LSTM only, no AI component. Not an observation.`)
      ),
      h('div', { class: 'hero-stats' },
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Last Completed Bar' }), h('div', { class: 'v', text: `${money(f.last_close, cur0)}` }), h('div', { class: 'dim', style: 'font-size:10px' }, f.as_of_bar)),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Target' }), h('div', { class: 'v', text: f.target.lstm })),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Provider' }), h('div', { class: 'v', text: f.data_provider })),
        h('div', { class: 'hero-stat' }, h('div', { class: 'l', text: 'Evaluation' }), h('div', { class: 'v', text: f.evaluation_summary.slice(0, 18) + '…' }))
      )
    ),
    f.dropped_partial_bar ? h('div', { class: 'c12' }, note('', `The ${f.dropped_partial_bar} bar belongs to a session that was still open, so it was excluded from modelling.`)) : '',
    panel('Components & uncertainty', [table([{ h: 'Model', f: r => r[0] }, { h: 'Predicted close', num: 1, f: r => nf(r[1]) }, { h: 'vs last close', num: 1, f: r => pct((r[1] / f.last_close - 1) * 100) }, { h: 'Weight', num: 1, f: r => nf((f.weights_used[r[2]] || 0) * 100, 0) + '%' }],
      Object.entries(f.components).map(([n, v]) => [n.replace('_', ' '), v, n])), h('div', { style: 'height:10px' }), h('div', { class: 'l mu', text: 'Prediction interval' }), ivEl,
      note('', 'No AI model contributes to this number by default (its weight is 0%). Model agreement is not a probability of being right.'),
      h('details', {}, h('summary', { text: 'Legacy heuristic confidence (not calibrated)' }), h('p', { class: 'mu', text: `Label: ${f.heuristic_confidence_legacy.label}. ${f.heuristic_confidence_legacy.note}` }))],
      `Horizon: ${f.horizon} · target: ${f.target.lstm} · ${f.target.adjustment}`, null, 'c6'),
    panel('Run details', h('dl', { class: 'kv' }, h('dt', { text: 'Data' }), h('dd', { text: `${f.data_provider}, ${f.training.data_first_bar} → ${f.training.data_last_bar}` }), h('dt', { text: 'Train / test rows' }), h('dd', { text: `${f.training.train_rows} / ${f.training.test_rows}` }),
      h('dt', { text: 'Lookback · split · seed' }), h('dd', { text: `${f.training.lookback} · ${f.training.train_test_split} · ${f.training.seed}` }), h('dt', { text: 'LSTM' }), h('dd', { text: f.training.lstm_source + (f.training.epochs ? ` (${f.training.epochs} epochs)` : '') }),
      h('dt', { text: 'LSTM training data ends' }), h('dd', { text: `${f.training.lstm_training_data_ends} (first evaluated date ${f.training.first_evaluated_date})` }),
      h('dt', { text: 'Timings of the run that produced this result' }), h('dd', { text: Object.entries(f.timings_ms).map(([a, b]) => `${a} ${nf(b / 1000, 1)}s`).join(' · ') || 'n/a' }), h('dt', { text: 'Generated' }), h('dd', { text: tsLocal(f.generated_ts) })),
      f.disclaimer, null, 'c6'),
    h('div', { class: 'c12' }, note('', h('b', { text: 'Evaluation verdict: ' }), f.evaluation_summary + '. See Model Evaluation for the full comparison.'))
  ));
}

VIEWS.evaluation = async root => {
  root.append(h('h1', { text: 'Model Evaluation' }), h('p', { class: 'sub', text: 'Every model is scored on the same held-out days with the same target. Training-set performance is never shown here.' }));
  const f = S.forecast; if (!f) { root.append(empty('No forecast yet', 'Run a forecast first; evaluation is computed with it.', h('a', { class: 'btn pri', href: '#/forecast', text: 'Open Forecast Lab' }))); return; }
  const ev = f.evaluation, na = (r, k) => r[k] == null ? h('span', { class: 'dim', title: (r.unavailable || {})[k] || 'unavailable', text: 'n/a' }) : null;
  const cols = [{ h: 'Model', f: r => r.model.replace(/_/g, ' ') }, { h: 'n', num: 1, f: r => r.n_obs }, { h: 'MAE', num: 1, f: r => na(r, 'mae') || nf(r.mae, 3) }, { h: 'RMSE', num: 1, f: r => na(r, 'rmse') || nf(r.rmse, 3) },
    { h: 'Return RMSE', num: 1, f: r => r.return_rmse == null ? 'n/a' : nf(r.return_rmse * 100, 3) + '%' }, { h: 'MAPE', num: 1, f: r => na(r, 'mape') || nf(r.mape, 2) + '%' },
    { h: 'Direction', num: 1, f: r => na(r, 'directional_accuracy_pct') || nf(r.directional_accuracy_pct, 1) + '%' },
    { h: 'MASE', num: 1, f: r => na(r, 'mase') || nf(r.mase, 3) },
    { h: 'RMSE vs naive', num: 1, cls: r => r.rmse_improvement_vs_naive_pct > 0 ? 'up' : r.rmse_improvement_vs_naive_pct < 0 ? 'dn' : '', f: r => r.model === 'naive' ? 'baseline' : (na(r, 'rmse_improvement_vs_naive_pct') || (r.rmse_improvement_vs_naive_pct > 0 ? '▲ ' : '▼ ') + pct(r.rmse_improvement_vs_naive_pct, 1)) },
    { h: '95% CI of that', num: 1, f: r => r.model === 'naive' ? '' : (r.rmse_improvement_ci95 ? `[${nf(r.rmse_improvement_ci95[0], 1)}, ${nf(r.rmse_improvement_ci95[1], 1)}]%` : h('span', { class: 'dim', title: (r.unavailable || {}).rmse_improvement_ci95 || 'unavailable', text: 'n/a' })) },
    { h: '99% CI (verdict)', num: 1, f: r => r.model === 'naive' ? '' : (r.rmse_improvement_ci99 ? `[${nf(r.rmse_improvement_ci99[0], 1)}, ${nf(r.rmse_improvement_ci99[1], 1)}]%` : 'n/a') },
    { h: 'DM z (p)', num: 1, f: r => r.diebold_mariano_vs_naive ? `${r.diebold_mariano_vs_naive.z} (${r.diebold_mariano_vs_naive.p_two_sided})` : (r.model === 'naive' ? '' : 'n/a') }];
  const reasons = ev.models.flatMap(r => Object.entries(r.unavailable || {}).map(([k, v]) => `${r.model} · ${k}: ${v}`));
  const a = h('div', {}), r1 = h('div', {}), r2 = h('div', {}); const lab = ev.series.map(s => s.date);
  const ie = ev.interval_evaluation, ib = h('div', {});
  const wf = ev.walk_forward;
  const wfTile = (wf.linear_regression || wf.lstm)
    ? panel('Walk-forward folds (expanding window)', table([{ h: 'Model', f: r => r.m.replace('_', ' ') }, { h: 'Fold', num: 1, f: r => r.fold }, { h: 'Train end', f: r => r.train_end_date }, { h: 'Test window', f: r => `${r.test_start_date} → ${r.test_end_date}` }, { h: 'n', num: 1, f: r => r.test_size }, { h: 'MAE', num: 1, f: r => nf(r.mae, 3) }, { h: 'RMSE', num: 1, f: r => nf(r.rmse, 3) }, { h: 'Direction', num: 1, f: r => nf(r.directional_accuracy_pct, 1) + '%' }], ['linear_regression', 'lstm'].flatMap(m => wf[m] ? wf[m].folds.map(x => ({ m, ...x })) : [])), 'LSTM folds use fewer epochs than the main model; compare with care.', null, 'c12')
    : h('div', { class: 'c12' }, note('', 'Walk-forward was not run for this forecast. Tick “Also run walk-forward” in the Forecast Lab to add it.'));

  root.append(bento(
    panel(`Out-of-sample comparison — ${ev.period.start} → ${ev.period.end} (${ev.n_obs} days)`, [table(cols, ev.models), ev.status === 'insufficient_evaluation_data' ? note('warn', h('b', { text: 'Insufficient evaluation data. ' }), ev.conclusion) : note('', h('b', { text: 'Verdict: ' }), ev.conclusion + '.'), ev.n_dropped_nonfinite ? note('warn', `${ev.n_dropped_nonfinite} day(s) with non-finite values were excluded (not imputed).`) : '',
      reasons.length ? h('details', {}, h('summary', { text: `Metrics that could not be computed (${reasons.length})` }), h('ul', {}, reasons.map(t => h('li', { text: t })))) : '',
      h('div', { class: 'dim', text: `Horizon: ${ev.horizon.definition}.` }), h('div', { class: 'dim', text: ev.mase_note }), h('div', { class: 'dim', text: ev.ci_note }),
      ev.n_dropped_gap ? note('warn', `${ev.n_dropped_gap} target(s) were excluded because more than ${ev.max_gap_days} calendar days separated them from their base bar (missing data), so they are not next-day forecasts.`) : '',
      h('div', { class: 'dim', text: `Target: ${ev.target}. Drift baseline uses mean daily return before ${ev.period.start}${ev.drift_mean_return_used == null ? ' (unavailable: <30 prior returns)' : ` (${nf(ev.drift_mean_return_used * 100, 3)}%/day)`}. Direction excludes flat days. "DM z" is an approximate Diebold–Mariano test vs naive; positive = lower squared error.` })],
      'Computed by evaluation.py on the pipeline’s chronological held-out split; scalers/models never saw these days.', null, 'c12'),
    panel('Actual vs predicted close (aligned by target date)', a, null, null, 'c12'),
    panel('Ensemble residuals (actual − predicted), histogram', r1, null, null, 'c6'),
    panel('Ensemble residuals over time', r2, null, null, 'c6'),
    panel('Prediction-interval backtest (the intervals shown are the intervals scored)', ie.status === 'ok' ? [
      h('div', { class: 'grid g3' }, metric('Nominal coverage', nf(ie.nominal_coverage * 100, 0) + '%'), metric('Observed coverage', nf(ie.observed_coverage * 100, 1) + '%', ie.observed_coverage < ie.nominal_coverage - .1 ? 'dn' : '', `${ie.n_covered} of ${ie.n_evaluated} days inside the interval`),
        metric('Mean interval width', `${nf(ie.mean_relative_width * 100, 2)}% of price`)), note(ie.observed_coverage < ie.nominal_coverage - .1 ? 'warn' : '', `${ie.assessment}. ${ie.caveat}`), ib]
      : [note('warn', h('b', { text: 'Insufficient evaluation data. ' }), ie.message || 'Intervals cannot be reliably estimated.')],
      ie.status === 'ok' ? `Window ${ie.evaluation_period.start} → ${ie.evaluation_period.end}. ${ie.method}.` : null, null, 'c12'),
    wfTile
  ));

  const col = { actual: css('--tx'), naive: css('--dim'), drift: css('--mut'), linear_regression: css('--info'), lstm: css('--acc'), lr_lstm_ensemble: css('--gain') };
  lineChart(a, { labels: lab, aria: 'Actual versus predicted closes', series: ['actual', 'naive', 'linear_regression', 'lstm', 'lr_lstm_ensemble'].map(k => ({ name: k.replace(/_/g, ' '), color: col[k], data: ev.series.map(s => s[k] ?? null), dash: k === 'naive' ? [3, 3] : null, width: k === 'actual' ? 2 : 1.3 })) });
  const res = ev.residual_series.lr_lstm_ensemble; histChart(r1, res.map(x => x.residual), { aria: 'Histogram of ensemble residuals' });
  lineChart(r2, { labels: res.map(x => x.date), zero: 0, height: 200, aria: 'Ensemble residuals over time', series: [{ name: 'residual', color: css('--info'), data: res.map(x => x.residual) }] });
  if (ie.status === 'ok') lineChart(ib, { labels: ie.series.map(x => x.date), height: 240, aria: 'Actual close against the evaluated prediction interval', series: [
    { name: 'actual', color: css('--tx'), data: ie.series.map(x => x.actual), width: 1.8 }, { name: 'interval low', color: css('--info'), data: ie.series.map(x => x.low), dash: [3, 3] }, { name: 'interval high', color: css('--info'), data: ie.series.map(x => x.high), dash: [3, 3] }] });
};

VIEWS.backtest = async root => {
  root.append(h('h1', { text: 'Strategy Backtesting' }), h('p', { class: 'sub', text: 'Replays the BUY/SELL/HOLD threshold rule on the models’ held-out days using real adjusted prices. A simulation, not a prediction of future results.' }));
  if (!S.fkey) { root.append(empty('No forecast yet', 'A backtest uses the out-of-sample predictions from a forecast run.', h('a', { class: 'btn pri', href: '#/forecast', text: 'Open Forecast Lab' }))); return; }
  const ev = S.forecast.evaluation, d0 = ev.period.start, d1 = ev.period.end, out = h('div', {});
  const F = (id, label, val, extra) => h('div', { class: 'fld' }, h('label', { for: id, text: label }), h('input', Object.assign({ class: 'inp', id, value: val, style: 'width:110px' }, extra || {})));
  const form = h('form', { class: 'row', onsubmit: async e => { e.preventDefault(); out.replaceChildren(loading('Running…'));
    const v = id => $('#' + id).value;
    try { out.replaceChildren(...renderBacktest(await api('/backtest', { key: S.fkey, model: v('bm'), initial_capital: +v('bc'), transaction_cost_pct: +v('bt'), slippage_bps: +v('bs'), buy_threshold_pct: +v('bb'), sell_threshold_pct: +v('bl'), position_size_pct: +v('bp') / 100, execution: v('be'), start: v('b0'), end: v('b1') }))); }
    catch (er) { out.replaceChildren(errBox(er)); } } },
    h('div', { class: 'fld' }, h('label', { for: 'bm', text: 'Signal source' }), h('select', { class: 'sel', id: 'bm' }, ['lr_lstm_ensemble', 'linear_regression', 'lstm'].map(m => h('option', { value: m, text: m.replace(/_/g, ' ') })))),
    F('bc', 'Initial capital', 100000), F('bt', 'Cost % per side', 0.1), F('bs', 'Slippage bps', 2), F('bb', 'Buy ≥ %', 0.5), F('bl', 'Sell ≤ %', -0.5), F('bp', 'Position size %', 100),
    h('div', { class: 'fld' }, h('label', { for: 'be', text: 'Execution' }), h('select', { class: 'sel', id: 'be' }, h('option', { value: 'next_open', text: 'Next open (default)' }), h('option', { value: 'close', text: 'Same-day close (MOC, optimistic)' }))),
    F('b0', 'From', d0, { type: 'date', min: d0, max: d1, style: 'width:150px' }), F('b1', 'To', d1, { type: 'date', min: d0, max: d1, style: 'width:150px' }), h('button', { class: 'btn pri', type: 'submit', text: 'Run backtest' }));
  root.append(bento(panel(`Backtest on ${S.symbol} — held-out window ${d0} → ${d1}`, form, 'Tuning thresholds on this same window makes the result in-sample. Treat it as a stress test, not evidence.', null, 'c12')), out);
};
function renderBacktest(r) {
  const [k] = dir(r.total_return_pct - r.buy_hold_return_pct), nfx = (v, d = 2) => v == null ? 'n/a' : nf(v, d);
  const eq = h('div', {}), dd = h('div', {});
  const metricsTile = h('div', { class: 'grid g3' }, metric('Strategy return', pct(r.total_return_pct), dir(r.total_return_pct)[0]), metric('Buy & Hold (same entry/costs)', pct(r.buy_hold_return_pct), dir(r.buy_hold_return_pct)[0]),
    metric('Strategy minus Buy & Hold', pct(r.total_return_pct - r.buy_hold_return_pct), k), metric('Max drawdown', pct(r.max_drawdown_pct), 'dn'), metric('Sharpe (rf as configured)', nfx(r.sharpe_ratio)), metric('Trades (round trips closed)', `${r.num_trades} fills`),
    metric('Win rate', r.win_rate_pct == null ? 'n/a (no closed trades)' : nf(r.win_rate_pct, 1) + '%'), metric('Costs paid', nf(r.total_transaction_costs)), metric('Skipped fills (no valid open)', r.skipped_fills), metric('Exposure (days in market)', r.exposure_pct == null ? 'n/a' : nf(r.exposure_pct, 1) + '%'), metric('Turnover (x capital traded)', r.turnover_x == null ? 'n/a' : nf(r.turnover_x, 2) + 'x'));
  
  let noteTile = null;
  if (r.num_trades === 0) {
    const sd = r.signal_diagnostics;
    const diag = sd && sd.zero_trade_reason ? ` (${sd.zero_trade_reason}. Forecast range: [${nf(sd.min_pct, 2)}%, ${nf(sd.max_pct, 2)}%], Buy threshold: +${nf(sd.buy_threshold_pct, 2)}%).` : '';
    noteTile = note('warn', h('b', { text: 'No trades were made. ' }), 'The strategy stayed in cash, so "Strategy minus Buy & Hold" below only reflects how the market moved over this window (a falling market flatters cash). It is not evidence of skill.' + diag);
  }
  else if (r.exposure_pct != null && r.exposure_pct < 50) {
    noteTile = note('', h('b', { text: `Invested only ${nf(r.exposure_pct, 0)}% of the time. ` }), 'A strategy that is often in cash will beat Buy & Hold in a falling market and lag in a rising one without any forecasting skill; compare risk-adjusted metrics and drawdown, not just the return difference.');
  }

  const idx = Object.fromEntries(r.equity_curve.map((e, i) => [e.date, i])), mk = r.trades.map(t => ({ i: idx[t.date], y: r.equity_curve[idx[t.date]].portfolio_value, glyph: t.action === 'BUY' ? '▲' : '▼', color: t.action === 'BUY' ? css('--gain') : css('--loss') })).filter(m => m.i != null);
  setTimeout(() => {
    lineChart(eq, { labels: r.equity_curve.map(e => e.date), aria: 'Equity curve versus buy and hold', markers: mk, series: [{ name: 'Strategy', color: css('--acc'), data: r.equity_curve.map(e => e.portfolio_value), width: 2 }, { name: 'Buy & Hold', color: css('--info'), data: r.benchmark_curve.map(e => e.portfolio_value), dash: [4, 3] }] });
    lineChart(dd, { labels: r.drawdown_curve.map(e => e.date), zero: 0, height: 170, yFmt: v => nf(v, 1) + '%', aria: 'Drawdown', series: [{ name: 'Drawdown %', color: css('--loss'), data: r.drawdown_curve.map(e => e.drawdown_pct), area: true }] });
  }, 0);

  return [bento(
    h('div', { class: 'c12' }, metricsTile),
    noteTile ? h('div', { class: 'c12' }, noteTile) : '',
    panel('Equity curve vs Buy & Hold', eq, r.assumptions, null, 'c8'),
    panel('Drawdown', dd, null, null, 'c4'),
    panel(`Trade ledger (${r.trades.length} fills) — ▲ buy, ▼ sell`, r.trades.length ? table([{ h: 'Date', f: t => t.date }, { h: 'Side', f: t => (t.action === 'BUY' ? '▲ ' : '▼ ') + t.action }, { h: 'Fill price', num: 1, f: t => nf(t.price) }, { h: 'Shares', num: 1, f: t => nf(t.shares, 4) }, { h: 'Cost', num: 1, f: t => nf(t.cost) }, { h: 'P&L', num: 1, cls: t => t.pnl > 0 ? 'up' : t.pnl < 0 ? 'dn' : '', f: t => t.pnl == null ? '' : nf(t.pnl) }], r.trades) : empty('No trades', (r.signal_diagnostics && r.signal_diagnostics.zero_trade_reason ? r.signal_diagnostics.zero_trade_reason + '. ' : '') + 'The model never predicted a move beyond the buy/sell thresholds in this window. Lowering the thresholds will produce trades, but tuning them on this same window makes the result in-sample.'), `Window ${r.window.start} → ${r.window.end} (${r.window.n_days} days)`, null, 'c12'),
    h('div', { class: 'c12' }, note('warn', h('b', { text: 'Limitations: ' }), r.caveats.join(' ')))
  )];
}

VIEWS.ai = async root => {
  root.append(h('h1', { text: 'AI Research' }), h('p', { class: 'sub', text: 'AI explains numbers the app computed. It does not forecast, does not vote on the price, and cannot trade. A provider is “verified” only after a real request succeeds.' }));
  const st = h('div', {}), out = h('div', {});
  root.append(bento(
    panel('Providers', st, 'Keys are read from .env on the server and never sent to this page.', null, 'c12'),
    panel('Commentary on the current forecast', [h('div', { class: 'row' },
      h('button', { class: 'btn pri', id: 'gen', text: 'Generate commentary', disabled: !S.fkey, onclick: () => gen(false) }), h('button', { class: 'btn', text: 'Regenerate (bypass cache)', disabled: !S.fkey, onclick: () => gen(true) }), S.fkey ? '' : h('span', { class: 'mu', text: 'Run a forecast first.' })), out], null, null, 'c12')
  ));
  async function status() { try { const { providers } = await api('/ai/status'); st.replaceChildren(table([{ h: 'Provider', f: p => p.label }, { h: 'Model', f: p => p.model || h('span', { class: 'dim', text: `not set (${p.model_env})` }) },
    { h: 'Status', f: p => h('span', { class: 'tag ' + (p.state === 'verified' ? 'ok' : p.state === 'missing_key' || p.state === 'untested' ? '' : 'no'), title: p.verified_ts ? `Verified ${p.verified_ts} via ${p.transport} transport` : '', text: p.state === 'mock_verified' ? 'verified by TEST DOUBLE (not real)' : p.state.replace(/_/g, ' ') }) },
    { h: 'Detail', f: p => h('div', {}, p.last_error ? h('div', { class: 'dn', text: p.last_error }) : '', p.action ? h('div', { class: 'mu', text: p.action }) : '', p.free_tier_note ? h('div', { class: 'dim', text: p.free_tier_note }) : '', p.data_use_note ? h('div', { class: 'dim', text: p.data_use_note }) : '', p.last_success_ts ? h('div', { class: 'dim', text: `Last success ${tsLocal(p.last_success_ts)} (${p.last_latency_ms} ms)` }) : '') },
    { h: '', f: p => h('button', { class: 'btn sm', text: 'Test connection', disabled: p.state === 'missing_key', title: p.state === 'missing_key' ? `Set ${p.key_env} in .env` : '', onclick: async ev => { ev.target.disabled = true; ev.target.textContent = 'Testing…'; try { const r = await api('/ai/test', { provider: p.id }); st.prepend(note(r.ok ? 'ok' : 'err', `${p.label}: ${r.ok ? 'verified (' + r.latency_ms + ' ms)' : r.kind + ' — ' + r.message}`, r.available_models && r.available_models.length ? h('details', {}, h('summary', { text: `Models this key can use (${r.available_models.length})` }), h('div', { class: 'num', style: 'text-align:left;white-space:normal', text: r.available_models.join(', ') })) : '')); } catch (e) { st.prepend(errBox(e)); } status(); } }) }], providers)); if (!providers.some(p => p.state !== 'missing_key')) st.prepend(note('', 'No AI provider is configured. The app works fully without one. For a free option, create a Gemini key in Google AI Studio and set GOOGLE_API_KEY, then press Test connection.')); } catch (e) { st.replaceChildren(errBox(e)); } }
  status();
  async function gen(force) {
    out.replaceChildren(loading('Asking configured providers (bounded by a global time budget)…'));
    try { const r = await api('/ai/research', { key: S.fkey, force }); out.replaceChildren(...renderResearch(r)); } catch (e) { out.replaceChildren(errBox(e)); status(); }
  }
};
function renderResearch(r) {
  if (r.status !== 'ok') return [note('warn', h('b', { text: 'No commentary generated. ' }), `${r.reason}. ${r.note || ''}`), table([{ h: 'Provider', f: a => a.provider }, { h: 'Model', f: a => a.model || '' }, { h: 'Outcome', f: a => a.outcome }, { h: 'Detail', f: a => a.message || '' }], r.attempts)];
  const c = r.commentary, facts = r.fact_pack.facts, fl = ids => ids.length ? h('span', { class: 'tag ok', title: ids.map(i => `${i}: ${facts[i].label} = ${facts[i].value}`).join('\n'), text: 'grounded: ' + ids.join(', ') }) : h('span', { class: 'tag no', text: 'no supporting fact cited' });
  const list = (t, items) => items.length ? [h('h2', { text: t, style: 'margin-top:12px' }), h('ul', {}, items.map(i => h('li', {}, i.text, ' ', fl(i.fact_ids))))] : [];
  return [h('div', { class: 'row' }, h('span', { class: 'tag', text: `AI-generated · ${r.provider} / ${r.model}` }), h('span', { class: 'dim', text: `${tsLocal(r.generated_ts)} · ${r.latency_ms} ms${r.from_cache ? ' · from cache' : ''}` })),
    h('div', { class: 'cmt' }, h('p', { text: c.summary }), c.model_disagreement ? h('p', {}, h('b', { text: 'Model disagreement: ' }), c.model_disagreement) : '', list('Observations (from supplied facts)', c.observations), list('Risk factors', c.risk_factors), list('Hypotheses (not facts)', c.hypotheses),
      c.news.length ? [h('h2', { text: 'News summaries (generated; read the originals)', style: 'margin-top:12px' }), h('ul', {}, c.news.map(n => { const it = r.news_items[Number(n.news_id.slice(1)) - 1]; return h('li', {}, n.summary, ' ', it && safeUrl(it.url) ? h('a', { href: safeUrl(it.url), target: '_blank', rel: 'noopener noreferrer', text: `[${it.publisher}: original]` }) : ''); }))] : [],
      list('', []), h('h2', { text: 'Limitations & missing inputs', style: 'margin-top:12px' }), h('ul', {}, [...c.limitations, ...c.missing_information.map(t => 'Missing: ' + t)].map(t => h('li', { text: t })))),
    c.unverified_numbers.length ? note('warn', h('b', { text: 'Numbers in this text that do not match any supplied fact: ' }), c.unverified_numbers.join(', '), '. Do not rely on them.') : note('ok', 'Every number in the prose matches a supplied fact (within rounding).'),
    c.dropped_citations.length ? note('warn', 'Citations to unknown fact ids were discarded: ' + c.dropped_citations.join(', ')) : '',
    h('details', {}, h('summary', { text: 'Fact pack sent to the model' }), h('pre', { class: 'num', style: 'text-align:left;white-space:pre-wrap;font-size:11px', text: JSON.stringify(r.fact_pack, null, 1) })), h('div', { class: 'dim', text: r.disclaimer })];
}

VIEWS.portfolio = async root => {
  root.append(h('h1', {}, 'Portfolio & Paper Trading ', h('span', { class: 'sim', text: 'SIMULATED' })), h('p', { class: 'sub', text: 'A local paper portfolio. Balances here are a simulation parameter you set, never a real account. Orders are priced from a freshly retrieved quote.' }));
  const body = h('div', {}); root.append(body); await drawPortfolio(body);
};
async function drawPortfolio(body) {
  body.replaceChildren(loading());
  try {
    const p = await api('/portfolio'); body.replaceChildren();
    const msg = h('div', {});
    body.append(bento(
      h('div', { class: 'c12 grid g3' }, metric('Paper cash', nf(p.cash)), metric('Paper equity (marked)', nf(p.equity_estimate)), metric('Trades today', p.trades_today)),
      panel('Paper positions', p.positions.length ? table([{ h: 'Symbol', f: x => x.symbol }, { h: 'Qty', num: 1, f: x => x.quantity }, { h: 'Avg entry', num: 1, f: x => nf(x.avg_entry) }, { h: 'Price', num: 1, f: x => nf(x.price) }, { h: 'Value', num: 1, f: x => nf(x.market_value) },
        { h: 'Unrealized', num: 1, cls: x => x.unrealized_pnl > 0 ? 'up' : x.unrealized_pnl < 0 ? 'dn' : '', f: x => x.unrealized_pnl == null ? '—' : (x.unrealized_pnl > 0 ? '▲ ' : '▼ ') + nf(x.unrealized_pnl) }, { h: 'Quote', f: x => badge({ freshness: x.quote_freshness }) }], p.positions) : empty('No paper positions', 'Place a simulated order below.'), `Retrieved ${tsLocal(p.retrieved_ts)}. ${p.equity_note}`, null, 'c12'),
      panel('Simulated order', [h('form', { class: 'row', onsubmit: async e => { e.preventDefault(); const f = e.target;
        try { const r = await api('/paper/order', { symbol: f.sym.value, side: f.side.value, quantity: +f.qty.value, confirm: f.cf.checked }); msg.replaceChildren(note('ok', `${r.result.message} Price ${nf(r.price_used)} from ${r.quote_provider} (${r.quote_freshness}, source time ${tsLocal(r.quote_source_ts)}).`)); drawPortfolio(body.parentNode ? body : body); } catch (er) { msg.replaceChildren(errBox(er)); } } },
        h('div', { class: 'fld' }, h('label', { for: 'osym', text: 'Symbol' }), h('input', { class: 'inp', id: 'osym', name: 'sym', value: S.symbol, required: true })),
        h('div', { class: 'fld' }, h('label', { for: 'oside', text: 'Side' }), h('select', { class: 'sel', id: 'oside', name: 'side' }, h('option', { value: 'BUY', text: 'Buy' }), h('option', { value: 'SELL', text: 'Sell' }))),
        h('div', { class: 'fld' }, h('label', { for: 'oqty', text: 'Quantity (whole shares)' }), h('input', { class: 'inp', id: 'oqty', name: 'qty', type: 'number', min: 1, step: 1, required: true, style: 'width:110px' })),
        h('label', { class: 'row', style: 'align-items:center;gap:4px' }, h('input', { type: 'checkbox', name: 'cf', required: true }), 'I understand this is a simulated order'), h('button', { class: 'btn pri', type: 'submit', text: 'Place simulated order' })), msg],
        'Passes the same risk limits as every other order. Refused if no usable quote.', null, 'c7'),
      panel('Set starting balance', h('form', { class: 'row', onsubmit: async e => { e.preventDefault(); if (!confirm('This deletes the current paper portfolio and starts over. Continue?')) return; try { await api('/paper/reset', { starting_cash: +e.target.cash.value, confirm: true }); drawPortfolio(body); } catch (er) { msg.replaceChildren(errBox(er)); } } },
        h('div', { class: 'fld' }, h('label', { for: 'cash', text: 'Starting cash (simulation parameter)' }), h('input', { class: 'inp', id: 'cash', name: 'cash', type: 'number', min: 1, value: p.starting_cash_config })), h('button', { class: 'btn', type: 'submit', text: 'Reset paper portfolio' })), 'Resetting clears simulated holdings only.', null, 'c5'),
      h('div', { class: 'c12' }, note('', h('b', { text: 'Real accounts: ' }), p.live_account.message))
    ));
  } catch (e) { body.replaceChildren(errBox(e)); }
}

VIEWS.risk = async root => {
  root.append(h('h1', { text: 'Risk & Exposure' }), h('p', { class: 'sub', text: 'Exposure of the paper portfolio, the limits every order must pass, and a position-size calculator.' }));
  const body = h('div', {}); root.append(body); body.append(loading());
  try {
    const r = await api('/risk'); body.replaceChildren();
    const res = h('div', {});
    body.append(bento(
      h('div', { class: 'c12 grid g3' }, metric('Equity (paper)', nf(r.equity_estimate)), metric('Cash weight', r.cash_weight_pct == null ? '—' : nf(r.cash_weight_pct, 1) + '%'), metric('Largest position', nf(r.largest_position_pct, 1) + '%')),
      panel('Exposure by position', r.positions.length ? table([{ h: 'Symbol', f: p => p.symbol }, { h: 'Weight', f: p => h('div', { style: 'min-width:120px' }, h('div', { class: 'bar' }, h('i', { style: `width:${Math.min(100, p.weight_pct || 0)}%` })), nf(p.weight_pct, 1) + '%') }, { h: 'Vol (ann.)', num: 1, f: p => p.vol_annualized_pct == null ? 'n/a' : nf(p.vol_annualized_pct, 1) + '%' }, { h: 'Max DD 1y', num: 1, f: p => p.max_drawdown_1y_pct == null ? 'n/a' : nf(p.max_drawdown_1y_pct, 1) + '%' }], r.positions) : empty('No exposure', 'The paper portfolio is all cash.'), r.note, null, 'c7'),
      panel('Active limits', h('dl', { class: 'kv' }, Object.entries(r.limits).flatMap(([k, v]) => [h('dt', { text: k.replace(/_/g, ' ') }), h('dd', { text: v == null ? 'disabled' : String(v) })]), h('dt', { text: 'trades today' }), h('dd', { text: r.trades_today })), 'Configured in .env; enforced server-side on every order.', null, 'c5'),
      panel('Position-size calculator', [h('form', { class: 'row', onsubmit: async e => { e.preventDefault(); const f = e.target; try { const z = await api(`/risk/size?symbol=${encodeURIComponent(f.sym.value)}&risk_pct=${f.rp.value}&stop_pct=${f.sp.value}`);
        res.replaceChildren(note('ok', `Suggested quantity ${z.suggested_quantity} at ${nf(z.price)} (quote ${z.quote_freshness}). By risk: ${z.quantity_by_risk}; cap from limits/cash: ${z.quantity_cap_by_limits}.`), h('div', { class: 'dim', text: z.formula })); } catch (er) { res.replaceChildren(errBox(er)); } } },
        h('div', { class: 'fld' }, h('label', { for: 'rs', text: 'Symbol' }), h('input', { class: 'inp', id: 'rs', name: 'sym', value: S.symbol, required: true })), h('div', { class: 'fld' }, h('label', { for: 'rp', text: 'Risk per trade, % of equity' }), h('input', { class: 'inp', id: 'rp', name: 'rp', value: 1, style: 'width:90px' })),
        h('div', { class: 'fld' }, h('label', { for: 'sp', text: 'Stop distance, %' }), h('input', { class: 'inp', id: 'sp', name: 'sp', value: 5, style: 'width:90px' })), h('button', { class: 'btn', type: 'submit', text: 'Calculate' })), res], null, null, 'c12')
    ));
  } catch (e) { body.replaceChildren(errBox(e)); }
};

VIEWS.health = async root => {
  root.append(h('h1', { text: 'Data & Model Health' }), h('p', { class: 'sub', text: 'Measured, not claimed: provider outcomes, latencies, caches, background jobs and saved models.' }));
  const body = h('div', {}); root.append(body, h('div', { style: 'margin-top:10px' }, h('button', { class: 'btn', text: 'Refresh', onclick: () => render() }))); body.append(loading());
  try {
    const d = await api('/health'), m = d.market_data; body.replaceChildren();
    const lat = m.measured_latency_ms; const L = k => lat[k].samples ? `median ${lat[k].median} ms · p95 ${lat[k].p95} ms (n=${lat[k].samples})` : 'no samples yet';
    body.append(bento(
      panel('Market-data providers', table([{ h: 'Provider', f: p => p.provider }, { h: 'Configured', f: p => p.configured ? 'yes' : 'no' }, { h: 'Real-time entitlement', f: p => p.supplies_realtime ? 'yes (per credential)' : 'no — never labelled Live' }, { h: 'Coverage', f: p => Array.isArray(p.covers_exchanges) ? p.covers_exchanges.join(', ') : p.covers_exchanges },
        { h: 'OK / fail', num: 1, f: p => `${p.successes} / ${p.failures}` }, { h: 'Circuit', f: p => p.circuit_open ? h('span', { class: 'dn', text: 'open (cooling down)' }) : 'closed' }, { h: 'Last error', f: p => p.last_error ? h('span', { class: 'dn', text: p.last_error }) : '—' }], m.providers), `Measured latency — quotes: ${L('quote')}; history: ${L('history')}`, null, 'c12'),
      panel('Caches & coalescing', h('dl', { class: 'kv' }, ...['quote', 'history'].flatMap(k => [h('dt', { text: `${k} cache` }), h('dd', { text: `${m.caches[k].entries}/${m.caches[k].maxsize} entries · ${m.caches[k].hits} hits · ${m.caches[k].misses} misses` })]), h('dt', { text: 'Coalesced upstream requests' }), h('dd', { text: m.coalesced_requests }), h('dt', { text: 'Cached forecasts' }), h('dd', { text: d.forecast_cache_entries }), h('dt', { text: 'News feeds configured' }), h('dd', { text: d.news_feeds_configured })), null, null, 'c6'),
      panel('AI providers', table([{ h: 'Provider', f: p => p.label }, { h: 'State', f: p => p.state.replace(/_/g, ' ') }, { h: 'Calls / fails', num: 1, f: p => `${p.calls} / ${p.failures}` }, { h: 'Cooling down', f: p => p.cooling_down ? 'yes' : 'no' }], d.ai), null, null, 'c6'),
      panel('Measured performance (this server process only)', Object.keys(d.performance).length ? table([{ h: 'Operation', f: r => r.k }, { h: 'Samples', num: 1, f: r => r.v.samples }, { h: 'Median', num: 1, f: r => nf(r.v.median_ms, 1) + ' ms' }, { h: 'p95', num: 1, f: r => nf(r.v.p95_ms, 1) + ' ms' }, { h: 'Max', num: 1, f: r => nf(r.v.max_ms, 1) + ' ms' }], Object.entries(d.performance).map(([k, v]) => ({ k, v })), { label: 'Measured latencies' }) : empty('Nothing measured yet', 'Latencies appear here after the operations run. No value is pre-filled.'), 'Wall-clock times recorded by the app itself (quote/history upstream calls, feature engineering, model train/load/inference, AI requests, backtests, API endpoints).', null, 'c12'),
      panel('Provider verification records (real requests only)', (d.verification.records.length ? table([{ h: 'Provider', f: r => r.provider }, { h: 'Endpoint class', f: r => r.endpoint_class }, { h: 'Instrument', f: r => r.instrument || '—' }, { h: 'Outcome', f: r => h('span', { class: 'tag ' + (r.outcome === 'PASS' ? 'ok' : r.outcome === 'FAIL' ? 'no' : '') , text: r.outcome }) }, { h: 'UTC time', f: r => r.utc }, { h: 'Evidence / detail', f: r => r.detail || (r.evidence && r.evidence.freshness_category ? `freshness: ${r.evidence.freshness_category}` : r.evidence && r.evidence.rows ? `${r.evidence.rows} bars to ${r.evidence.last_bar}` : r.evidence && r.evidence.model ? `model ${r.evidence.model}` : '') }], d.verification.records)
        : empty('Nothing has been verified against a real provider', 'Run  python scripts/verify_providers.py  on your machine with your credentials. Until then every provider is UNVERIFIED.')), d.verification.note, null, 'c12'),
      panel('Background jobs', d.jobs.length ? table([{ h: 'Job', f: j => j.id }, { h: 'Status', f: j => j.status }, { h: 'Stage', f: j => j.stage }, { h: 'Timings (ms)', f: j => Object.entries(j.timings || {}).map(([a, b]) => `${a}:${b}`).join(' ') }, { h: 'Error', f: j => j.error || '' }], d.jobs) : empty('No jobs yet', 'Run a forecast to see training timings.'), null, null, 'c6'),
      panel('Saved LSTM checkpoints', d.checkpoints.length ? table([{ h: 'Name', f: c => c.name }, { h: 'Ticker', f: c => c.ticker }, { h: 'Target', f: c => c.lstm_prediction_target }, { h: 'Lookback', num: 1, f: c => c.lookback }, { h: 'Saved', f: c => tsLocal(c.saved_at) }], d.checkpoints) : empty('No checkpoints', 'Created automatically after the first forecast per instrument.'), null, null, 'c6')
    ));
  } catch (e) { body.replaceChildren(errBox(e)); }
};

VIEWS.settings = async root => {
  root.append(h('h1', { text: 'Settings' }), h('p', { class: 'sub', text: 'Server settings and secrets live in .env and need a restart. Nothing secret is shown or stored in the browser; the preferences below are non-secret and stay in this browser.' }));
  const pr = panel('Browser preferences', h('form', { class: 'row', onsubmit: e => { e.preventDefault(); prefs.refresh = +e.target.rf.value; prefs.period = e.target.pp.value; store.set('ws.prefs', prefs); $('#refresh').value = prefs.refresh; schedule(); setTheme(e.target.th.value); e.target.after(note('ok', 'Saved in this browser.')); } },
    h('div', { class: 'fld' }, h('label', { for: 'th', text: 'Color theme' }), h('select', { class: 'sel', id: 'th', name: 'th' }, [['auto', 'Auto (system)'], ['dark', 'Dark (graphite)'], ['light', 'Light (paper)']].map(([v, t]) => h('option', { value: v, selected: getTheme() === v, text: t })))),
    h('div', { class: 'fld' }, h('label', { for: 'rf', text: 'Quote auto-refresh' }), h('select', { class: 'sel', id: 'rf', name: 'rf' }, [[0, 'Off (manual)'], [15, '15 s'], [30, '30 s'], [60, '1 min'], [300, '5 min']].map(([v, t]) => h('option', { value: v, selected: Number(prefs.refresh) === v, text: t })))),
    h('div', { class: 'fld' }, h('label', { for: 'pp', text: 'Default chart range' }), h('select', { class: 'sel', id: 'pp', name: 'pp' }, ['1mo', '3mo', '6mo', '1y', '2y', '5y'].map(p => h('option', { value: p, selected: p === prefs.period, text: p })))), h('button', { class: 'btn', type: 'submit', text: 'Save' })), 'Auto-refresh only re-fetches quotes; it never retrains models and pauses while the tab is hidden.', null, 'c12');
  const body = h('div', { class: 'c12' }); root.append(bento(pr, body));
  try {
    const s = await api('/settings'), kv = o => h('dl', { class: 'kv' }, Object.entries(o).flatMap(([k, v]) => [h('dt', { text: k.replace(/_/g, ' ') }), h('dd', { text: v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v) })]));
    body.append(bento(panel('Data sources', kv(s.data), null, null, 'c6'), panel('Models', kv(s.models), null, null, 'c6'), panel('Risk & trading', kv({ ...s.risk, ...s.trading }), 'Paper is the default. Live trading needs ENABLE_LIVE_TRADING=true AND a 32+ character WEBAPP_LIVE_TRADING_TOKEN, and is available only through the gated classic flow.', null, 'c6'),
      panel('AI provider readiness', table([{ h: 'Provider', f: p => p.id }, { h: 'Key variable', f: p => p.key_env }, { h: 'Key set', f: p => p.key_configured ? 'yes' : 'no' }, { h: 'Model variable', f: p => p.model_env }, { h: 'Model', f: p => p.model || 'not set' }], s.ai.providers), 'Key set ≠ working. Use Test connection in AI Research.', null, 'c6'), h('div', { class: 'c12' }, note('', `${s.editable_in} News feeds: ${s.news.feeds_configured} configured (${s.news.env}).`))));
  } catch (e) { body.append(errBox(e)); }
};

/* ---------------------------------------------------------------- router */
const cur = () => (location.hash.replace(/^#\//, '') || 'overview').split('?')[0];
function render() {
  const name = VIEWS[cur()] ? cur() : 'overview';
  document.querySelectorAll('.rail nav a').forEach(a => a.getAttribute('href') === '#/' + name ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current'));
  view.replaceChildren(); document.title = `${name[0].toUpperCase()}${name.slice(1)} · Quant Workstation`;
  VIEWS[name](view).catch(e => view.append(errBox(e)));
}
window.addEventListener('hashchange', () => { render(); window.scrollTo(0, 0); view.focus({ preventScroll: true }); });
$('#symform').addEventListener('submit', e => { e.preventDefault(); const v = $('#sym').value.trim(); if (v) setSymbol(v); });
document.querySelectorAll('#top-pills .pill').forEach(b => {
  b.addEventListener('click', () => setSymbol(b.dataset.sym));
});
$('#refresh').value = prefs.refresh; $('#refresh').addEventListener('change', e => { prefs.refresh = +e.target.value; store.set('ws.prefs', prefs); schedule(); });
$('#refresh-now').addEventListener('click', () => { loadQuote(true); if (cur() === 'overview') loadWatchQuotes(); });
const themeSel = $('#theme-sel');
if (themeSel) {
  themeSel.value = getTheme();
  themeSel.addEventListener('change', e => setTheme(e.target.value));
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (getTheme() === 'auto') applyTheme('auto');
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.nextRefresh && Date.now() > S.nextRefresh) loadQuote(true); });
$('#sym').value = S.symbol; renderStrip(); updateTopPills(); if (S.symbol) loadQuote(); render();
(async () => {   // restore this tab's last forecast after a reload; the server cache may have expired -> silently start fresh
  try { const saved = JSON.parse(sessionStorage.getItem('ws.fkey') || 'null');
    if (saved && saved.symbol === S.symbol && saved.key) { S.forecast = await api(`/forecast/result?key=${encodeURIComponent(saved.key)}`); S.fkey = saved.key; if (cur() !== 'overview') render(); } } catch { try { sessionStorage.removeItem('ws.fkey'); } catch { /* ignore */ } }
})();
})();
