/* Microplastics Research Monitor: reads the JSON of site/data (written by pipeline/export_site.py and
   pipeline/newsletter.py) and draws the three tabs. No build step; ECharts comes from the CDN. */
'use strict';

const LEVELS = ['coarse', 'mid', 'fine'];
const LEVEL_NAMES = { coarse: 'Broad', mid: 'Medium', fine: 'Detailed' };
const FIRST_YEAR_SHOWN = 2000;

const state = { level: 'coarse', selected: null, basin: null, timeMode: 'count', sunRoot: null };
const D = {};
const charts = {};
let exploreReady = false;

// ---------------------------------------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (n) => (n == null ? '–' : Number(n).toLocaleString('en'));
const pct = (x, digits = 0) => (x == null || !isFinite(x) ? '–' : `${(100 * x).toFixed(digits)}%`);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sum = (a) => a.reduce((x, y) => x + y, 0);
const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

async function getJSON(path, fallback) {
  try {
    const response = await fetch(path, { cache: 'no-cache' });
    if (!response.ok) throw new Error(response.status);
    return await response.json();
  } catch (e) {
    if (fallback !== undefined) return fallback;
    throw e;
  }
}

function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (e) { /* private mode: no persistence */ }
  return null;
}

/** Minimal Markdown: paragraphs, [links](url) and *emphasis*, on escaped text. */
function markdown(text) {
  return String(text || '').split(/\n\s*\n/).filter((p) => p.trim()).map((p) => {
    let html = esc(p.trim());
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (m, t, url) => `<a href="${url}" target="_blank" rel="noopener">${t}</a>`);
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return `<p>${html}</p>`;
  }).join('');
}

// ---------------------------------------------------------------------------------------------- taxonomy
function topicsAt(level) { return D.taxonomy.levels[level]; }
function topic(level, id) { return D.topicIndex[level].get(id); }
function label(level, id) { const t = topic(level, id); return t ? t.label : 'No clear topic'; }
/** The topic at `level` of a fine topic. */
function fineTo(level, fine) { return fine === -1 ? -1 : D.parents.get(fine)[LEVELS.indexOf(level)]; }
/** The ancestor at `to` (a coarser level) of topic `id` of level `from`. */
function ancestor(from, id, to) {
  if (from === to) return id;
  for (const [fine, row] of D.parents) if (row[LEVELS.indexOf(from)] === id) return row[LEVELS.indexOf(to)];
  return null;
}
/** Topics of the level below `level` whose parent is `id`. */
function children(level, id) {
  const below = LEVELS[LEVELS.indexOf(level) + 1];
  if (!below) return [];
  const ids = new Set();
  for (const row of D.parents.values()) if (row[LEVELS.indexOf(level)] === id) ids.add(row[LEVELS.indexOf(below)]);
  return topicsAt(below).filter((t) => ids.has(t.id));
}
function levelData() { return D.levels[state.level]; }
function rowOf(id) { return levelData().topics.indexOf(id); }

function select(level, id) {
  state.level = level;
  state.selected = id;
  renderExplore();
}

// ---------------------------------------------------------------------------------------------- loading
async function load() {
  const [meta, taxonomy, coarse, mid, fine, basins, map, geoBasins, land, issues] = await Promise.all([
    getJSON('data/meta.json'), getJSON('data/taxonomy.json'), getJSON('data/levels/coarse.json'),
    getJSON('data/levels/mid.json'), getJSON('data/levels/fine.json'), getJSON('data/basins.json'),
    getJSON('data/map.json'), getJSON('data/geo/basins.geojson'), getJSON('data/geo/land.geojson'),
    getJSON('data/newsletter/index.json', []),
  ]);
  Object.assign(D, { meta, taxonomy, basins, map, issues });
  D.levels = { coarse, mid, fine };
  D.topicIndex = Object.fromEntries(LEVELS.map((l) => [l, new Map(taxonomy.levels[l].map((t) => [t.id, t]))]));
  D.parents = new Map(taxonomy.parents.map(([f, m, c]) => [f, [c, m, f]]));  // indexed like LEVELS
  D.totalByYear = basins.years.map((_, j) => sum(basins.basins.map((b) => b.by_year[j])));
  echarts.registerMap('oceans', { type: 'FeatureCollection', features: geoBasins.features.concat(land.features) });
}

// ---------------------------------------------------------------------------------------------- header
function renderHeader() {
  const m = D.meta;
  const located = m.n_located / m.n_papers;
  $('tiles').innerHTML = [
    [fmt(m.n_papers), 'papers reviewed'],
    [m.week ? fmt(m.n_week) : '–', m.week ? `added in the week of ${m.week}` : 'added this week'],
    [`${m.n_topics.coarse} · ${m.n_topics.mid} · ${m.n_topics.fine}`, 'research topics (broad · medium · detailed)'],
    [pct(located), 'located in an ocean basin'],
  ].map(([v, l]) => `<div class="tile"><div class="value">${v}</div><div class="label">${l}</div></div>`).join('');
  document.querySelectorAll('.min-age').forEach((e) => { e.textContent = m.thresholds.min_citation_age; });
  document.querySelectorAll('.min-gini').forEach((e) => { e.textContent = m.thresholds.min_gini_papers; });
  const values = { n_papers: fmt(m.n_papers), encoder: m.quality.encoder, labels_model: m.labels_model,
    outlier_share: pct(m.outlier_share), no_basin_share: pct(1 - located) };
  document.querySelectorAll('[data-meta]').forEach((e) => { e.textContent = values[e.dataset.meta] ?? ''; });
  $('footer').innerHTML = `Data updated ${esc(m.updated)} · topics version ${esc(m.version)} · paper data from Scopus.
    Topic names and the newsletter are AI-generated.`;
}

// ---------------------------------------------------------------------------------------------- newsletter
async function renderWeek() {
  const issues = D.issues;
  if (!issues.length) {
    $('issue').innerHTML = `<div class="card empty"><h2>No issue yet</h2>
      <p>The first weekly issue will appear after the next update, which runs every Monday.</p></div>`;
    $('archive').innerHTML = '<li class="muted">None yet</li>';
    return;
  }
  const wanted = (location.hash.match(/^#week\/(\d{4}-\d{2}-\d{2})$/) || [])[1];
  const date = issues.some((i) => i.date === wanted) ? wanted : issues[0].date;
  $('archive').innerHTML = issues.map((i) => `<li><button class="linkish" data-issue="${i.date}">${esc(i.date)}</button>
      <div class="secondary" style="font-size:.85rem">${esc(i.title)} · ${fmt(i.n_papers)} papers</div></li>`).join('');
  $('archive').querySelectorAll('[data-issue]').forEach((b) => b.addEventListener('click', () => { location.hash = `#week/${b.dataset.issue}`; }));
  const issue = await getJSON(`data/newsletter/${date}.json`);
  const sections = issue.sections.map((s) => `<h3>${esc(s.heading)}</h3>
      <div class="topic-tag">Broad topic: <button class="linkish" data-topic="${s.topic}">${esc(s.label)}</button> · ${fmt(s.n_papers)} papers</div>
      ${markdown(s.text)}`).join('');
  const also = issue.also && issue.also.trim() ? `<h3>Also this week</h3><div class="topic-tag">${fmt(issue.n_also)} papers in other topics</div>${markdown(issue.also)}` : '';
  $('issue').innerHTML = `<article class="issue">
      <div class="muted">Week of ${esc(issue.date)} · ${fmt(issue.n_papers)} new papers</div>
      <h2>${esc(issue.title)}</h2>${markdown(issue.overview)}${sections}${also}</article>`;
  $('issue').querySelectorAll('[data-topic]').forEach((b) => b.addEventListener('click', () => {
    state.sunRoot = null;
    location.hash = `#explore/coarse/${b.dataset.topic}`;
  }));
}

// ---------------------------------------------------------------------------------------------- explore: controls
function renderControls() {
  document.querySelectorAll('#level-switch button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.level === state.level)));
  const options = topicsAt(state.level).map((t) => `<option value="${t.id}"${t.id === state.selected ? ' selected' : ''}>${esc(t.label)} (${fmt(t.n)})</option>`);
  $('topic-select').innerHTML = `<option value=""${state.selected == null ? ' selected' : ''}>All topics: select one…</option>${options.join('')}`;
}

function setupControls() {
  document.querySelectorAll('#level-switch button').forEach((b) => b.addEventListener('click', () => {
    const to = b.dataset.level;
    const coarser = LEVELS.indexOf(to) < LEVELS.indexOf(state.level);
    state.selected = state.selected != null && coarser ? ancestor(state.level, state.selected, to) : null;
    state.level = to;
    state.sunRoot = null;
    renderExplore();
  }));
  $('topic-select').addEventListener('change', (e) => {
    state.selected = e.target.value === '' ? null : Number(e.target.value);
    renderExplore();
  });
  document.querySelectorAll('#time-mode button').forEach((b) => b.addEventListener('click', () => {
    state.timeMode = b.dataset.mode;
    document.querySelectorAll('#time-mode button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    renderOverTime();
  }));
}

// ---------------------------------------------------------------------------------------------- explore: charts
function chart(id) {
  if (!charts[id]) charts[id] = echarts.init($(id), null, { renderer: 'canvas' });
  return charts[id];
}

function baseText() { return { color: css('--ink-2'), fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', fontSize: 12 }; }
function tooltip(extra) {
  return { backgroundColor: css('--surface'), borderColor: css('--border'), textStyle: { color: css('--ink'), fontSize: 12 },
    extraCssText: 'max-width:320px;white-space:normal;box-shadow:0 2px 8px rgba(0,0,0,.12);', confine: true, ...extra };
}
function axisStyle(extra) {
  return { axisLine: { lineStyle: { color: css('--axis') } }, axisTick: { show: false },
    axisLabel: { color: css('--muted') }, splitLine: { lineStyle: { color: css('--grid') } }, ...extra };
}
/** Colour of a mark of topic `id`: the accent when selected, or when nothing is selected and `alone`. */
function markColor(id, alone = false) {
  if (state.selected == null) return alone ? css('--accent') : css('--context');
  return id === state.selected ? css('--accent') : css('--context');
}
const trunc = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function renderSunburst() {
  const depthColor = [css('--seq-300'), css('--seq-200'), css('--seq-100')];
  const ids = { coarse: null, mid: null };
  if (state.selected != null) {
    ids.coarse = ancestor(state.level, state.selected, 'coarse');
    if (state.level !== 'coarse') ids.mid = ancestor(state.level, state.selected, 'mid');
  }
  const node = (level, t, depth) => {
    const selected = level === state.level && t.id === state.selected;
    const onPath = (level === 'coarse' && t.id === ids.coarse) || (level === 'mid' && t.id === ids.mid);
    const kids = children(level, t.id);
    return {
      id: `${level}:${t.id}`, name: t.label, level, topicId: t.id, description: t.description, n: t.n,
      value: kids.length ? undefined : t.n,
      itemStyle: { color: selected ? css('--accent') : depthColor[depth], borderColor: css('--surface'), borderWidth: 1.5,
        opacity: state.selected == null || selected || onPath ? 1 : 0.75 },
      label: { color: selected ? '#fff' : css('--ink') },
      children: kids.map((k) => node(LEVELS[depth + 1], k, depth + 1)),
    };
  };
  const narrow = $('sunburst').clientWidth < 480;
  const c = chart('sunburst');
  c.setOption({
    tooltip: tooltip({ formatter: (p) => p.data && p.data.n ? `<b>${esc(p.data.name)}</b><br>${fmt(p.data.n)} papers · ${LEVEL_NAMES[p.data.level]} topic<br><span style="color:${css('--ink-2')}">${esc(p.data.description)}</span>` : '' }),
    series: [{
      type: 'sunburst', data: topicsAt('coarse').map((t) => node('coarse', t, 0)), sort: null, nodeClick: false,
      radius: ['0%', '96%'], emphasis: { focus: 'ancestor' },
      label: { fontSize: 11, minAngle: 9, overflow: 'truncate' },
      levels: [{}, { r0: '10%', r: '52%', label: { rotate: 'radial', width: narrow ? 60 : 92, minAngle: narrow ? 12 : 7 } },
        { r0: '52%', r: '78%', label: { show: !narrow, rotate: 'radial', width: 58, minAngle: 9 } },
        { r0: '78%', r: '97%', label: { show: false } }],
    }],
  }, true);
  if (state.sunRoot) c.dispatchAction({ type: 'sunburstRootToNode', targetNode: state.sunRoot });
  c.off('click');
  c.on('click', (p) => {
    const d = p.data;
    if (!d || d.topicId == null) return;
    if (d.id === state.sunRoot) {  // the centre: zoom back out one level
      const up = LEVELS.indexOf(d.level) - 1;
      if (up < 0) { state.sunRoot = null; select(d.level, d.topicId); return; }
      const parent = ancestor(d.level, d.topicId, LEVELS[up]);
      state.sunRoot = up > 0 ? `${LEVELS[up]}:${parent}` : null;
      select(LEVELS[up], parent);
      return;
    }
    if (d.children && d.children.length) state.sunRoot = d.id;
    select(d.level, d.topicId);
  });
}

function topicStats(id) {
  const L = levelData();
  const i = rowOf(id);
  const years = L.years;
  const series = L.over_time[i];
  const last = years.length - 1;
  const recent = sum(series.slice(last - 2, last + 1));
  const before = sum(series.slice(last - 5, last - 2));
  const basins = L.basin_counts[i];
  const n = sum(basins);
  const noBasin = basins[L.basins.indexOf('No basin')] || 0;
  const k = basins.slice(0, -1).reduce((best, v, j) => (v > basins[best] ? j : best), 0);
  return { n, growth: before ? recent / before - 1 : null, period: `${years[last - 2]}–${years[last]} vs ${years[last - 5]}–${years[last - 3]}`,
    located: n ? 1 - noBasin / n : null, topBasin: L.basins[k], median: L.median_rate[i], gini: L.gini_topic[i] };
}

function renderTopicCard() {
  const card = $('topic-card');
  const level = state.level;
  if (state.selected == null) {
    const top = topicsAt(level).slice(0, 12);
    card.innerHTML = `<h2>${LEVEL_NAMES[level]} topics</h2>
      <p class="secondary">${topicsAt(level).length} topics at this level of detail. Select one in the chart, the list above or
        below to see its numbers and highlight it in every chart.</p>
      <h3>Largest topics</h3><div class="chips">${top.map((t) => `<button class="chip" data-topic="${t.id}">${esc(t.label)}</button>`).join('')}</div>`;
  } else {
    const t = topic(level, state.selected);
    const s = topicStats(state.selected);
    // Ancestors, without those that are the same topic under the same name (unmerged at that level)
    const path = LEVELS.slice(0, LEVELS.indexOf(level)).map((l) => [l, ancestor(level, state.selected, l)])
      .filter(([l, a]) => label(l, a) !== t.label)
      .map(([l, a]) => `<button class="linkish" data-level="${l}" data-topic="${a}">${esc(label(l, a))}</button>`);
    const kids = children(level, state.selected);
    card.innerHTML = `${path.length ? `<div class="path">${path.join(' › ')} ›</div>` : ''}
      <h2>${esc(t.label)}</h2><p>${esc(t.description)}</p>
      <div class="stats">
        <div class="stat"><div class="value">${fmt(s.n)}</div><div class="label">papers (${pct(s.n / D.meta.n_papers, 1)} of all)</div></div>
        <div class="stat"><div class="value">${s.growth == null ? '–' : `${s.growth >= 0 ? '+' : ''}${pct(s.growth)}`}</div><div class="label">papers ${s.period}</div></div>
        <div class="stat"><div class="value">${pct(s.located)}</div><div class="label">located in a basin; most in the ${esc(s.topBasin)}</div></div>
        <div class="stat"><div class="value">${s.median == null ? '–' : s.median.toFixed(1)}</div><div class="label">median citations per year</div></div>
      </div>
      <div class="keywords">Characteristic words: ${esc(t.keywords)}</div>
      ${kids.length > 1 ? `<h3 style="margin-top:12px">Subtopics</h3><div class="chips">${kids.map((k) => `<button class="chip" data-level="${LEVELS[LEVELS.indexOf(level) + 1]}" data-topic="${k.id}">${esc(k.label)}</button>`).join('')}</div>` : ''}
      <p style="margin-top:12px"><button class="linkish" id="clear-topic">Clear the selection</button></p>`;
    $('clear-topic').addEventListener('click', () => { state.sunRoot = null; select(level, null); });
  }
  card.querySelectorAll('[data-topic]').forEach((b) => b.addEventListener('click', () => {
    const l = b.dataset.level || level;
    state.sunRoot = null;
    select(l, Number(b.dataset.topic));
  }));
}

function renderPaperMap() {
  const { x, y, topic: fine } = D.map;
  const level = state.level;
  const context = [];
  const chosen = [];
  const byTopic = new Map();
  for (let i = 0; i < x.length; i++) {
    const t = fineTo(level, fine[i]);
    (t === state.selected ? chosen : context).push([x[i], y[i], t]);
    if (t !== -1) {
      if (!byTopic.has(t)) byTopic.set(t, [[], []]);
      byTopic.get(t)[0].push(x[i]); byTopic.get(t)[1].push(y[i]);
    }
  }
  const labels = topicsAt(level).filter((t) => byTopic.has(t.id)).map((t) => {
    const [xs, ys] = byTopic.get(t.id);
    return { value: [median(xs), median(ys), t.id], name: t.label };
  });
  const c = chart('paper-map');
  c.setOption({
    animation: false,
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { show: false, scale: true }, yAxis: { show: false, scale: true },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => esc(label(level, p.value[2])) }),
    series: [
      { type: 'scatter', data: context, symbolSize: 3, progressive: 0, itemStyle: { color: css('--context'), opacity: state.selected == null ? 0.7 : 0.45 }, emphasis: { scale: 2 } },
      { type: 'scatter', data: chosen, symbolSize: 4, progressive: 0, zlevel: 1, itemStyle: { color: css('--accent'), opacity: 0.9 }, emphasis: { scale: 2 } },
      { type: 'scatter', data: labels, symbolSize: 0, silent: true, zlevel: 2,
        label: { show: true, formatter: (p) => trunc(p.name, 32), color: css('--ink'), fontSize: 11, fontWeight: 600,
          textBorderColor: css('--surface'), textBorderWidth: 3 },
        labelLayout: { hideOverlap: true } },
    ],
  }, true);
  c.off('click');
  c.on('click', (p) => { if (p.seriesIndex < 2 && p.value[2] !== -1) select(level, p.value[2]); });
}

function renderOverTime() {
  const L = levelData();
  const narrow = $('over-time').clientWidth < 560;  // no room for an end label: the topic is named above
  const start = L.years.indexOf(FIRST_YEAR_SHOWN) >= 0 ? L.years.indexOf(FIRST_YEAR_SHOWN) : 0;
  const years = L.years.slice(start);
  const totals = D.basins.years.map((year, j) => [year, D.totalByYear[j]]);
  const total = new Map(totals);
  const value = (row) => row.slice(start).map((v, j) => (state.timeMode === 'share' ? (total.get(years[j]) ? v / total.get(years[j]) : null) : v));
  const series = L.topics.map((id, i) => {
    const on = id === state.selected;
    return { type: 'line', name: label(state.level, id), data: value(L.over_time[i]), symbol: 'none', smooth: false,
      lineStyle: { width: on ? 2.5 : 1, color: markColor(id) }, itemStyle: { color: markColor(id) }, z: on ? 10 : 2,
      emphasis: { focus: 'series', lineStyle: { width: 2.5 } },
      endLabel: { show: on && !narrow, formatter: () => trunc(label(state.level, id), 28), color: css('--ink'), fontWeight: 600 } };
  });
  const c = chart('over-time');
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 48, right: state.selected != null && !narrow ? 170 : 16, top: 16, bottom: 28 },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(p.seriesName)}</b><br>${p.name}: ${state.timeMode === 'share' ? pct(p.value, 1) : `${fmt(p.value)} papers`}` }),
    xAxis: axisStyle({ type: 'category', data: years, boundaryGap: false, splitLine: { show: false } }),
    yAxis: axisStyle({ type: 'value', axisLabel: { color: css('--muted'), formatter: (v) => (state.timeMode === 'share' ? pct(v) : fmt(v)) } }),
    series,
  }, true);
  c.off('click');
  c.on('click', (p) => select(state.level, L.topics[p.seriesIndex]));
}

function renderBasinMap() {
  const L = levelData();
  const i = state.selected == null ? -1 : rowOf(state.selected);
  const counts = new Map(L.basins.map((b, k) => [b, i < 0 ? sum(L.basin_counts.map((r) => r[k])) : L.basin_counts[i][k]]));
  const located = L.basins.filter((b) => b !== 'No basin');
  const total = sum([...counts.values()]);
  $('no-basin-note').textContent = `${pct((counts.get('No basin') || 0) / total)} of these papers name no ocean place and are not on the map.`;
  const data = located.map((b) => ({ name: b, value: counts.get(b) }));
  const max = Math.max(1, ...data.map((d) => d.value));
  const c = chart('basin-map');
  c.setOption({
    animation: false,
    tooltip: tooltip({ trigger: 'item', formatter: (p) => (p.name === 'Land' || p.value == null || isNaN(p.value) ? '' : `<b>${esc(p.name)}</b><br>${fmt(p.value)} papers`) }),
    visualMap: { type: 'continuous', min: 0, max, seriesIndex: 0, orient: 'horizontal', left: 'center', bottom: 0, itemHeight: 160,
      text: [fmt(max), '0'], textStyle: { color: css('--ink-2') }, calculable: false,
      inRange: { color: [css('--seq-100'), css('--seq-300'), css('--seq-500'), css('--seq-700')] } },
    geo: { map: 'oceans', roam: false, top: 8, bottom: 44, left: 0, right: 0, aspectScale: 1,
      itemStyle: { areaColor: css('--land'), borderColor: css('--surface'), borderWidth: 0.6 },
      emphasis: { label: { show: false }, itemStyle: { areaColor: css('--accent') } }, select: { disabled: true },
      regions: [{ name: 'Land', silent: true, itemStyle: { areaColor: css('--land'), borderColor: css('--land') }, emphasis: { disabled: true } },
        ...(state.basin ? [{ name: state.basin, itemStyle: { borderColor: css('--ink'), borderWidth: 1.5 } }] : [])] },
    series: [{ type: 'map', geoIndex: 0, data }],
  }, true);
  c.off('click');
  c.on('click', (p) => { if (p.name && p.name !== 'Land') { state.basin = p.name; renderBasinMap(); renderBasinTopics(); } });
}

function renderBasinTopics() {
  const L = levelData();
  const located = L.basins.filter((b) => b !== 'No basin');
  if (!state.basin || !located.includes(state.basin)) state.basin = located[0];
  const k = L.basins.indexOf(state.basin);
  const rows = L.topics.map((id, i) => ({ id, n: L.basin_counts[i][k] })).filter((r) => r.n > 0).sort((a, b) => b.n - a.n).slice(0, 12).reverse();
  const basinTotal = sum(L.basin_counts.map((r) => r[k]));
  $('basin-topics-title').textContent = `Topics in the ${state.basin}`;
  const c = chart('basin-topics');
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 8, right: 24, top: 8, bottom: 24, containLabel: true },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(label(state.level, p.data.id))}</b><br>${fmt(p.value)} papers (${pct(p.value / basinTotal)} of the basin's)` }),
    xAxis: axisStyle({ type: 'value' }),
    yAxis: axisStyle({ type: 'category', data: rows.map((r) => trunc(label(state.level, r.id), 34)), splitLine: { show: false }, axisLabel: { color: css('--ink-2') } }),
    series: [{ type: 'bar', barMaxWidth: 16, data: rows.map((r) => ({ value: r.n, id: r.id, itemStyle: { color: markColor(r.id, true), borderRadius: [0, 4, 4, 0] } })) }],
  }, true);
  c.off('click');
  c.on('click', (p) => select(state.level, p.data.id));
}

function renderHeatmap() {
  const L = levelData();
  const topics = [...L.topics].reverse();  // largest on top
  const data = [];
  topics.forEach((id, r) => {
    const i = rowOf(id);
    const n = sum(L.basin_counts[i]);
    L.basins.forEach((b, k) => { const v = L.basin_counts[i][k]; data.push([k, r, Math.log10(1 + v), v, n]); });
  });
  const el = $('heatmap');
  el.style.height = `${Math.max(320, topics.length * 20 + 150)}px`;
  el.style.minWidth = '760px';
  const c = chart('heatmap');
  c.resize();
  const maxLog = Math.max(...data.map((d) => d[2]));
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 8, right: 16, top: 8, bottom: 8, containLabel: true },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(label(state.level, topics[p.value[1]]))}</b><br>${esc(L.basins[p.value[0]])}: ${fmt(p.value[3])} papers (${pct(p.value[3] / p.value[4])} of the topic's)` }),
    xAxis: axisStyle({ type: 'category', data: L.basins, position: 'top', splitLine: { show: false }, axisLabel: { color: css('--ink-2'), rotate: 40, interval: 0 } }),
    yAxis: axisStyle({ type: 'category', data: topics.map((id) => id), splitLine: { show: false },
      axisLabel: { interval: 0, formatter: (id) => (Number(id) === state.selected ? `{sel|${trunc(label(state.level, Number(id)), 40)}}` : trunc(label(state.level, Number(id)), 40)),
        color: css('--ink-2'), rich: { sel: { color: css('--accent-ink'), fontWeight: 700 } } } }),
    visualMap: { show: false, min: 0, max: maxLog, dimension: 2, inRange: { color: [css('--surface'), css('--seq-200'), css('--seq-400'), css('--seq-600'), css('--seq-700')] } },
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: css('--surface'), borderWidth: 1 } }],
  }, true);
  c.off('click');
  c.on('click', (p) => select(state.level, topics[p.value[1]]));
}

function renderImpact() {
  const L = levelData();
  const rows = L.topics.map((id, i) => ({ id, v: L.median_rate[i], n: L.rated[i] })).filter((r) => r.v != null).sort((a, b) => a.v - b.v);
  const el = $('impact');
  el.style.height = `${Math.max(420, rows.length * 18 + 60)}px`;
  const c = chart('impact');
  c.resize();
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 8, right: 24, top: 8, bottom: 24, containLabel: true },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(label(state.level, p.data.id))}</b><br>Median ${p.value.toFixed(2)} citations per year (${fmt(p.data.n)} papers)` }),
    xAxis: axisStyle({ type: 'value', name: 'citations per year', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: css('--muted') } }),
    yAxis: axisStyle({ type: 'category', data: rows.map((r) => trunc(label(state.level, r.id), 34)), splitLine: { show: false }, axisLabel: { color: css('--ink-2'), interval: 0 } }),
    series: [{ type: 'bar', barMaxWidth: 14, data: rows.map((r) => ({ value: r.v, id: r.id, n: r.n, itemStyle: { color: markColor(r.id, true), borderRadius: [0, 4, 4, 0] } })) }],
  }, true);
  c.off('click');
  c.on('click', (p) => select(state.level, p.data.id));
}

function renderLorenz() {
  const L = levelData();
  const xs = L.lorenz_all.map((_, j) => j / (L.lorenz_all.length - 1));
  const series = L.topics.map((id, i) => ({ id, curve: L.lorenz[i] })).filter((r) => r.curve).map((r) => {
    const on = r.id === state.selected;
    return { type: 'line', name: label(state.level, r.id), topicId: r.id, data: r.curve.map((y, j) => [xs[j], y]), symbol: 'none',
      lineStyle: { width: on ? 2.5 : 1, color: markColor(r.id) }, z: on ? 10 : 2, emphasis: { focus: 'series' } };
  });
  series.push({ type: 'line', name: 'Perfect equality', data: [[0, 0], [1, 1]], symbol: 'none', silent: true,
    lineStyle: { width: 1, color: css('--muted') },
    endLabel: { show: true, formatter: 'equality', color: css('--muted'), fontSize: 11 } });
  const c = chart('lorenz');
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 48, right: 56, top: 16, bottom: 44 },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(p.seriesName)}</b><br>The least-cited ${pct(p.value[0])} of papers get ${pct(p.value[1])} of the citations` }),
    xAxis: axisStyle({ type: 'value', min: 0, max: 1, name: 'share of papers, least cited first', nameLocation: 'middle', nameGap: 28,
      nameTextStyle: { color: css('--muted') }, axisLabel: { color: css('--muted'), formatter: (v) => pct(v) } }),
    yAxis: axisStyle({ type: 'value', min: 0, max: 1, axisLabel: { color: css('--muted'), formatter: (v) => pct(v) } }),
    series,
  }, true);
  c.off('click');
  c.on('click', (p) => { if (p.seriesName !== 'Perfect equality') select(state.level, series[p.seriesIndex].topicId); });
}

function renderGini() {
  const L = levelData();
  const byTopic = new Map();
  for (const [t, year, n, g] of L.gini) {
    if (g == null) continue;
    if (!byTopic.has(t)) byTopic.set(t, []);
    byTopic.get(t).push([year, g, n]);
  }
  const ids = [...byTopic.keys()];
  const series = ids.map((id) => {
    const on = id === state.selected;
    return { type: 'line', name: label(state.level, id), topicId: id, data: byTopic.get(id), symbol: 'circle', symbolSize: on ? 6 : 3, connectNulls: false,
      lineStyle: { width: on ? 2.5 : 1, color: markColor(id) }, itemStyle: { color: markColor(id) }, z: on ? 10 : 2, emphasis: { focus: 'series' } };
  });
  const c = chart('gini');
  const note = state.selected != null && !byTopic.has(state.selected)
    ? `The selected topic has no year with ${D.meta.thresholds.min_gini_papers} or more cited papers.` : '';
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 48, right: 16, top: 28, bottom: 28 },
    title: { text: note, left: 'center', top: 0, textStyle: { fontSize: 12, fontWeight: 'normal', color: css('--ink-2') } },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(p.seriesName)}</b><br>${p.value[0]}: Gini ${p.value[1].toFixed(2)} (${fmt(p.value[2])} papers)` }),
    xAxis: axisStyle({ type: 'value', min: 'dataMin', max: 'dataMax', minInterval: 1, splitLine: { show: false }, axisLabel: { color: css('--muted'), formatter: (v) => String(v) } }),
    yAxis: axisStyle({ type: 'value', min: 0, max: 1 }),
    series,
  }, true);
  c.off('click');
  c.on('click', (p) => select(state.level, series[p.seriesIndex].topicId));
}

function renderTable() {
  const L = levelData();
  const rows = L.topics.map((id) => ({ id, ...topicStats(id) }));
  $('topic-table').innerHTML = `<table><thead><tr><th>Topic</th><th class="num">Papers</th><th class="num">Growth*</th>
      <th class="num">Located</th><th class="num">Median citations / year</th><th class="num">Gini</th></tr></thead><tbody>
      ${rows.map((r) => `<tr data-topic="${r.id}"${r.id === state.selected ? ' class="selected"' : ''}><td><button class="linkish" data-topic="${r.id}">${esc(label(state.level, r.id))}</button></td>
        <td class="num">${fmt(r.n)}</td><td class="num">${r.growth == null ? '–' : `${r.growth >= 0 ? '+' : ''}${pct(r.growth)}`}</td>
        <td class="num">${pct(r.located)}</td><td class="num">${r.median == null ? '–' : r.median.toFixed(2)}</td>
        <td class="num">${r.gini == null ? '–' : r.gini.toFixed(2)}</td></tr>`).join('')}</tbody></table>
      <p class="muted" style="font-size:.8rem">* Papers published in ${rows.length ? rows[0].period : ''}. ${fmt(D.taxonomy.outliers)} papers fit no topic and are not listed.</p>`;
  $('topic-table').querySelectorAll('button[data-topic]').forEach((b) => b.addEventListener('click', () => {
    select(state.level, Number(b.dataset.topic));
    $('topic-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));
}

function renderExplore() {
  if (!exploreReady) return;
  renderControls();
  renderSunburst();
  renderTopicCard();
  renderPaperMap();
  renderOverTime();
  renderBasinMap();
  renderBasinTopics();
  renderHeatmap();
  renderImpact();
  renderLorenz();
  renderGini();
  renderTable();
  // Keep the selection in the address, so that it can be shared (without a hashchange event)
  const hash = `#explore/${state.level}${state.selected == null ? '' : `/${state.selected}`}`;
  if (location.hash !== hash && location.hash.startsWith('#explore')) history.replaceState(null, '', hash);
}

// ---------------------------------------------------------------------------------------------- methods
function renderMethods() {
  const m = D.meta;
  const q = m.quality;
  const split = q.split_half || {};
  const fit = q.fit || {};
  const lastRun = m.runs.length ? m.runs[m.runs.length - 1] : null;
  $('quality').innerHTML = `<p>A clustering algorithm gives slightly different topics each time it runs. To make the
      topics robust, they are the consensus of ${fmt(q.runs)} runs with different random starts. Splitting those runs
      into two halves and comparing the topics each half finds measures how stable they are:</p>
    <table><tbody>
      <tr><td>Agreement between the two halves (adjusted Rand index; 1 = identical)</td><td class="num">${split.ari ?? '–'}</td></tr>
      <tr><td>Shared information between the two halves (adjusted mutual information; 1 = identical)</td><td class="num">${split.ami ?? '–'}</td></tr>
      <tr><td>Papers that fit no topic well</td><td class="num">${pct(m.outlier_share, 1)}</td></tr>
      <tr><td>Papers that keep their topic when assigned like a new paper (leave-one-out test)</td><td class="num">${pct(fit.loo_kept, 1)}</td></tr>
      <tr><td>Expected share of new papers that fit no topic</td><td class="num">${pct(fit.expected_unmatched, 1)}</td></tr>
      ${lastRun ? `<tr><td>New papers that fit no topic since this version (last update)</td><td class="num">${pct(lastRun.pool_share, 1)}</td></tr>` : ''}
    </tbody></table>
    ${m.rebaseline_due ? `<p class="notice" style="margin-top:12px">Many recent papers no longer fit the current topics: a rebuild of the topics is due and will start a new version.</p>` : ''}`;
  const cw = m.crosswalk;
  $('versions').innerHTML = `<table><thead><tr><th>Version</th><th>Date</th><th class="num">Topics</th><th class="num">Papers</th></tr></thead><tbody>
      ${m.versions.map((v) => `<tr><td>${esc(v.version)}${v.version === m.version ? ' (current)' : ''}</td><td>${esc(v.date)}</td><td class="num">${fmt(v.n_topics)}</td><td class="num">${fmt(v.n_papers)}</td></tr>`).join('')}
    </tbody></table>
    ${cw && cw.previous ? `<p class="secondary" style="font-size:.9rem">Of the ${cw.new_topics} detailed topics of ${esc(m.version)}, ${cw.matched} correspond to one of the ${cw.old_topics} topics of ${esc(cw.previous)} (they share at least 10% of their papers); the others split, merged or are new.</p>` : ''}`;
}

// ---------------------------------------------------------------------------------------------- tabs, theme
function showTab() {
  const tab = (location.hash.slice(1).split('/')[0]) || 'week';
  const name = ['week', 'explore', 'methods'].includes(tab) ? tab : 'week';
  document.querySelectorAll('section.tab').forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  document.querySelectorAll('nav.tabs a').forEach((a) => { if (a.dataset.tab === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (name === 'week') renderWeek();
  if (name === 'explore') {
    // Deep link: #explore/<level>/<topic>
    const [, level, id] = location.hash.slice(1).split('/');
    if (LEVELS.includes(level)) {
      state.level = level;
      state.selected = id !== undefined && D.topicIndex[level].has(Number(id)) ? Number(id) : null;
    }
    exploreReady = true;
    renderExplore();
    Object.values(charts).forEach((c) => c.resize());
  }
}

const THEMES = ['auto', 'light', 'dark'];
function applyTheme(theme) {
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  $('theme-toggle').textContent = `Theme: ${theme}`;
  storage('theme', theme);
  renderExplore();
}

async function main() {
  const saved = storage('theme');
  applyTheme(THEMES.includes(saved) ? saved : 'auto');
  $('theme-toggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'auto';
    applyTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => renderExplore());
  try {
    await load();
  } catch (e) {
    $('issue').innerHTML = `<div class="card empty">The data could not be loaded (${esc(e.message)}).</div>`;
    return;
  }
  renderHeader();
  renderMethods();
  setupControls();
  window.addEventListener('hashchange', showTab);
  let timer;
  window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(() => Object.values(charts).forEach((c) => c.resize()), 150); });
  showTab();
}

main();
