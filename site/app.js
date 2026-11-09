/* Marine Plastics Research Monitor: reads the JSON of site/data (written by pipeline/export_site.py and
   pipeline/newsletter.py) and draws the three tabs. No build step; ECharts comes from the CDN. */
'use strict';

const LEVELS = ['coarse', 'mid', 'fine'];
const LEVEL_NAMES = { coarse: 'Broad', mid: 'Medium', fine: 'Detailed' };
const FIRST_YEAR_SHOWN = 2000;

const state = { level: 'coarse', selected: null, basin: null, timeMode: 'count', sunRoot: null, sunTop: null };
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
/** The subtopics of topic `id` of `level` that split it: the children, skipping levels where a topic has one child
    only (the same papers), and none when it is never split. */
function subtopics(level, id) {
  let l = level, kids = [topic(level, id)];
  while (kids.length === 1 && LEVELS.indexOf(l) < LEVELS.length - 1) {
    kids = children(l, kids[0].id);
    l = LEVELS[LEVELS.indexOf(l) + 1];
  }
  return { level: l, kids: kids.length > 1 ? kids : [] };
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
  const [meta, taxonomy, coarse, mid, fine, basins, map, geoBasins, land, issues, papers] = await Promise.all([
    getJSON('data/meta.json'), getJSON('data/taxonomy.json'), getJSON('data/levels/coarse.json'),
    getJSON('data/levels/mid.json'), getJSON('data/levels/fine.json'), getJSON('data/basins.json'),
    getJSON('data/map.json'), getJSON('data/geo/basins.geojson'), getJSON('data/geo/land.geojson'),
    getJSON('data/newsletter/index.json', []), getJSON('data/papers.json', null),
  ]);
  Object.assign(D, { meta, taxonomy, basins, map, issues, papers });
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
    [m.week ? fmt(m.n_week) : '–', m.week ? `added in the week of <span class="nowrap">${m.week}</span>` : 'added this week'],
  ].map(([v, l]) => `<div class="tile"><div class="value">${v}</div><div class="label">${l}</div></div>`).join('');
  document.querySelectorAll('.min-age').forEach((e) => { e.textContent = m.thresholds.min_citation_age; });
  document.querySelectorAll('.min-gini').forEach((e) => { e.textContent = m.thresholds.min_gini_papers; });
  const values = { n_papers: fmt(m.n_papers), encoder: m.quality.encoder, labels_model: m.labels_model,
    outlier_share: pct(m.outlier_share), no_basin_share: pct(1 - located), search_model: m.search_model };
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
    $('archive').innerHTML = '<li class="secondary">None yet</li>';
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
      <div class="secondary">Week of ${esc(issue.date)} · ${fmt(issue.n_papers)} new papers</div>
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
  setupSearch();
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

// ---------------------------------------------------------------------------------------------- explore: search
/* Topics are ranked by the cosine similarity of the query's embedding with the topics' (data/search.json, written by
   export_site.py), computed in the browser by transformers.js, plus a bonus for the query's words in the topic's text.
   The model loads on the first focus of the search box; until it has (or if it fails), only word matches are shown. */
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
const SEARCH_RESULTS = 8;
const WORD_BONUS = 0.1;  // added to the similarity per share of the query's words found in the topic's name
const search = { index: null, embed: null, loading: null, failed: false, results: [], query: null, active: -1, ticket: 0,
  semantic: false, timer: null };

const words = (s) => String(s).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];

function loadSearch() {
  if (!search.loading) {
    search.loading = (async () => {
      const s = await getJSON('data/search.json');
      const bytes = Uint8Array.from(atob(s.vectors), (c) => c.charCodeAt(0));
      const all = new Int8Array(bytes.buffer);
      search.index = s.topics.map(([level, id], i) => {
        const v = all.subarray(i * s.dim, (i + 1) * s.dim);
        return { level, id, v, norm: Math.hypot(...v) };
      });
      const { pipeline, env } = await import(TRANSFORMERS);
      env.allowLocalModels = false;
      const extractor = await pipeline('feature-extraction', s.model, { dtype: 'q8' });
      search.embed = async (q) => (await extractor(s.query_prefix + q, { pooling: 'cls', normalize: true })).data;
    })().catch((e) => { search.failed = true; console.warn('Topic search: ranking by meaning unavailable', e); })
      .finally(() => { if ($('topic-search').value.trim()) runSearch(); });
  }
  return search.loading;
}

/** Share of the query's words found in the topic: a whole word (or a word starting with a query word of 3+ letters)
    of its name counts 1, of its description or keywords 0.5. */
function wordMatch(query, t) {
  const name = words(t.label);
  const rest = words(`${t.description} ${t.keywords}`);
  const hit = (ws, w) => ws.some((x) => x === w || (w.length >= 3 && x.startsWith(w)));
  return query.length ? sum(query.map((w) => (hit(name, w) ? 1 : hit(rest, w) ? 0.5 : 0))) / query.length : 0;
}

async function runSearch() {
  const text = $('topic-search').value.trim();
  const ticket = ++search.ticket;
  if (!text) { search.results = []; search.query = text; renderSearch(); return; }
  const query = words(text);
  const candidates = search.index || LEVELS.flatMap((level) => topicsAt(level).map((t) => ({ level, id: t.id })));
  let q = null;
  if (search.embed) {
    try { q = await search.embed(text); } catch (e) { console.warn('Topic search', e); }
  }
  if (ticket !== search.ticket) return;  // a newer search has started
  let rows = candidates.map((c) => {
    const match = wordMatch(query, topic(c.level, c.id));
    if (!q) return { ...c, match, score: match };
    let dot = 0;
    for (let k = 0; k < q.length; k++) dot += q[k] * c.v[k];
    return { ...c, match, score: dot / c.norm + WORD_BONUS * match };
  });
  if (!q) rows = rows.filter((r) => r.match > 0);
  rows.sort((a, b) => b.score - a.score || topic(b.level, b.id).n - topic(a.level, a.id).n);
  const seen = new Set();  // a name repeated at two levels: keep the better ranked
  search.results = rows.filter((r) => !seen.has(label(r.level, r.id)) && seen.add(label(r.level, r.id))).slice(0, SEARCH_RESULTS);
  search.semantic = q != null;
  search.query = text;
  search.active = search.results.length ? 0 : -1;
  renderSearch();
}

function renderSearch() {
  const input = $('topic-search');
  const open = document.activeElement === input && input.value.trim() !== '';
  $('search-panel').hidden = !open;
  input.setAttribute('aria-expanded', String(open));
  if (!open) { input.removeAttribute('aria-activedescendant'); return; }
  const n = search.results.length;
  const loading = !search.embed && !search.failed;
  $('search-status').textContent = search.semantic ? 'Topics closest in meaning to your search'
    : loading ? (n ? 'Topics using these words. Loading the search by meaning…' : 'Loading the search by meaning (about 35 MB, only the first time)…')
      : (n ? 'Topics using these words' : 'No topic uses these words');
  $('search-results').innerHTML = search.results.map((r, i) => {
    const t = topic(r.level, r.id);
    const broad = r.level === 'coarse' ? null : label('coarse', ancestor(r.level, r.id, 'coarse'));
    return `<li role="option" id="search-option-${i}" data-i="${i}" aria-selected="${i === search.active}">
      <div class="name">${esc(t.label)}</div>
      <div class="meta">${LEVEL_NAMES[r.level]} topic · ${fmt(t.n)} papers${broad && broad !== t.label ? ` · in ${esc(broad)}` : ''}</div>
      <div class="desc">${esc(t.description)}</div></li>`;
  }).join('');
  if (search.active >= 0) {
    input.setAttribute('aria-activedescendant', `search-option-${search.active}`);
    $(`search-option-${search.active}`).scrollIntoView({ block: 'nearest' });
  } else input.removeAttribute('aria-activedescendant');
}

function chooseResult(i) {
  const r = search.results[i];
  if (!r) return;
  $('topic-search').blur();
  renderSearch();
  state.sunRoot = null;
  select(r.level, r.id);
}

/** Enter: the best result of the query as typed, without waiting for the search that runs after typing pauses, once
    the search by meaning has loaded (or failed to). */
async function submitSearch() {
  clearTimeout(search.timer);
  const text = $('topic-search').value.trim();
  if (!text) return;
  if (search.query !== text) await runSearch();
  if (!search.embed && !search.failed) {  // word matches are a stopgap: wait for the search by meaning
    await loadSearch();
    await runSearch();
  }
  if ($('topic-search').value.trim() === text) chooseResult(search.active);  // unless typing went on meanwhile
}

function setupSearch() {
  const input = $('topic-search');
  input.addEventListener('focus', () => { loadSearch(); renderSearch(); });
  input.addEventListener('input', () => { loadSearch(); clearTimeout(search.timer); search.timer = setTimeout(runSearch, 120); });
  input.addEventListener('blur', () => setTimeout(renderSearch, 0));
  input.addEventListener('keydown', (e) => {
    const n = search.results.length;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!n) return;
      e.preventDefault();
      search.active = (search.active + (e.key === 'ArrowDown' ? 1 : n - 1)) % n;
      renderSearch();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      submitSearch();
    } else if (e.key === 'Escape' && !$('search-panel').hidden) {
      e.preventDefault();  // close, keeping the text (the browser's default clears it)
      input.blur();
    }
  });
  // mousedown, not click: it comes before the input's blur closes the panel
  $('search-results').addEventListener('mousedown', (e) => {
    const li = e.target.closest('li[data-i]');
    if (!li) return;
    e.preventDefault();
    chooseResult(Number(li.dataset.i));
  });
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
    axisLabel: { color: css('--ink-2') }, splitLine: { lineStyle: { color: css('--grid') } }, ...extra };
}
/** Colour of a mark of topic `id`: the accent when selected, or when nothing is selected and `alone`. */
function markColor(id, alone = false) {
  if (state.selected == null) return alone ? css('--accent') : css('--context');
  return id === state.selected ? css('--accent') : css('--context');
}
const trunc = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/* The sunburst shows one ring, the topics of the chosen level, until a topic with subtopics is clicked: that topic
   then fills the centre, with its subtopics (and theirs) as the rings around it. `state.sunRoot` is the topic in the
   centre (`<level>:<id>`), `state.sunTop` the level of the ring it was opened from. */
function renderSunburst() {
  const depthColor = [css('--seq-300'), css('--seq-200'), css('--seq-100')];
  // A selection made elsewhere, outside the opened topic, closes it
  if (state.sunRoot) {
    const [rootLevel, rootId] = state.sunRoot.split(':');
    const inside = state.selected != null && LEVELS.indexOf(state.level) >= LEVELS.indexOf(rootLevel)
      && ancestor(state.level, state.selected, rootLevel) === Number(rootId);
    if (!inside) state.sunRoot = null;
  }
  const ids = {};  // the selected topic and its ancestors
  if (state.selected != null) LEVELS.slice(0, LEVELS.indexOf(state.level) + 1).forEach((l) => { ids[l] = ancestor(state.level, state.selected, l); });
  const inSelection = (level, id) => state.selected != null && LEVELS.indexOf(level) > LEVELS.indexOf(state.level)
    && ancestor(level, id, state.level) === state.selected;
  const node = (level, t, open) => {
    const selected = level === state.level && t.id === state.selected;
    const sub = open ? subtopics(level, t.id) : { kids: [] };
    const kids = sub.kids;
    return {
      id: `${level}:${t.id}`, name: t.label, level, topicId: t.id, description: t.description, n: t.n,
      value: kids.length ? undefined : t.n,
      itemStyle: { color: selected ? css('--accent') : depthColor[LEVELS.indexOf(level)], borderColor: css('--surface'), borderWidth: 1.5,
        opacity: state.selected == null || ids[level] === t.id || inSelection(level, t.id) ? 1 : 0.75 },
      label: { color: selected ? '#fff' : css('--ink') },
      children: kids.map((k) => node(sub.level, k, true)),
    };
  };
  let data, rings;  // rings: [r0, r] in % of the radius, one per depth
  if (state.sunRoot) {
    const [rootLevel, rootId] = state.sunRoot.split(':');
    data = [node(rootLevel, topic(rootLevel, Number(rootId)), true)];
    const height = (d) => 1 + Math.max(0, ...d.children.map(height));
    const depths = height(data[0]) - 1;
    const step = (97 - 26) / depths;
    rings = [[0, 26], ...Array.from({ length: depths }, (_, i) => [26 + step * i, 26 + step * (i + 1)])];
  } else {
    data = topicsAt(state.level).map((t) => node(state.level, t, false));
    rings = [[24, 97]];
  }
  const el = $('sunburst');
  const R = Math.min(el.clientWidth, el.clientHeight) / 2;
  const levels = [{}, ...rings.map(([r0, r], i) => {
    const centre = r0 === 0;
    const inner = (R * (r0 + (r - r0) / 4)) / 100;
    return {
      r0: `${r0}%`, r: `${r}%`,
      label: centre
        ? { rotate: 0, width: (R * r * 1.7) / 100, overflow: 'break', fontSize: 13, fontWeight: 600, minAngle: 0 }
        // Radial labels as long as the ring is wide, shown where the segment is tall enough for a line of text
        // (measured a quarter into the ring, where long labels get crowded)
        : { rotate: 'radial', width: (R * (r - r0)) / 100 - 8, minAngle: (14 / inner) * (180 / Math.PI) },
    };
  })];
  // Segments wide enough for more than one line of text wrap their label (up to three lines) instead of truncating it
  const total = sum(data.map((d) => d.n));
  const wrap = (d, depth) => {
    const [r0, r] = rings[depth - 1];
    const lines = Math.min(3, Math.floor(((2 * Math.PI * d.n) / total) * ((R * (r0 + (r - r0) / 4)) / 100) / 13));
    if (r0 > 0 && lines >= 2) d.label = { ...d.label, overflow: 'break', lineHeight: 13, height: lines * 13, lineOverflow: 'truncate' };
    d.children.forEach((k) => wrap(k, depth + 1));
  };
  data.forEach((d) => wrap(d, 1));
  const c = chart('sunburst');
  c.setOption({
    title: state.sunRoot ? undefined : {
      text: String(topicsAt(state.level).length), subtext: `${LEVEL_NAMES[state.level].toLowerCase()} topics`,
      left: 'center', top: 'middle', itemGap: 2,
      textStyle: { color: css('--ink'), fontSize: 22, fontWeight: 600 }, subtextStyle: { color: css('--ink-2'), fontSize: 12 },
    },
    tooltip: tooltip({ formatter: (p) => p.data && p.data.n ? `<b>${esc(p.data.name)}</b><br>${fmt(p.data.n)} papers · ${LEVEL_NAMES[p.data.level]} topic<br><span style="color:${css('--ink-2')}">${esc(p.data.description)}</span>${p.data.id === state.sunRoot ? `<br><i>Click to go back</i>` : ''}` : '' }),
    series: [{
      type: 'sunburst', data, sort: null, nodeClick: false, radius: ['0%', '97%'], emphasis: { focus: 'ancestor' },
      label: { fontSize: 11, overflow: 'truncate' }, levels,
    }],
  }, true);
  c.off('click');
  c.on('click', (p) => {
    const d = p.data;
    if (!d || d.topicId == null) return;
    if (d.id === state.sunRoot) {  // the centre: back to the ring it was opened from, or up one level
      if (d.level === state.sunTop) { state.sunRoot = null; select(d.level, d.topicId); return; }
      const up = LEVELS[LEVELS.indexOf(d.level) - 1];
      const parent = ancestor(d.level, d.topicId, up);
      state.sunRoot = `${up}:${parent}`;
      select(up, parent);
      return;
    }
    if (subtopics(d.level, d.topicId).kids.length) {  // open a topic with subtopics
      if (!state.sunRoot) state.sunTop = d.level;
      state.sunRoot = d.id;
    }
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

/** The topic's most cited and newest papers (data/papers.json), as two linked lists; empty without the file. */
function topicPapers(level, id) {
  const lists = D.papers && D.papers.topics[level][id];
  if (!lists) return '';
  const item = (i, meta) => {
    const [title, journal, year, url, rate, added] = D.papers.papers[i];
    const details = [journal, year, meta(rate, added)].filter(Boolean).map(esc).join(' · ');
    return `<li><a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a><div class="meta">${details}</div></li>`;
  };
  const list = (title, ids, meta) => (ids.length ? `<h3>${title}</h3>
      <ol class="paper-list">${ids.map((i) => item(i, meta)).join('')}</ol>` : '');
  return `<div class="topic-papers">
    ${list('Most cited', lists.cited, (rate) => `${rate.toFixed(1)} citations a year`)}
    ${list('New papers', lists.new, (rate, added) => (added ? `added ${added}` : ''))}
  </div>`;
}

function renderTopicCard() {
  const card = $('topic-card');
  const level = state.level;
  if (state.selected == null) {
    const top = topicsAt(level).slice(0, 12);
    card.innerHTML = `<h2>${LEVEL_NAMES[level]} topics</h2>
      <p class="secondary">${topicsAt(level).length} topics at this level of detail.</p>
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
      ${topicPapers(level, state.selected)}
      <p style="margin-top:12px"><button class="linkish" id="clear-topic">Clear the selection</button></p>`;
    $('clear-topic').addEventListener('click', () => { state.sunRoot = null; select(level, null); });
  }
  // The panel scrolls on its own: back to its top when another topic is shown, not on other re-renders
  const shown = `${level}:${state.selected}`;
  if (card.dataset.shown !== shown) { card.scrollTop = 0; card.dataset.shown = shown; }
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
    yAxis: axisStyle({ type: 'value', axisLabel: { color: css('--ink-2'), formatter: (v) => (state.timeMode === 'share' ? pct(v) : fmt(v)) } }),
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
    xAxis: axisStyle({ type: 'value', name: 'citations per year', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: css('--ink-2') } }),
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
    lineStyle: { width: 1, color: css('--axis') },
    endLabel: { show: true, formatter: 'equality', color: css('--ink-2'), fontSize: 11 } });
  const c = chart('lorenz');
  c.setOption({
    animation: false, textStyle: baseText(),
    grid: { left: 48, right: 56, top: 16, bottom: 44 },
    tooltip: tooltip({ trigger: 'item', formatter: (p) => `<b>${esc(p.seriesName)}</b><br>The least-cited ${pct(p.value[0])} of papers get ${pct(p.value[1])} of the citations` }),
    xAxis: axisStyle({ type: 'value', min: 0, max: 1, name: 'share of papers, least cited first', nameLocation: 'middle', nameGap: 28,
      nameTextStyle: { color: css('--ink-2') }, axisLabel: { color: css('--ink-2'), formatter: (v) => pct(v) } }),
    yAxis: axisStyle({ type: 'value', min: 0, max: 1, axisLabel: { color: css('--ink-2'), formatter: (v) => pct(v) } }),
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
    xAxis: axisStyle({ type: 'value', min: 'dataMin', max: 'dataMax', minInterval: 1, splitLine: { show: false }, axisLabel: { color: css('--ink-2'), formatter: (v) => String(v) } }),
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
      <p class="secondary" style="font-size:.8rem">* Papers published in ${rows.length ? rows[0].period : ''}. ${fmt(D.taxonomy.outliers)} papers fit no topic and are not listed.</p>`;
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
  const name = ['week', 'explore', 'methods', 'about'].includes(tab) ? tab : 'week';
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
  $('theme-toggle').innerHTML = `<span class="theme-word">Theme: </span>${theme}`;
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
