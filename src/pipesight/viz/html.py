"""Self-contained HTML swimlane timeline rendering for `pipesight report --html`
and `pipesight compare --html`. No external resources (system sans font only,
inline CSS/JS) -- the output is a single file, openable directly in a browser.
"""

from __future__ import annotations

import json
from typing import Any

from pipesight.analysis.idle import gpu_idle_from_samples, gpu_idle_from_spans, gpu_idle_gaps
from pipesight.analysis.stats import stage_stats, wall_clock_span
from pipesight.trace.schema import Trace


def _panel_data(trace: Trace, title: str) -> dict[str, Any]:
    spans = trace.spans
    window_start = min((s.start_ns for s in spans), default=0)
    window_end = max((s.end_ns for s in spans), default=0)
    lanes = sorted({(s.proc_id, s.thread_id) for s in spans})
    idle_gaps = gpu_idle_gaps(spans) if spans else []
    idle_report = gpu_idle_from_spans(spans) if spans else gpu_idle_from_samples(trace.samples)
    stats = stage_stats(spans)

    return {
        "title": title,
        "windowStartNs": window_start,
        "windowEndNs": window_end - window_start,
        "lanes": [{"procId": p, "threadId": t} for p, t in lanes],
        "spans": [
            {
                "name": s.name,
                "device": s.device,
                "startNs": s.start_ns - window_start,
                "durNs": s.duration_ns,
                "procId": s.proc_id,
                "threadId": s.thread_id,
                "itemId": s.item_id,
                "args": s.args,
            }
            for s in spans
        ],
        "idleGaps": [
            {"startNs": g.start_ns - window_start, "durNs": g.duration_ns} for g in idle_gaps
        ],
        "idlePct": idle_report.idle_pct,
        "wallNs": wall_clock_span(spans) if spans else idle_report.window_ns,
        "stageStats": [
            {
                "name": s.name,
                "device": s.device,
                "count": s.count,
                "totalNs": s.total_ns,
                "avgNs": s.avg_ns,
                "p95Ns": s.p95_ns,
                "wallSharePct": s.wall_share_pct,
            }
            for s in sorted(stats.values(), key=lambda s: s.total_ns, reverse=True)
        ],
    }


def render_trace_html(trace: Trace, title: str = "trace") -> str:
    return _render_page([_panel_data(trace, title)], page_title="pipesight timeline")


def render_compare_html(
    trace_a: Trace, trace_b: Trace, title_a: str = "before", title_b: str = "after"
) -> str:
    panel_a = _panel_data(trace_a, title_a)
    panel_b = _panel_data(trace_b, title_b)
    summary = _compare_summary(panel_a, panel_b)
    return _render_page([panel_a, panel_b], page_title="pipesight comparison", summary=summary)


def _compare_summary(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    wall_delta_pct = 100.0 * (b["wallNs"] - a["wallNs"]) / a["wallNs"] if a["wallNs"] else 0.0
    return {
        "wallDeltaPct": wall_delta_pct,
        "idleDeltaPp": b["idlePct"] - a["idlePct"],
        "wallA": a["wallNs"],
        "wallB": b["wallNs"],
    }


def _render_page(
    panels: list[dict[str, Any]], page_title: str, summary: dict[str, Any] | None = None
) -> str:
    data_json = json.dumps(panels)
    summary_html = _summary_html(summary) if summary else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{page_title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="viz-root">
  <header class="ps-header">
    <h1>pipesight</h1>
    <p class="ps-subtitle">{page_title}</p>
  </header>
  {summary_html}
  <div id="panels"></div>
</div>
<div id="tooltip" class="tooltip" role="tooltip"></div>
<script>
const PANELS = {data_json};
{_JS}
</script>
</body>
</html>
"""


def _summary_html(summary: dict[str, Any]) -> str:
    wall_sign = "+" if summary["wallDeltaPct"] >= 0 else ""
    idle_sign = "+" if summary["idleDeltaPp"] >= 0 else ""
    wall_good = summary["wallDeltaPct"] < 0
    idle_good = summary["idleDeltaPp"] < 0
    wall_class = "delta-good" if wall_good else "delta-bad"
    idle_class = "delta-good" if idle_good else "delta-bad"
    return f"""
  <div class="summary-row">
    <div class="stat-tile">
      <p class="stat-label">Wall-clock change</p>
      <p class="stat-value {wall_class}">{wall_sign}{summary["wallDeltaPct"]:.1f}%</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">GPU idle change</p>
      <p class="stat-value {idle_class}">{idle_sign}{summary["idleDeltaPp"]:.1f}pp</p>
    </div>
  </div>
"""


_CSS = """
:root {
  --surface-1: #fcfcfb;
  --page-plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-cpu: #2a78d6;
  --series-gpu: #1baf7a;
  --series-other: #eda100;
  --status-warning: #fab219;
  --status-good: #0ca30c;
  --status-critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --page-plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-cpu: #3987e5;
    --series-gpu: #199e70;
    --series-other: #c98500;
    --status-warning: #fab219;
    --status-good: #0ca30c;
    --status-critical: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page-plane);
  color: var(--text-primary);
}
.viz-root { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
.ps-header h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.ps-subtitle { color: var(--text-secondary); margin: 0 0 20px; font-size: 14px; }

.summary-row, .stat-row { display: flex; gap: 28px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-label {
  font-size: 11px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: 0.05em; margin: 0 0 3px;
}
.stat-value { font-size: 22px; font-weight: 600; margin: 0; }
.stat-value.delta-good { color: var(--status-good); }
.stat-value.delta-bad { color: var(--status-critical); }

.panel {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px; margin-bottom: 20px;
}
.panel-title { font-size: 15px; font-weight: 600; margin: 0 0 14px; }

.legend {
  display: flex; gap: 16px; margin-bottom: 12px; font-size: 12px; color: var(--text-secondary);
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

.timeline-scroll { overflow-x: auto; }
svg.timeline { display: block; }
.lane-label { font-size: 11px; fill: var(--text-muted); }
.axis-tick { font-size: 10px; fill: var(--text-muted); }
.gridline { stroke: var(--gridline); stroke-width: 1; }

rect.span-bar { cursor: pointer; }
rect.span-bar.device-cpu { fill: var(--series-cpu); }
rect.span-bar.device-gpu { fill: var(--series-gpu); }
rect.span-bar.device-other { fill: var(--series-other); }
rect.span-bar:hover { filter: brightness(1.12); }
rect.span-bar:focus-visible { outline: 2px solid var(--text-primary); outline-offset: 1px; }

rect.idle-gap { fill: var(--status-warning); opacity: 0.18; }

.empty-note { color: var(--text-secondary); font-size: 13px; margin: 0; }

details.stats-table { margin-top: 14px; }
details.stats-table summary {
  cursor: pointer; font-size: 12px; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.04em;
}
table.stage-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
table.stage-table th, table.stage-table td {
  text-align: right; padding: 4px 8px; font-variant-numeric: tabular-nums;
}
table.stage-table th:first-child, table.stage-table td:first-child { text-align: left; }
table.stage-table th {
  color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--gridline);
}
table.stage-table td { border-bottom: 1px solid var(--gridline); color: var(--text-primary); }

.tooltip {
  position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
  font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.18); max-width: 260px;
  z-index: 10; display: none;
}
.tooltip-name {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--text-secondary); margin: 0 0 2px;
}
.tooltip-value { font-size: 16px; font-weight: 600; color: var(--text-primary); margin: 0 0 6px; }
.tooltip-row {
  display: flex; justify-content: space-between; gap: 16px; color: var(--text-secondary);
}
.tooltip-row + .tooltip-row { margin-top: 2px; }

@media (prefers-reduced-motion: reduce) {
  rect.span-bar { transition: none; }
}
"""

_JS = r"""
const NS_PER_MS = 1e6;
const SVG_NS = 'http://www.w3.org/2000/svg';

function fmtMs(ns) {
  const ms = ns / NS_PER_MS;
  if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
  return ms.toFixed(1) + 'ms';
}

function laneKey(procId, threadId) { return procId + ':' + threadId; }

function niceStep(raw) {
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  for (const s of steps) if (s >= raw) return s;
  return steps[steps.length - 1];
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function svgEl(tag, attrs) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function buildStatsTable(panel) {
  if (!panel.stageStats.length) return null;
  const details = el('details', 'stats-table');
  const summary = el('summary');
  summary.textContent = 'Stage stats table';
  details.appendChild(summary);

  const table = el('table', 'stage-table');
  const thead = el('thead');
  const headRow = el('tr');
  for (const h of ['stage', 'device', 'count', 'total', 'avg', 'p95', 'wall %']) {
    const th = el('th');
    th.textContent = h;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  for (const s of panel.stageStats) {
    const row = el('tr');
    const cells = [
      s.name, s.device, String(s.count), fmtMs(s.totalNs), fmtMs(s.avgNs),
      fmtMs(s.p95Ns), s.wallSharePct.toFixed(1) + '%',
    ];
    for (const c of cells) {
      const td = el('td');
      td.textContent = c;
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  details.appendChild(table);
  return details;
}

function buildSvg(panel) {
  const laneIndex = {};
  const laneLabels = [];
  const hasIdle = panel.idleGaps.length > 0;
  if (hasIdle) { laneIndex['idle'] = 0; laneLabels.push('GPU idle'); }
  for (const s of panel.spans) {
    const k = laneKey(s.procId, s.threadId);
    if (!(k in laneIndex)) {
      laneIndex[k] = laneLabels.length;
      laneLabels.push('pid ' + s.procId + ' / tid ' + s.threadId);
    }
  }

  const laneHeight = 26;
  const laneGap = 6;
  const marginLeft = 150;
  const marginTop = 20;
  const marginBottom = 26;
  const pxPerMs = 0.5;
  const totalMs = Math.max(panel.windowEndNs / NS_PER_MS, 1);
  const width = marginLeft + Math.max(500, totalMs * pxPerMs) + 24;
  const height = marginTop + laneLabels.length * (laneHeight + laneGap) + marginBottom;

  const svg = svgEl('svg', {
    class: 'timeline', width, height, viewBox: `0 0 ${width} ${height}`,
    role: 'img', 'aria-label': panel.title + ' timeline',
  });

  const xScale = (ns) => marginLeft + (ns / NS_PER_MS) * pxPerMs;

  const tickStepMs = niceStep(totalMs / 8);
  for (let t = 0; t <= totalMs; t += tickStepMs) {
    const x = xScale(t * NS_PER_MS);
    svg.appendChild(svgEl('line', {
      class: 'gridline', x1: x, x2: x, y1: marginTop, y2: height - marginBottom,
    }));
    const label = svgEl('text', {
      class: 'axis-tick', x, y: height - marginBottom + 14, 'text-anchor': 'middle',
    });
    label.textContent = t + 'ms';
    svg.appendChild(label);
  }

  laneLabels.forEach((label, i) => {
    const y = marginTop + i * (laneHeight + laneGap) + laneHeight / 2 + 4;
    const text = svgEl('text', { class: 'lane-label', x: 6, y });
    text.textContent = label;
    svg.appendChild(text);
  });

  if (hasIdle) {
    const y = marginTop;
    for (const g of panel.idleGaps) {
      svg.appendChild(svgEl('rect', {
        class: 'idle-gap',
        x: xScale(g.startNs), y,
        width: Math.max(1, (g.durNs / NS_PER_MS) * pxPerMs),
        height: laneHeight,
      }));
    }
  }

  for (const s of panel.spans) {
    const laneI = laneIndex[laneKey(s.procId, s.threadId)];
    const y = marginTop + laneI * (laneHeight + laneGap);
    const x = xScale(s.startNs);
    const w = Math.max(2, (s.durNs / NS_PER_MS) * pxPerMs);

    const rect = svgEl('rect', {
      class: 'span-bar device-' + s.device,
      x, y, width: w, height: laneHeight, rx: 3, tabindex: 0,
    });
    rect.addEventListener('pointermove', (e) => showTooltip(e, s));
    rect.addEventListener('pointerleave', hideTooltip);
    rect.addEventListener('focus', (e) => showTooltip(e, s));
    rect.addEventListener('blur', hideTooltip);
    svg.appendChild(rect);
  }

  return svg;
}

function showTooltip(evt, span) {
  const tooltip = document.getElementById('tooltip');
  tooltip.style.display = 'block';
  const rect = evt.target.getBoundingClientRect
    ? evt.target.getBoundingClientRect()
    : { left: evt.clientX, top: evt.clientY, width: 0, height: 0 };
  tooltip.style.left = (rect.left + rect.width / 2 + 12) + 'px';
  tooltip.style.top = (rect.top - 8) + 'px';
  tooltip.textContent = '';

  const nameEl = el('p', 'tooltip-name');
  nameEl.textContent = span.name;
  tooltip.appendChild(nameEl);

  const valueEl = el('p', 'tooltip-value');
  valueEl.textContent = fmtMs(span.durNs);
  tooltip.appendChild(valueEl);

  const rows = [['device', span.device]];
  if (span.itemId !== null && span.itemId !== undefined) rows.push(['item', String(span.itemId)]);
  if (span.args && span.args.error) rows.push(['error', span.args.error]);
  if (span.args && span.args.timing_method) rows.push(['timing', span.args.timing_method]);

  for (const [k, v] of rows) {
    const row = el('div', 'tooltip-row');
    const kEl = document.createElement('span');
    kEl.textContent = k;
    const vEl = document.createElement('span');
    vEl.textContent = v;
    row.appendChild(kEl);
    row.appendChild(vEl);
    tooltip.appendChild(row);
  }
}

function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

function buildPanel(panel, container) {
  const section = el('section', 'panel');

  const h2 = el('h2', 'panel-title');
  h2.textContent = panel.title;
  section.appendChild(h2);

  const statRow = el('div', 'stat-row');
  const stats = [
    ['Wall time', fmtMs(panel.wallNs)],
    ['GPU idle', panel.idlePct.toFixed(1) + '%'],
    ['Spans', String(panel.spans.length)],
  ];
  for (const [label, value] of stats) {
    const tile = el('div', 'stat-tile');
    const l = el('p', 'stat-label');
    l.textContent = label;
    const v = el('p', 'stat-value');
    v.textContent = value;
    tile.appendChild(l);
    tile.appendChild(v);
    statRow.appendChild(tile);
  }
  section.appendChild(statRow);

  if (panel.spans.length > 0) {
    const legend = el('div', 'legend');
    const deviceLabels = [['cpu', 'CPU'], ['gpu', 'GPU'], ['other', 'Other']];
    for (const [device, label] of deviceLabels) {
      if (!panel.spans.some((s) => s.device === device)) continue;
      const item = el('div', 'legend-item');
      const sw = el('span', 'legend-swatch');
      sw.style.background = 'var(--series-' + device + ')';
      const txt = document.createElement('span');
      txt.textContent = label;
      item.appendChild(sw);
      item.appendChild(txt);
      legend.appendChild(item);
    }
    if (panel.idleGaps.length > 0) {
      const item = el('div', 'legend-item');
      const sw = el('span', 'legend-swatch');
      sw.style.background = 'var(--status-warning)';
      sw.style.opacity = '0.5';
      const txt = document.createElement('span');
      txt.textContent = 'GPU idle';
      item.appendChild(sw);
      item.appendChild(txt);
      legend.appendChild(item);
    }
    section.appendChild(legend);

    const scrollWrap = el('div', 'timeline-scroll');
    scrollWrap.appendChild(buildSvg(panel));
    section.appendChild(scrollWrap);

    const table = buildStatsTable(panel);
    if (table) section.appendChild(table);
  } else {
    const note = el('p', 'empty-note');
    note.textContent = 'No named stage spans in this trace (quick-look mode has no ' +
      'stage markers) -- only CPU/GPU utilization samples were captured.';
    section.appendChild(note);
  }

  container.appendChild(section);
}

const container = document.getElementById('panels');
for (const panel of PANELS) buildPanel(panel, container);
"""
