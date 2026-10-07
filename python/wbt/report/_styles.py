REPORT_CSS = """
[data-theme="light"] {
    color-scheme: light;
    --bg: #f3f5f7; --panel: #ffffff; --panel-2: #eef2f6;
    --ink: #17283b; --muted: #52647a; --border: #dde4ec; --border-strong: #b9c6d5;
    --accent: #174d75; --up: #b52e40; --down: #087765;
}
[data-theme="dark"] {
    color-scheme: dark;
    --bg: #111b28; --panel: #182536; --panel-2: #203146;
    --ink: #e7edf5; --muted: #a6b7cb; --border: #314359; --border-strong: #566c86;
    --accent: #94c9ef; --up: #ff9ba7; --down: #75d9bd;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 90px; }
body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
button { font: inherit; cursor: pointer; }
button, a, summary, [tabindex] { -webkit-tap-highlight-color: transparent; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
.skip-link { position: fixed; top: -100px; left: 16px; z-index: 100; padding: 12px; background: var(--panel); color: var(--ink); }
.skip-link:focus { top: 12px; }
.container { width: min(1280px, 100% - 64px); margin-inline: auto; }
.header-section { background: var(--panel); border-top: 4px solid var(--accent); border-bottom: 1px solid var(--border); padding: 36px 0 28px; }
.header-bar { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }
.header-title { margin: 0; font-size: clamp(23px, 3vw, 32px); font-weight: 650; line-height: 1.3; letter-spacing: -.025em; overflow-wrap: anywhere; }
.header-subtitle { margin: 10px 0 0; color: var(--muted); max-width: 70ch; }
.theme-switch { display: flex; flex: none; padding: 3px; border: 1px solid var(--border); border-radius: 6px; gap: 2px; }
.theme-switch button { background: transparent; color: var(--muted); border: 0; border-radius: 3px; padding: 6px 12px; }
.theme-switch button.active { background: var(--panel-2); color: var(--ink); font-weight: 600; }
.param-badges { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-top: 24px; }
.param-badge { color: var(--muted); font-size: 12px; }
.param-badge b { margin-left: 6px; font-weight: 550; color: var(--ink); font-variant-numeric: tabular-nums; }
.main-content { padding-top: 32px; padding-bottom: 48px; }
.main-content > section { margin-bottom: 32px; }
.section-header { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
.section-icon { display: none; }
.section-title { font-size: 18px; font-weight: 650; margin: 0; }
.stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)) !important; gap: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--panel); }
.stat-tile { padding: 18px 24px; min-width: 0; border-bottom: 1px solid var(--border); }
.stat-label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.stat-value { display: block; font-size: 24px; font-weight: 600; letter-spacing: -.025em; font-variant-numeric: tabular-nums; }
.metric-positive, .t-up { color: var(--up); }
.metric-negative, .t-down { color: var(--down); }
.metric-neutral { color: var(--ink); }
.nav-tabs { position: sticky; top: 0; z-index: 10; margin: 0; padding: 8px 0; display: flex; gap: 6px; list-style: none; overflow-x: auto; background: var(--bg); border-bottom: 1px solid var(--border-strong); }
.nav-item { flex: none; }
.nav-link { padding: 10px 18px; border: 0; background: transparent; border-radius: 5px; color: var(--muted); white-space: nowrap; }
.nav-link.active { background: var(--panel); box-shadow: inset 0 -2px var(--accent); color: var(--accent); font-weight: 650; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }
.chart-grid { display: grid; gap: 24px; padding-top: 24px; }
.chart-grid-item { min-width: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.chart-grid-item.full-width { grid-column: 1 / -1; }
.chart-grid-title { margin: 0; padding: 18px 24px; font-size: 15px; font-weight: 650; border-bottom: 1px solid var(--border); }
.chart-grid-item .plotly-graph-div { width: 100% !important; }
.chart-body, .section-content { padding-top: 20px; min-width: 0; }
.fin-wrap, .data-table { width: 100%; overflow-x: auto; padding: 10px 24px 20px; }
.fin-table, .table { border-collapse: collapse; width: 100%; font-size: 13px; line-height: 1.5; }
.fin-table caption { text-align: left; padding: 8px 0 16px; color: var(--muted); font-size: 12px; }
.fin-wrap[data-overflow="true"] caption::after { content: "左右滚动查看完整数据"; display: block; margin-top: 4px; font-size: 11px; }
.fin-table th, .fin-table td, .table th, .table td { padding: 12px 16px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.fin-table thead th, .table thead th { background: var(--panel-2); color: var(--muted); font-size: 12px; font-weight: 600; border-bottom-color: var(--border-strong); }
.fin-table tbody th, .fin-table thead th:first-child { text-align: left; }
.fin-table tbody th { font-weight: 500; color: var(--ink); }
.key-value-table tbody th { white-space: normal; min-width: 90px; }
.fin-table tbody tr:last-child > *, .table tbody tr:last-child > * { border-bottom: 0; }
.fin-table tbody tr:hover > *, .table tbody tr:hover > * { background: var(--panel-2); }
.verdict { padding: 20px 0 8px; }
.verdict-mode-title { margin: 0; padding: 22px 24px 0; font-size: 13px; font-weight: 650; color: var(--muted); }
.verdict-head { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 0 24px 16px; }
.verdict-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 650; background: var(--panel-2); }
.verdict-badge.good { color: var(--down); }
.verdict-badge.bad { color: var(--up); }
.verdict-sub { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
.verdict-conds { margin: 0 24px; }
.verdict-cond { display: grid; grid-template-columns: 16px minmax(90px, auto) 1fr; gap: 10px; padding: 9px 0; font-size: 13px; }
.verdict-cond .ck { color: var(--muted); }
.verdict-cond .ct { font-weight: 600; }
.verdict-cond .cd { color: var(--muted); overflow-wrap: anywhere; }
.verdict-reason { margin: 16px 24px; color: var(--muted); font-size: 13px; max-width: 80ch; overflow-wrap: anywhere; }
.badge { font-size: 12px; font-weight: 600; }
.badge-pass { color: var(--down); }
.badge-fail { color: var(--muted); }
.verdict-details { border-top: 1px solid var(--border); margin-top: 16px; }
.verdict-details summary { padding: 16px 24px; cursor: pointer; color: var(--accent); font-size: 13px; }
.verdict-details .fin-table td { white-space: normal; overflow-wrap: anywhere; min-width: 180px; text-align: left; }
.footer { border-top: 1px solid var(--border); padding: 24px 0; color: var(--muted); font-size: 12px; }
.footer p { margin: 0; }
.bi { display: none; }
@media (max-width: 800px) {
    .container { width: calc(100% - 32px); }
    .chart-grid { grid-template-columns: minmax(0, 1fr) !important; }
    .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .stat-tile { padding: 16px; }
    .stat-value { font-size: 22px; }
    .header-bar { flex-direction: column; gap: 16px; }
    .header-section { padding-top: 24px; }
    .nav-link { padding: 9px 12px; }
    .chart-grid-title { padding: 16px; }
    .fin-wrap, .data-table { padding: 8px 16px 16px; }
    .fin-table th, .fin-table td { padding: 11px 12px; }
    .verdict-head, .verdict-conds, .verdict-reason { margin-inline: 16px; }
    .verdict-mode-title, .verdict-details summary { padding-inline: 16px; }
    .verdict-cond { grid-template-columns: 16px 1fr; }
    .verdict-cond .cd { grid-column: 2; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
@media print {
    .nav-tabs, .theme-switch { display: none; }
    .tab-pane { display: block; }
    .fin-wrap { overflow: visible; }
}
"""
