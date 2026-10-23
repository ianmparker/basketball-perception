"""Build a landing page that links every broadcast demo run under a directory.

    python scripts/build_demo_index.py --root outputs/broadcast_demo

Each subdirectory with a ``summary.json`` becomes a card. The cards use the
poster frame and show the event counts, ball detection, and a link to the demo.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Broadcast Demo Runs</title>
<style>
  :root { --bg:#080c11; --panel:#101923; --panel2:#152231; --line:#243447; --text:#e6edf5; --muted:#8fa3b8; --accent:#f59e0b; --blue:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; background: radial-gradient(circle at 20% 0%, #142334 0%, var(--bg) 45%); color:var(--text); font-family:"Segoe UI",system-ui,Arial,sans-serif; line-height:1.5; }
  .wrap { max-width:1200px; margin:0 auto; padding:32px 22px 60px; }
  h1 { font-size:clamp(28px,4vw,44px); margin:0 0 8px; letter-spacing:-.02em; }
  .sub { color:var(--muted); max-width:760px; margin:0 0 22px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:18px; }
  .card { background:linear-gradient(180deg,rgba(255,255,255,.025),transparent),var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  .card img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#000; }
  .card-body { padding:14px 16px 16px; }
  .card h2 { font-size:18px; margin:0 0 4px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:10px; }
  .stats { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
  .stat { background:var(--panel2); border:1px solid var(--line); border-radius:999px; padding:4px 9px; font-size:12px; color:var(--muted); }
  .stat b { color:var(--text); }
  a.button { display:inline-block; background:var(--accent); color:#1a1205; padding:8px 13px; border-radius:9px; text-decoration:none; font-weight:700; font-size:13px; }
  a.button:hover { filter:brightness(1.08); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:26px; }
  th, td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .note { border-left:3px solid var(--accent); padding:10px 14px; background:rgba(245,158,11,.07); border-radius:0 10px 10px 0; color:var(--muted); margin-top:26px; }
  code { background:#0b121a; border:1px solid var(--line); border-radius:6px; padding:2px 6px; font-size:12px; }
  footer { color:var(--muted); font-size:12px; margin-top:26px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Broadcast Demo Runs</h1>
  <p class="sub">Every run below is the same pipeline: court calibration, ball and player tracking, team/possession, event detection, and a tactical view. Open a run to see the annotated video, shot chart, and event timeline.</p>
  <div class="grid">
    __CARDS__
  </div>
  <table>
    <thead><tr><th>Run</th><th class="num">Shots</th><th class="num">Screens</th><th class="num">PnRs</th><th class="num">Passes</th><th class="num">Interceptions</th><th class="num">Ball detected</th><th class="num">Players/frame</th></tr></thead>
    <tbody>__ROWS__</tbody>
  </table>
  <div class="note">
    <b>This is a broadcast demo, not a product accuracy claim.</b> Event detection is heuristic; shots can be early or missed, and team/possession can drift. Handheld/tripod footage is Phase B. Research-only footage stays local and is not redistributed.
  </div>
  <footer>
    Run a new clip: <code>python scripts/broadcast_demo.py --video &lt;clip&gt; --out-dir outputs/broadcast_demo/&lt;name&gt;</code><br>
    Rebuild this page: <code>python scripts/build_demo_index.py --root outputs/broadcast_demo</code>
  </footer>
</div>
</body>
</html>
"""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _card(run_dir: Path, summary: dict) -> str:
    title = html.escape(summary.get("title") or run_dir.name)
    source = Path(summary.get("source") or "unknown").name
    counts = summary.get("event_counts") or {}
    poster = run_dir / "assets" / "poster.jpg"
    img = f'<img src="{html.escape(run_dir.name)}/assets/poster.jpg" alt="{title}">' if poster.is_file() else '<div style="aspect-ratio:16/9;background:#0b121a"></div>'
    stats = (
        f'<span class="stat"><b>{counts.get("shots", 0)}</b> shots</span>'
        f'<span class="stat"><b>{counts.get("screens", 0)}</b> screens</span>'
        f'<span class="stat"><b>{counts.get("pnrs", 0)}</b> PnRs</span>'
        f'<span class="stat"><b>{counts.get("passes", 0)}</b> passes</span>'
        f'<span class="stat"><b>{summary.get("ball", {}).get("detected_pct", 0):.0f}%</b> ball</span>'
    )
    return (
        f'<div class="card">{img}<div class="card-body">'
        f'<h2>{title}</h2>'
        f'<div class="meta">{html.escape(source)} · {summary.get("duration_s", 0):.1f}s · {summary.get("fps", 0)} fps</div>'
        f'<div class="stats">{stats}</div>'
        f'<a class="button" href="{html.escape(run_dir.name)}/index.html">Open demo</a>'
        f'</div></div>'
    )


def _row(run_dir: Path, summary: dict) -> str:
    counts = summary.get("event_counts") or {}
    return (
        f'<tr><td><a href="{html.escape(run_dir.name)}/index.html" style="color:var(--blue)">'
        f'{html.escape(summary.get("title") or run_dir.name)}</a></td>'
        f'<td class="num">{counts.get("shots", 0)}</td>'
        f'<td class="num">{counts.get("screens", 0)}</td>'
        f'<td class="num">{counts.get("pnrs", 0)}</td>'
        f'<td class="num">{counts.get("passes", 0)}</td>'
        f'<td class="num">{counts.get("interceptions", 0)}</td>'
        f'<td class="num">{summary.get("ball", {}).get("detected_pct", 0):.1f}%</td>'
        f'<td class="num">{summary.get("players", {}).get("mean_per_frame", 0):.1f}</td></tr>'
    )


def build_index(root: Path) -> int:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            summary = _load(summary_path)
        except (json.JSONDecodeError, OSError):
            continue
        runs.append((summary_path.parent, summary))
    if not runs:
        print(f"[demo-index] no runs found under {root}", file=sys.stderr)
        return 1
    cards = "\n".join(_card(run_dir, summary) for run_dir, summary in runs)
    rows = "\n".join(_row(run_dir, summary) for run_dir, summary in runs)
    page = TEMPLATE.replace("__CARDS__", cards).replace("__ROWS__", rows)
    (root / "index.html").write_text(page, encoding="utf-8")
    print(f"[demo-index] wrote {root / 'index.html'} ({len(runs)} run(s))")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/broadcast_demo"))
    args = parser.parse_args(argv)
    return build_index(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
