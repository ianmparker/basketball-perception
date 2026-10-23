"""Local labeling page for broadcast shot candidates.

Serves the seven candidate clips and writes labels directly to
``eval/broadcast_demo/labels.json``. No cloud, no upload, no copy/paste.

    python scripts/label_broadcast_events.py --port 8766

Then open http://127.0.0.1:8766/ and label each clip:

    1 = real shot, 0 = false positive, S = skip/unsure
    R = replay, arrows = previous/next

After all candidates are labeled, the page can run the event scorer and show
``eval/broadcast_demo/event_scorecard.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import date
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "eval" / "broadcast_demo" / "labels.json"
SCORE_SCRIPT = REPO_ROOT / "scripts" / "score_broadcast_events.py"
LABEL_FIELDS = ("label", "release_frame", "shooter_track_id_gt", "made", "notes")


def load_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_labels(data: dict, path: Path) -> None:
    data["updated"] = date.today().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def flatten_events(data: dict) -> list[dict]:
    events = []
    for run in data.get("runs") or []:
        for event in run.get("events") or []:
            events.append({"run_id": run.get("id"), "run_clip": run.get("clip"), **event})
    return events


def progress(data: dict) -> dict:
    total = sum(len(run.get("events") or []) for run in data.get("runs") or [])
    labeled = sum(
        1
        for run in data.get("runs") or []
        for event in run.get("events") or []
        if event.get("label") is not None
    )
    return {"total": total, "labeled": labeled, "remaining": total - labeled}


def update_event(data: dict, run_id: str, event_id: str, fields: dict) -> dict:
    for run in data.get("runs") or []:
        if run.get("id") != run_id:
            continue
        for event in run.get("events") or []:
            if event.get("id") != event_id:
                continue
            if "label" in fields:
                label = fields["label"]
                if label not in (None, 0, 1, "gather"):
                    raise ValueError("label must be 1, 0, 'gather', or null")
                event["label"] = label
            for key in ("release_frame", "shooter_track_id_gt"):
                if key in fields:
                    value = fields[key]
                    event[key] = None if value in (None, "") else int(value)
            if "made" in fields:
                value = fields["made"]
                event["made"] = None if value in (None, "") else bool(value)
            if "notes" in fields:
                event["notes"] = str(fields["notes"] or "")
            counts = progress(data)
            data["status"] = "labeled" if counts["remaining"] == 0 else "in_progress"
            data["updated"] = date.today().isoformat()
            return event
    raise KeyError(f"unknown event {run_id}/{event_id}")


def clip_path(data: dict, run_id: str, event_id: str) -> Path | None:
    for run in data.get("runs") or []:
        if run.get("id") != run_id:
            continue
        for event in run.get("events") or []:
            if event.get("id") != event_id:
                continue
            raw = event.get("clip")
            if not raw:
                return None
            path = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            root = REPO_ROOT.resolve()
            if root not in path.parents or path.suffix.lower() != ".mp4" or not path.is_file():
                return None
            return path
    return None


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Broadcast Shot Labeling</title>
<style>
  :root { --bg:#080c11; --panel:#101923; --panel2:#152231; --line:#243447; --text:#e6edf5; --muted:#8fa3b8; --accent:#f59e0b; --green:#34d399; --red:#fb7185; --blue:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; background:radial-gradient(circle at 20% 0%, #142334 0%, var(--bg) 45%); color:var(--text); font-family:"Segoe UI",system-ui,Arial,sans-serif; line-height:1.5; }
  .wrap { max-width:1200px; margin:0 auto; padding:22px 20px 70px; }
  header { position:sticky; top:0; z-index:5; background:rgba(8,12,17,.92); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
  .head-inner { max-width:1200px; margin:0 auto; padding:14px 20px; display:flex; flex-wrap:wrap; align-items:center; gap:14px; justify-content:space-between; }
  h1 { font-size:20px; margin:0; letter-spacing:-.02em; }
  .muted { color:var(--muted); }
  .small { font-size:12px; }
  .progress { flex:1; min-width:200px; }
  .bar { height:7px; background:#1c2a39; border-radius:999px; overflow:hidden; margin-top:6px; }
  .bar > i { display:block; height:100%; width:0%; background:linear-gradient(90deg,var(--accent),var(--green)); transition:width .2s; }
  .card { background:linear-gradient(180deg,rgba(255,255,255,.025),transparent),var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; margin-top:18px; }
  .grid { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:18px; }
  video { width:100%; border-radius:10px; background:#000; display:block; }
  .meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-bottom:14px; }
  .meta div { background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:9px 11px; }
  .meta b { display:block; font-size:15px; }
  .meta span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .buttons { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
  button, a.btn { border:1px solid var(--line); background:var(--panel2); color:var(--text); border-radius:9px; padding:9px 13px; cursor:pointer; font-size:14px; text-decoration:none; display:inline-block; }
  button:hover, a.btn:hover { border-color:var(--accent); }
  button.real { background:rgba(52,211,153,.14); border-color:rgba(52,211,153,.55); color:var(--green); font-weight:700; }
  button.fp { background:rgba(251,113,133,.14); border-color:rgba(251,113,133,.55); color:var(--red); font-weight:700; }
  button.skip { color:var(--muted); }
  button:disabled { opacity:.5; cursor:wait; }
  .keys { color:var(--muted); font-size:12px; margin-top:8px; }
  .kbd { display:inline-block; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:1px 5px; font-family:ui-monospace,Consolas,monospace; color:var(--text); background:#0b121a; font-size:11px; }
  .fields { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:12px; }
  .fields label { display:flex; flex-direction:column; gap:4px; color:var(--muted); font-size:12px; }
  .fields input, .fields select, .fields textarea { background:#0b121a; border:1px solid var(--line); color:var(--text); border-radius:7px; padding:7px 9px; font-size:13px; }
  .fields .wide { grid-column:1 / -1; }
  .status { display:inline-block; border-radius:999px; padding:2px 9px; font-size:11px; font-weight:700; }
  .status.real { background:rgba(52,211,153,.16); color:var(--green); }
  .status.fp { background:rgba(251,113,133,.16); color:var(--red); }
  .status.gather { background:rgba(245,158,11,.16); color:var(--accent); }
  .status.todo { background:rgba(143,163,184,.14); color:var(--muted); }
  .list { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:8px; margin-top:14px; }
  .item { background:var(--panel2); border:1px solid var(--line); border-radius:9px; padding:9px 10px; cursor:pointer; font-size:12px; }
  .item:hover { border-color:var(--accent); }
  .item.current { border-color:var(--accent); box-shadow:0 0 0 1px rgba(245,158,11,.35); }
  pre { background:#0b121a; border:1px solid var(--line); border-radius:10px; padding:12px; white-space:pre-wrap; font-size:12px; max-height:420px; overflow:auto; }
  .banner { border-left:3px solid var(--accent); background:rgba(245,158,11,.07); padding:10px 14px; border-radius:0 10px 10px 0; margin-top:14px; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div class="head-inner">
    <div>
      <h1>Broadcast Shot Labeling</h1>
      <div class="small muted">One clip at a time. Labels write straight to <code>eval/broadcast_demo/labels.json</code>.</div>
    </div>
    <div class="progress">
      <div class="small"><span id="progress-text">Loading…</span></div>
      <div class="bar"><i id="progress-bar"></i></div>
    </div>
  </div>
</header>

<div class="wrap">
  <div id="app"><p class="muted">Loading candidates…</p></div>
</div>

<script>
const app = document.getElementById('app');
let data = null;
let events = [];
let index = 0;

function esc(value) {
  return String(value === null || value === undefined ? '' : value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
function numOrNull(id) {
  const el = document.getElementById(id);
  if (!el || el.value === '') return null;
  const value = Number(el.value);
  return Number.isFinite(value) ? value : null;
}
function madeValue() {
  const el = document.getElementById('made');
  if (!el || el.value === '') return null;
  return el.value === 'yes';
}
function flatten(payload) {
  const out = [];
  for (const run of payload.runs || []) {
    for (const event of run.events || []) {
      out.push({run_id: run.id, run_clip: run.clip, ...event});
    }
  }
  return out;
}
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
async function load() {
  data = await api('/api/labels');
  events = flatten(data);
  const firstUnlabeled = events.findIndex((event) => event.label === null);
  index = firstUnlabeled >= 0 ? firstUnlabeled : 0;
  render();
}
function current() { return events[index]; }
function videoSrc(event) {
  return `/clip/${encodeURIComponent(event.run_id)}/${encodeURIComponent(event.id)}`;
}
function statusHtml(event) {
  if (event.label === 1) return '<span class="status real">REAL SHOT</span>';
  if (event.label === 0) return '<span class="status fp">FALSE POSITIVE</span>';
  if (event.label === 'gather') return '<span class="status gather">GATHER ONLY</span>';
  return '<span class="status todo">UNLABELED</span>';
}
function render() {
  if (!events.length) { app.innerHTML = '<p class="muted">No candidates found.</p>'; return; }
  const event = current();
  const labeled = events.filter((item) => item.label !== null).length;
  document.getElementById('progress-text').textContent = `${labeled} of ${events.length} labeled · candidate ${index + 1}`;
  document.getElementById('progress-bar').style.width = `${100 * labeled / events.length}%`;
  app.innerHTML = `
    <div class="card">
      <div class="grid">
        <div>
          <video id="video" controls autoplay muted loop playsinline src="${videoSrc(event)}"></video>
          <div class="buttons">
            <button class="real" onclick="setLabel(1)">1 · Real shot (release visible)</button>
            <button class="fp" onclick="setLabel(0)">0 · No shot</button>
            <button class="skip" onclick="setLabel('gather')">G · Gather only</button>
            <button class="skip" onclick="setLabel(null)">S · Skip / unsure</button>
            <button onclick="replay()">R · Replay</button>
          </div>
          <div class="keys">
            <span class="kbd">1</span> release visible ·
            <span class="kbd">0</span> no shot ·
            <span class="kbd">G</span> gather only ·
            <span class="kbd">S</span> skip ·
            <span class="kbd">R</span> replay ·
            <span class="kbd">←</span><span class="kbd">→</span> navigate
          </div>
          <div class="banner" style="margin-top:10px;">
            <b>What counts as 1:</b> a real shot attempt with the release visible in the clip.
            If the clip stops at the gather, use <b>G</b> — it is not a false positive and it is
            not a confirmed shot. Metadata errors (type, distance, location, shooter) do not
            make a real shot a 0.
          </div>
        </div>
        <div>
          <div class="meta">
            <div><span>Run</span><b>${esc(event.run_id)}</b></div>
            <div><span>Candidate</span><b>${index + 1} / ${events.length}</b></div>
            <div><span>Predicted frame</span><b>${esc(event.predicted_frame)}</b></div>
            <div><span>Time</span><b>${esc(event.predicted_time_s)}s</b></div>
            <div><span>Type</span><b>${esc(event.shot_type)}</b></div>
            <div><span>Distance</span><b>${event.distance_ft ? Number(event.distance_ft).toFixed(1) + ' ft' : '—'}</b></div>
            <div><span>Location</span><b>${event.location_valid ? 'valid' : 'unavailable'}</b></div>
            <div><span>Shooter track</span><b>${esc(event.shooter_track_id ?? '—')}</b></div>
          </div>
          <div style="margin-bottom:8px;">${statusHtml(event)}</div>
          <div class="fields">
            <label>Release frame (optional)
              <input id="release" type="number" min="0" placeholder="predicted: ${esc(event.predicted_frame)}" value="${esc(event.release_frame ?? '')}">
            </label>
            <label>Shooter track ID (optional)
              <input id="shooter" type="number" min="0" value="${esc(event.shooter_track_id_gt ?? '')}">
            </label>
            <label>Made?
              <select id="made">
                <option value="" ${event.made === null || event.made === undefined ? 'selected' : ''}>unknown</option>
                <option value="yes" ${event.made === true ? 'selected' : ''}>yes</option>
                <option value="no" ${event.made === false ? 'selected' : ''}>no</option>
              </select>
            </label>
            <label>Notes
              <input id="notes" type="text" placeholder="optional" value="${esc(event.notes ?? '')}">
            </label>
            <label class="wide" style="flex-direction:row;align-items:center;gap:8px;">
              <button onclick="saveFields()">Save fields</button>
              <span class="small muted">Fields save with the next label click too.</span>
            </label>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;">
        <b>All candidates</b>
        <span class="small muted">Click a card to jump. Missed shots can be added later with <code>predicted_frame: null</code>.</span>
      </div>
      <div class="list">
        ${events.map((item, i) => `
          <div class="item ${i === index ? 'current' : ''}" onclick="jump(${i})">
            <div><b>${esc(item.run_id)}</b> · frame ${esc(item.predicted_frame)}</div>
            <div class="small muted">${esc(item.shot_type)} · ${item.location_valid ? 'located' : 'no location'}</div>
            <div style="margin-top:4px;">${statusHtml(item)}</div>
          </div>`).join('')}
      </div>
      <div class="banner" id="score-banner" style="display:none;">
        <b>All candidates labeled.</b> Run the scorer to turn this into precision/recall/F1.
        <div style="margin-top:8px;">
          <button id="score-btn" onclick="score()">Score now</button>
          <a class="btn" href="/api/labels" download="labels.json">Download labels.json</a>
        </div>
      </div>
      <pre id="scorecard" style="display:none;"></pre>
    </div>`;
  if (labeled === events.length) {
    document.getElementById('score-banner').style.display = 'block';
  }
}
function jump(target) { index = Math.max(0, Math.min(events.length - 1, target)); render(); }
function replay() { const video = document.getElementById('video'); if (video) { video.currentTime = 0; video.play(); } }
async function saveFields() {
  const event = current();
  const fields = {
    label: event.label,
    release_frame: numOrNull('release'),
    shooter_track_id_gt: numOrNull('shooter'),
    made: madeValue(),
    notes: document.getElementById('notes').value,
  };
  const result = await api('/api/label', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({run_id:event.run_id, event_id:event.id, ...fields})});
  Object.assign(event, fields);
  data.status = result.status;
  render();
}
async function setLabel(value) {
  const event = current();
  const fields = {
    label: value,
    release_frame: numOrNull('release'),
    shooter_track_id_gt: numOrNull('shooter'),
    made: madeValue(),
    notes: document.getElementById('notes') ? document.getElementById('notes').value : '',
  };
  await api('/api/label', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({run_id:event.run_id, event_id:event.id, ...fields})});
  Object.assign(event, fields);
  const nextUnlabeled = events.findIndex((item, i) => i > index && item.label === null);
  index = nextUnlabeled >= 0 ? nextUnlabeled : (index + 1) % events.length;
  render();
}
async function score() {
  const button = document.getElementById('score-btn');
  button.disabled = true;
  button.textContent = 'Scoring…';
  const result = await api('/api/score', {method:'POST'});
  const box = document.getElementById('scorecard');
  box.style.display = 'block';
  box.textContent = result.scorecard || result.stdout || 'No scorecard produced.';
  button.disabled = false;
  button.textContent = 'Score again';
}
document.addEventListener('keydown', (event) => {
  const tag = document.activeElement ? document.activeElement.tagName : '';
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
  if (event.key === '1') setLabel(1);
  else if (event.key === '0') setLabel(0);
  else if (event.key.toLowerCase() === 'g') setLabel('gather');
  else if (event.key.toLowerCase() === 's') setLabel(null);
  else if (event.key.toLowerCase() === 'r') replay();
  else if (event.key === 'ArrowRight') jump(index + 1);
  else if (event.key === 'ArrowLeft') jump(index - 1);
});
load().catch((error) => { app.innerHTML = `<p style="color:#fb7185">${esc(error.message)}</p>`; });
</script>
</body>
</html>
"""


def create_app(labels_path: Path = DEFAULT_LABELS) -> Flask:
    app = Flask(__name__)
    lock = threading.Lock()
    state = {"data": load_labels(labels_path)}

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/labels")
    def get_labels():
        return jsonify(state["data"])

    @app.post("/api/label")
    def post_label():
        payload = request.get_json(silent=True) or {}
        run_id, event_id = payload.get("run_id"), payload.get("event_id")
        if not run_id or not event_id:
            abort(400, "run_id and event_id are required")
        fields = {key: payload[key] for key in LABEL_FIELDS if key in payload}
        with lock:
            try:
                update_event(state["data"], run_id, event_id, fields)
            except (KeyError, ValueError) as exc:
                abort(400, str(exc))
            save_labels(state["data"], labels_path)
            counts = progress(state["data"])
        return jsonify({"ok": True, "status": state["data"].get("status"), **counts})

    @app.get("/clip/<run_id>/<event_id>")
    def clip(run_id: str, event_id: str):
        path = clip_path(state["data"], run_id, event_id)
        if path is None:
            abort(404, "clip not found")
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @app.post("/api/score")
    def score():
        proc = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT), "--labels", str(labels_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        scorecard = ""
        path = REPO_ROOT / "eval" / "broadcast_demo" / "event_scorecard.md"
        if path.is_file():
            scorecard = path.read_text(encoding="utf-8", errors="replace")
        return jsonify({
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "scorecard": scorecard,
        })

    @app.get("/health")
    def health():
        return jsonify({"ok": True, **progress(state["data"])})

    return app


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    app = create_app(args.labels)
    print(f"[labels] http://{args.host}:{args.port}/")
    print(f"[labels] writing to {args.labels}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
