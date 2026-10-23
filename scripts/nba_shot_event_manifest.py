"""Build an NBA broadcast shot-event validation set from PBP.

Example:
    python scripts/nba_shot_event_manifest.py --count 20

Outputs:
    outputs/shot-events/<event>/shot_events.json   all field-goal attempts
    outputs/shot-events/<event>/validation.json   selected clips + PBP labels
    outputs/shot-events/<event>/clips/*.mp4
    outputs/shot-events/<event>/validation.html   alignment-audit page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analytics.shot_events import build_shot_events, select_validation  # noqa: E402
from make_play_label_page import cut_clip  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def video_info(path: Path) -> tuple[float, int]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"bad video metadata: fps={fps} frames={frames}")
    return fps, frames


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Shot event alignment — __EVENT__</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #101214; color: #e8eaed; }
  header { position: sticky; top: 0; z-index: 5; background: #171a1d; border-bottom: 1px solid #2b3035;
           padding: 12px 18px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .small { color: #9aa0a6; font-size: 13px; }
  .bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  .bar button { padding: 6px 10px; background: #263238; color: #e8eaed; border: 1px solid #455a64;
                border-radius: 4px; cursor: pointer; }
  #list { padding: 14px 18px 60px; display: grid; gap: 16px; max-width: 1100px; }
  .card { background: #171a1d; border: 1px solid #2b3035; border-radius: 8px; padding: 12px; }
  .card.current { border-color: #ffd54a; }
  .head { display: flex; gap: 12px; flex-wrap: wrap; align-items: baseline; margin-bottom: 8px; }
  .head b { font-size: 16px; }
  .meta { color: #9aa0a6; font-size: 13px; }
  .meta span { margin-right: 12px; }
  video { width: 100%; max-height: 420px; background: #000; border-radius: 4px; display: block; }
  .truth { font-size: 13px; color: #cfd8dc; margin: 8px 0; }
  .truth b { color: #ffd54a; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .pick { padding: 5px 9px; background: #263238; color: #e8eaed; border: 1px solid #455a64;
          border-radius: 4px; cursor: pointer; font-size: 13px; }
  .pick.sel.good { background: #2e7d32; border-color: #66bb6a; }
  .pick.sel.bad { background: #b71c1c; border-color: #ef5350; }
  .pick.sel.amb { background: #8d6e00; border-color: #ffd54a; }
  pre { background: #0b0d0e; border: 1px solid #2b3035; padding: 10px; white-space: pre-wrap;
        max-height: 260px; overflow: auto; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Shot event alignment — __EVENT__</h1>
  <div class="small">Does the clip show the labeled shot? Mark aligned, misaligned, or ambiguous.
    Use this to reject bad PBP-to-frame alignment before model scoring.</div>
  <div class="bar">
    <span id="progress"></span>
    <button id="export">Export</button>
    <button id="download">Download alignment.json</button>
    <button id="copy">Copy JSON</button>
  </div>
</header>
<div id="list"></div>
<pre id="out">No audits yet.</pre>
<script>
const EVENTS = __EVENTS__;
const EVENT_ID = __EVENT_ID__;
const STORAGE_KEY = "shot-alignment-" + EVENT_ID;
let audits = {};
try { audits = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { audits = {}; }
const list = document.getElementById("list");
const out = document.getElementById("out");
const progress = document.getElementById("progress");
const cards = [];

EVENTS.forEach((e, i) => {
  const card = document.createElement("section");
  card.className = "card";
  card.dataset.event = e.event_id;
  const reb = e.rebound ? `${e.rebound.team} ${e.rebound.offensive ? "offensive" : "defensive"} rebound` : "no rebound";
  card.innerHTML = `
    <div class="head"><b>${e.event_id}</b>
      <span class="meta"><span>Q${e.period}</span><span>${e.clock_s}s</span>
      <span>${e.team}</span><span>${e.shot_value}PT ${e.shot_type}</span></span></div>
    <video controls preload="metadata" playsinline src="${e.video}"></video>
    <div class="truth"><b>PBP:</b> ${e.text}<br>
      <b>Label:</b> ${e.made ? "MADE" : "MISSED"} · ${e.shooter || "unknown"} · ${reb}</div>
    <div class="row">
      <button class="pick good" data-pick="aligned">Aligned</button>
      <button class="pick bad" data-pick="misaligned">Misaligned</button>
      <button class="pick amb" data-pick="ambiguous">Ambiguous</button>
    </div>`;
  card.querySelectorAll(".pick").forEach(btn => {
    btn.onclick = () => {
      audits[e.event_id] = { state: btn.dataset.pick, updated_at: new Date().toISOString() };
      save(); render();
    };
  });
  list.appendChild(card);
  cards.push(card);
});

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(audits)); }
  catch (e) { out.textContent = "localStorage unavailable; use Export/Copy."; }
}

function render() {
  let aligned = 0, bad = 0, amb = 0;
  cards.forEach(card => {
    const id = card.dataset.event;
    const state = audits[id]?.state;
    card.querySelectorAll(".pick").forEach(b => b.classList.toggle("sel", b.dataset.pick === state));
    if (state === "aligned") aligned++;
    if (state === "misaligned") bad++;
    if (state === "ambiguous") amb++;
  });
  progress.textContent = `${aligned} aligned · ${bad} misaligned · ${amb} ambiguous / ${cards.length}`;
  out.textContent = JSON.stringify({ event_id: EVENT_ID, audits, events: EVENTS }, null, 2);
}
render();

document.getElementById("export").onclick = render;
document.getElementById("download").onclick = () => {
  render();
  const blob = new Blob([out.textContent], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "alignment.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
document.getElementById("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(out.textContent); progress.textContent = "Copied JSON."; }
  catch (e) { progress.textContent = "Copy failed; select the JSON manually."; }
};
</script>
</body>
</html>
"""


def write_html(path: Path, event_id: str, events: list[dict]) -> None:
    payload = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    html = (HTML_TEMPLATE
            .replace("__EVENT__", event_id)
            .replace("__EVENT_ID__", json.dumps(event_id))
            .replace("__EVENTS__", payload))
    path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeline", default="data/state/401810030/timeline.json")
    parser.add_argument("--state", default="data/state/401810030/state.json")
    parser.add_argument("--manual", default="state_layer/examples/manual_game.json")
    parser.add_argument("--video", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=4.0)
    parser.add_argument("--clip-width", type=int, default=640)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    timeline = _read(args.timeline)
    state = _read(args.state)
    manual = _read(args.manual)
    video = Path(args.video or manual["video_path"])
    if not video.exists():
        print(f"[shot-events] video not found: {video}", file=sys.stderr)
        return 1
    fps, total_frames = video_info(video)
    event_id = str(state.get("event_id") or timeline.get("event_id") or "game")

    events = build_shot_events(
        timeline.get("plays", []),
        state.get("player_names", {}),
        state.get("team_names", {}),
        manual.get("clock_anchors", []),
    )
    if not events:
        print("[shot-events] no field-goal events", file=sys.stderr)
        return 1
    selected = select_validation(events, count=args.count, seed=args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / "shot-events" / event_id
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    validation = []
    for event in selected:
        clip_id = f"{event['event_id']}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        start = max(0, event["frame"] - int(args.before * fps))
        end = min(total_frames - 1, event["frame"] + int(args.after * fps))
        duration = (end - start) / fps
        cut_clip(video, clip_path, start / fps, duration, args.clip_width, args.skip_existing)
        validation.append({
            **event,
            "clip_id": clip_id,
            "video": f"clips/{clip_id}.mp4",
            "clip_start_frame": start,
            "clip_end_frame": end,
            "clip_start_s": round(start / fps, 3),
            "duration_s": round(duration, 3),
        })
        print(f"[shot-events] {clip_id}: Q{event['period']} {event['clock_s']}s "
              f"{'MADE' if event['made'] else 'MISS'} {event['shot_value']}PT "
              f"{event['shooter']} ({start / fps:.1f}s +{duration:.1f}s)")

    (out_dir / "shot_events.json").write_text(
        json.dumps({"event_id": event_id, "count": len(events), "events": events},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "validation.json").write_text(
        json.dumps({"event_id": event_id, "fps": fps, "count": len(validation),
                    "source_video": str(video), "events": validation},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_html(out_dir / "validation.html", event_id, validation)
    (out_dir / "README.md").write_text(
        f"""# Shot-event validation — `{event_id}`

{len(events)} field-goal attempts in the PBP; {len(validation)} selected clips
({sum(1 for e in validation if e['made'])} made, {sum(1 for e in validation if not e['made'])} missed).

## Run

```powershell
python -m http.server 8767 --directory outputs/shot-events/{event_id}
```

Open <http://localhost:8767/validation.html> and audit whether each clip shows the PBP shot.
The PBP label is a teacher label, not video ground truth. Reject misaligned clips before
scoring a detector.

## Files

- `shot_events.json` — all PBP field-goal attempts with aligned frames and rebound context.
- `validation.json` — the selected clips and labels.
- `validation.html` — alignment-audit page.
""",
        encoding="utf-8",
    )
    print(f"[shot-events] wrote {out_dir / 'shot_events.json'}")
    print(f"[shot-events] wrote {out_dir / 'validation.json'}")
    print(f"[shot-events] wrote {out_dir / 'validation.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
