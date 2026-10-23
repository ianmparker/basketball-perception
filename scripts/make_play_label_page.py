"""Cut possession clips and generate a browser play-labeling page.

Example:
    python scripts/make_play_label_page.py \
        --state data/state/401810030/state.json \
        --manual state_layer/examples/manual_game.json

The page writes clips under outputs/play-labels/<event>/clips and a labels.html page.
Run a local server from that directory, label the possessions, then export labels.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analytics.play_labels import build_windows, select_windows  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover - only needed for clip extraction
    cv2 = None

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover
    imageio_ffmpeg = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_info(path: Path) -> tuple[float, int, int, int]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to read video metadata")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"bad video metadata: fps={fps} frames={frames}")
    return fps, frames, width, height


def ffmpeg_bin() -> str:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass
    return "ffmpeg"


def cut_clip(video: Path, out: Path, start_s: float, duration_s: float,
             width: int, skip_existing: bool, fps: float | None = None) -> None:
    if skip_existing and out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", str(video),
        "-t", f"{duration_s:.3f}",
    ]
    if fps:
        cmd += ["-r", f"{fps:g}"]
    cmd += [
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out.name}: {proc.stderr[-800:]}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
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
  .bar button:hover { background: #37474f; }
  #list { padding: 14px 18px 60px; display: grid; gap: 16px; max-width: 1100px; }
  .card { background: #171a1d; border: 1px solid #2b3035; border-radius: 8px; padding: 12px; }
  .card.current { border-color: #ffd54a; box-shadow: 0 0 0 1px #ffd54a55; }
  .card.unlabeled { border-left: 4px solid #5c6bc0; }
  .head { display: flex; gap: 12px; flex-wrap: wrap; align-items: baseline; margin-bottom: 8px; }
  .head b { font-size: 16px; }
  .meta { color: #9aa0a6; font-size: 13px; }
  .meta span { margin-right: 12px; }
  video { width: 100%; max-height: 420px; background: #000; border-radius: 4px; display: block; }
  .lineup { color: #b0bec5; font-size: 12px; margin: 8px 0; }
  .labels { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .lab { padding: 5px 9px; background: #263238; color: #e8eaed; border: 1px solid #455a64;
         border-radius: 4px; cursor: pointer; font-size: 13px; }
  .lab:hover { background: #37474f; }
  .lab.sel { background: #2e7d32; border-color: #66bb6a; }
  .key { display: inline-block; min-width: 14px; color: #ffd54a; font-weight: 700; margin-right: 4px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; color: #b0bec5; font-size: 13px; }
  .row input[type=text] { flex: 1; min-width: 220px; padding: 5px 7px; background: #101214;
                          color: #e8eaed; border: 1px solid #37474f; border-radius: 4px; }
  .clear { padding: 3px 7px; background: #37474f; color: #cfd8dc; border: 1px solid #546e7a;
           border-radius: 4px; cursor: pointer; font-size: 12px; }
  pre { background: #0b0d0e; border: 1px solid #2b3035; padding: 10px; white-space: pre-wrap;
        max-height: 300px; overflow: auto; font-size: 12px; }
  #hint { color: #ffd54a; font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="small">Label the action that creates the first advantage, not every screen or the final
    shot. Keys: 1-9, 0, Q = play type; A = ambiguous; J / K or arrows = next / previous.
    Export labels.json when done.</div>
  <div class="bar">
    <span id="hint"></span>
    <span id="progress" class="small"></span>
    <label class="small"><input id="auto" type="checkbox" checked> auto-advance</label>
    <label class="small"><input id="hide" type="checkbox"> hide labeled</label>
    <button id="export">Export</button>
    <button id="download">Download labels.json</button>
    <button id="copy">Copy JSON</button>
  </div>
</header>
<div id="list"></div>
<pre id="out">No labels yet.</pre>
<script>
const POSSESSIONS = __POSSESSIONS__;
const PLAY_TYPES = __PLAY_TYPES__;
const EVENT_ID = __EVENT_ID__;
const VIDEO = __VIDEO__;
const STORAGE_KEY = "play-labels-" + EVENT_ID;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { labels = {}; }
let current = 0;

const list = document.getElementById("list");
const progress = document.getElementById("progress");
const hint = document.getElementById("hint");
const out = document.getElementById("out");
const auto = document.getElementById("auto");
const hide = document.getElementById("hide");
const cards = [];

function shortcut(i) {
  if (i < 9) return String(i + 1);
  if (i === 9) return "0";
  return "Q";
}

function teamName(id) {
  const p = POSSESSIONS.find(x => x.offense_team_id === id || x.defense_team_id === id);
  if (!p) return String(id);
  return p.offense_team_id === id ? p.offense : p.defense;
}

function makeCard(p, index) {
  const card = document.createElement("section");
  card.className = "card unlabeled";
  card.dataset.pid = String(p.possession_id);
  card.dataset.index = String(index);
  card.innerHTML = `
    <div class="head">
      <b>#${p.possession_id}</b>
      <span class="meta">
        <span>Q${p.period}</span>
        <span>${p.offense} offense vs ${p.defense}</span>
        <span>${p.points} pts</span>
        <span>${p.end_reason}</span>
        <span>${p.duration_s.toFixed(1)}s</span>
      </span>
    </div>
    <video controls preload="metadata" playsinline src="${p.video}"></video>
    <div class="lineup">Offense: ${(p.lineup_offense || []).join(", ") || "unknown"}<br>
      Defense: ${(p.lineup_defense || []).join(", ") || "unknown"}</div>
    <div class="labels"></div>
    <div class="row">
      <label><input type="checkbox" class="amb"> ambiguous</label>
      <input type="text" class="note" placeholder="note (optional)">
      <button class="clear">clear</button>
    </div>`;
  const labelBox = card.querySelector(".labels");
  PLAY_TYPES.forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "lab";
    b.dataset.label = t.id;
    b.title = t.definition || t.synergy || "";
    b.innerHTML = `<span class="key">${shortcut(i)}</span>${t.name}`;
    b.onclick = (e) => { e.stopPropagation(); setLabel(p.possession_id, t.id); };
    labelBox.appendChild(b);
  });
  const amb = card.querySelector(".amb");
  amb.onchange = () => setAmbiguous(p.possession_id, amb.checked);
  const note = card.querySelector(".note");
  note.onchange = () => setNote(p.possession_id, note.value);
  card.querySelector(".clear").onclick = (e) => { e.stopPropagation(); clearLabel(p.possession_id); };
  card.addEventListener("click", () => setCurrent(index));
  card.querySelector("video").addEventListener("click", () => setCurrent(index));
  list.appendChild(card);
  cards.push(card);
}

POSSESSIONS.forEach(makeCard);
setCurrent(0);
render();

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(labels)); }
  catch (e) { out.textContent = "localStorage unavailable; use Export/Copy before closing."; }
}

function record(pid) {
  return labels[String(pid)] || {};
}

function setLabel(pid, label) {
  const key = String(pid);
  labels[key] = { ...(labels[key] || {}), play_type: label, updated_at: new Date().toISOString() };
  save(); render();
  if (auto.checked) nextUnlabeled();
}

function setAmbiguous(pid, value) {
  const key = String(pid);
  labels[key] = { ...(labels[key] || {}), ambiguous: !!value, updated_at: new Date().toISOString() };
  save(); render();
}

function setNote(pid, value) {
  const key = String(pid);
  labels[key] = { ...(labels[key] || {}), note: value, updated_at: new Date().toISOString() };
  save();
}

function clearLabel(pid) {
  delete labels[String(pid)];
  save(); render();
}

function setCurrent(index) {
  if (!cards.length) return;
  current = Math.max(0, Math.min(index, cards.length - 1));
  cards.forEach((c, i) => c.classList.toggle("current", i === current));
  cards[current].scrollIntoView({ behavior: "smooth", block: "center" });
  const p = POSSESSIONS[current];
  hint.textContent = `Editing #${p.possession_id} (${current + 1}/${POSSESSIONS.length})`;
}

function nextUnlabeled() {
  for (let step = 1; step <= cards.length; step++) {
    const i = (current + step) % cards.length;
    if (!record(POSSESSIONS[i].possession_id).play_type) { setCurrent(i); return; }
  }
}

function render() {
  let done = 0, ambiguous = 0;
  cards.forEach((card, i) => {
    const p = POSSESSIONS[i];
    const r = record(p.possession_id);
    const labeled = !!r.play_type;
    if (labeled) done++;
    if (r.ambiguous) ambiguous++;
    card.classList.toggle("unlabeled", !labeled);
    card.classList.toggle("current", i === current);
    card.querySelectorAll(".lab").forEach(b => {
      b.classList.toggle("sel", b.dataset.label === r.play_type);
    });
    card.querySelector(".amb").checked = !!r.ambiguous;
    card.querySelector(".note").value = r.note || "";
    card.style.display = hide.checked && labeled ? "none" : "";
  });
  progress.textContent = `${done}/${cards.length} labeled · ${ambiguous} ambiguous`;
  const payload = { event_id: EVENT_ID, video: VIDEO, labels, possessions: POSSESSIONS };
  out.textContent = JSON.stringify(payload, null, 2);
}

document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  const p = POSSESSIONS[current];
  if (!p) return;
  const key = e.key.toLowerCase();
  let idx = -1;
  if (key === "q") idx = 10;
  else if (/^[0-9]$/.test(key)) idx = key === "0" ? 9 : Number(key) - 1;
  if (idx >= 0 && PLAY_TYPES[idx]) {
    setLabel(p.possession_id, PLAY_TYPES[idx].id);
    e.preventDefault(); return;
  }
  if (key === "a") { setAmbiguous(p.possession_id, !record(p.possession_id).ambiguous); e.preventDefault(); return; }
  if (key === "j" || e.key === "ArrowRight") { setCurrent(current + 1); e.preventDefault(); return; }
  if (key === "k" || e.key === "ArrowLeft") { setCurrent(current - 1); e.preventDefault(); return; }
});

document.getElementById("export").onclick = render;
document.getElementById("hide").onchange = render;
document.getElementById("download").onclick = () => {
  render();
  const blob = new Blob([out.textContent], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
document.getElementById("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(out.textContent); hint.textContent = "Copied JSON."; }
  catch (e) { hint.textContent = "Copy failed; select the JSON manually."; }
};
</script>
</body>
</html>
"""


def write_html(out: Path, event_id: str, video: Path, possessions: list[dict],
               play_types: list[dict]) -> None:
    js = lambda value: json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", f"Play labels — {event_id}")
    html = html.replace("__EVENT_ID__", js(event_id))
    html = html.replace("__VIDEO__", js(video.name))
    html = html.replace("__POSSESSIONS__", js(possessions))
    html = html.replace("__PLAY_TYPES__", js(play_types))
    out.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True)
    parser.add_argument("--manual", default="state_layer/examples/manual_game.json")
    parser.add_argument("--video", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--before", type=float, default=14.0)
    parser.add_argument("--after", type=float, default=3.0)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--clip-width", type=int, default=640)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    state_path = Path(args.state)
    state = _read_json(state_path)
    manual = _read_json(Path(args.manual))
    video = Path(args.video or manual["video_path"])
    if not video.exists():
        print(f"[play-labels] video not found: {video}", file=sys.stderr)
        return 1

    fps, total_frames, width, height = video_info(video)
    anchors = manual.get("clock_anchors", [])
    names = state.get("player_names", {})
    possessions = [
        p for p in state.get("possessions", [])
        if p.get("end_reason") not in ("period_end", "game_end")
    ]
    windows = build_windows(
        possessions, fps, total_frames, anchors,
        state.get("lineups", []), names,
        before_s=args.before, after_s=args.after, min_duration_s=args.min_duration,
    )
    selected = select_windows(windows, count=args.count, seed=args.seed)
    print(f"[play-labels] video {width}x{height} @ {fps:.3f} fps, {total_frames} frames")
    print(f"[play-labels] {len(state.get('possessions', []))} possessions, "
          f"{len(windows)} usable windows, {len(selected)} selected")

    event_id = str(state.get("event_id") or manual.get("game_id") or "game")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / "play-labels" / event_id
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    team_names = state.get("team_names", {})
    manifest = []
    for i, window in enumerate(selected, start=1):
        clip_id = f"pos_{i:04d}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        start_s = window["clip_start_frame"] / fps
        duration_s = (window["clip_end_frame"] - window["clip_start_frame"]) / fps
        cut_clip(video, clip_path, start_s, duration_s, args.clip_width, args.skip_existing)
        offense_id = int(window["offense_team_id"])
        defense_id = int(window["defense_team_id"])
        manifest.append({
            "clip_id": clip_id,
            "video": f"clips/{clip_id}.mp4",
            "possession_id": window["possession_id"],
            "period": window["period"],
            "offense_team_id": offense_id,
            "defense_team_id": defense_id,
            "offense": team_names.get(str(offense_id), str(offense_id)),
            "defense": team_names.get(str(defense_id), str(defense_id)),
            "points": window.get("points", 0),
            "end_reason": window.get("end_reason"),
            "confidence": window.get("confidence"),
            "start_frame": window["start_frame"],
            "end_frame": window["end_frame"],
            "clip_start_frame": window["clip_start_frame"],
            "clip_end_frame": window["clip_end_frame"],
            "clip_start_s": round(start_s, 3),
            "duration_s": round(duration_s, 3),
            "lineup_offense": window.get("lineup_offense", []),
            "lineup_defense": window.get("lineup_defense", []),
        })
        print(f"[play-labels] {clip_id}: possession {window['possession_id']} "
              f"Q{window['period']} {start_s:.1f}s +{duration_s:.1f}s")

    taxonomy = _read_json(REPO_ROOT / "research" / "ontology" / "play_taxonomy.json")
    play_types = [
        {"id": t["id"], "name": t["name"], "synergy": t.get("synergy", ""),
         "definition": t.get("definition", "")}
        for t in taxonomy["play_types"]
    ]
    (out_dir / "possessions.json").write_text(
        json.dumps({
            "event_id": event_id,
            "video": str(video),
            "fps": fps,
            "count": len(manifest),
            "possessions": manifest,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_html(out_dir / "labels.html", event_id, video, manifest, play_types)
    (out_dir / "README.md").write_text(
        f"""# Play label set — `{event_id}`

{len(manifest)} possession clips from `{video.name}`.

## Run

```powershell
python -m http.server 8765 --directory outputs/play-labels/{event_id}
```

Open <http://localhost:8765/labels.html>.

Label the primary offensive action. Keyboard: `1-9`, `0`, `Q` pick a play type; `A`
toggles ambiguous; `J`/`K` or arrows move. Export or download `labels.json` when done.

Protocol: `docs/PLAY_LABELING.md`.

The clips are local artifacts and are not committed. The label file should be saved next
to this README; the next step is to validate it against the frozen taxonomy and use it for
the play-recognition baseline.
""",
        encoding="utf-8",
    )
    print(f"[play-labels] wrote {out_dir / 'possessions.json'}")
    print(f"[play-labels] wrote {out_dir / 'labels.html'}")
    print(f"[play-labels] wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
