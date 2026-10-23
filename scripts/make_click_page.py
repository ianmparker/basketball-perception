"""Generate a self-contained browser page for clicking court landmarks.

Usage:
    python scripts/make_click_page.py --image frame.jpg --out frame_click.html

Open the HTML file in a browser, click the landmarks in order, and copy the result.
The page maps each click back to the original image pixels, even when the image is
scaled to fit the window.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

# (name, COURT_KPS_FT index, (x_ft, y_ft))
LANDMARKS = [
    ("near baseline / top sideline", 0, (0.0, 50.0)),
    ("near baseline / bottom sideline", 5, (0.0, 0.0)),
    ("halfcourt / top sideline", 7, (47.0, 50.0)),
    ("halfcourt / bottom sideline", 6, (47.0, 0.0)),
    ("near FT line / paint top", 8, (19.0, 33.0)),
    ("near FT line / paint bottom", 9, (19.0, 17.0)),
    ("far FT line / paint top", 16, (75.0, 33.0)),
    ("far FT line / paint bottom", 17, (75.0, 17.0)),
    ("far baseline / top sideline", 15, (94.0, 50.0)),
    ("far baseline / bottom sideline", 10, (94.0, 0.0)),
]

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 18px; background: #111; color: #eee; }}
  h2 {{ margin: 0 0 6px; }}
  #hint {{ color: #ffd54a; margin: 8px 0 12px; font-size: 15px; }}
  #wrap {{ position: relative; display: inline-block; max-width: 100%; }}
  canvas {{ max-width: 100%; height: auto; cursor: crosshair; border: 1px solid #444; display: block; }}
  button {{ margin: 4px 6px 4px 0; padding: 6px 10px; background: #263238; color: #eee;
            border: 1px solid #546e7a; border-radius: 4px; cursor: pointer; }}
  button:hover {{ background: #37474f; }}
  pre {{ background: #1b1b1b; border: 1px solid #333; padding: 10px; white-space: pre-wrap;
         max-height: 320px; overflow: auto; }}
  .small {{ color: #9e9e9e; font-size: 12px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="small">Click the landmarks in order. If one is not visible, press Skip.
The page records coordinates in the original image pixels.</div>
<div id="hint"></div>
<div id="wrap"><canvas id="c"></canvas></div>
<div>
  <button id="undo">Undo</button>
  <button id="skip">Skip</button>
  <button id="reset">Reset</button>
  <button id="copy">Copy result</button>
</div>
<pre id="out">No points yet.</pre>
<script>
const LANDMARKS = {landmarks};
const IMG = "data:image/jpeg;base64,{image}";
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const hint = document.getElementById('hint');
const out = document.getElementById('out');
let pts = [];
let current = 0;
const img = new Image();
img.onload = () => {{
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  draw();
  update();
}};
img.src = IMG;

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  pts.forEach((p, i) => {{
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, 2 * Math.PI);
    ctx.fillStyle = "#00e676";
    ctx.fill();
    ctx.fillStyle = "#000";
    ctx.font = "bold 12px system-ui";
    ctx.fillText(String(i + 1), p.x + 10, p.y + 4);
  }});
}}

function update() {{
  if (current < LANDMARKS.length) {{
    hint.textContent = `Click ${{current + 1}}/${{LANDMARKS.length}}: ${{LANDMARKS[current].name}}`;
  }} else {{
    hint.textContent = "All landmarks clicked. Copy the result and send it back.";
  }}
  const payload = {{
    image: {image_name!r},
    points: pts,
    matches: pts.map(p => `${{p.index}}:${{p.x}},${{p.y}}`)
  }};
  out.textContent = pts.length ? JSON.stringify(payload, null, 2) : "No points yet.";
}}

canvas.addEventListener('click', (e) => {{
  if (current >= LANDMARKS.length) return;
  const r = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * canvas.width / r.width);
  const y = Math.round((e.clientY - r.top) * canvas.height / r.height);
  pts.push({{ name: LANDMARKS[current].name, index: LANDMARKS[current].index, x, y }});
  current += 1;
  draw();
  update();
}});

document.getElementById('undo').onclick = () => {{
  if (!pts.length) return;
  pts.pop();
  current = Math.max(0, current - 1);
  draw();
  update();
}};
document.getElementById('skip').onclick = () => {{
  if (current < LANDMARKS.length) current += 1;
  update();
}};
document.getElementById('reset').onclick = () => {{
  pts = [];
  current = 0;
  draw();
  update();
}};
document.getElementById('copy').onclick = async () => {{
  try {{ await navigator.clipboard.writeText(out.textContent); hint.textContent = "Copied."; }}
  catch (err) {{ hint.textContent = "Copy failed; select the text manually."; }}
}};
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Court Calibration")
    args = parser.parse_args()

    image_path = Path(args.image)
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    html = TEMPLATE.format(
        title=args.title,
        landmarks=json.dumps(LANDMARKS),
        image=data,
        image_name=image_path.name,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[click-page] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
