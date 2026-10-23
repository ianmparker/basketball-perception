"""One-command broadcast analysis demo.

Runs the pipeline on a broadcast clip, then builds the static demo page. This is
the current product bet: make the broadcast path visible and judgeable, not
perfect.

    python scripts/broadcast_demo.py --video data/sample/videos/broadcast_74.mp4
    python scripts/broadcast_demo.py --video C:\\path\\game.mp4 --frames 1800
    python scripts/broadcast_demo.py --run-json outputs/run/clip.json \
        --render outputs/run/render.mp4 --out-dir outputs/run

Open ``<out-dir>/index.html`` or serve it:

    python -m http.server 8000 --directory <out-dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_demo_page import build_demo  # noqa: E402
from export_event_clips import export_clips  # noqa: E402


def _run_pipeline(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, stream its output, and keep a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def _handoff(out_dir: Path, summary: dict, runtime_s: float, command: str) -> None:
    counts = summary["event_counts"]
    lines = [
        "# Broadcast Demo Handoff",
        "",
        f"**Generated:** {datetime.now().astimezone().isoformat(timespec='minutes')}  ",
        f"**Source:** `{summary['source'] or 'unknown'}`  ",
        f"**Frames:** {summary['frame_count']} at {summary['fps']} fps "
        f"({summary['duration_s']:.1f} s)  ",
        f"**Pipeline runtime:** {runtime_s / 60:.1f} min",
        "",
        "## What is here",
        "",
        f"- `index.html` — the demo page",
        f"- `assets/demo.mp4` — browser-playable annotated render",
        f"- `clip.json` — the versioned ClipResult",
        f"- `shot_chart.svg` — shot locations",
        f"- `report.md` — one-page summary",
        f"- `summary.json` — machine-readable demo summary",
        "- `pipeline.log` — full pipeline output (if the pipeline ran here)",
        "",
        "## Event counts",
        "",
        "| Event | Count |",
        "|---|---|",
        f"| Shots | {counts.get('shots', 0)} |",
        f"| Screens | {counts.get('screens', 0)} |",
        f"| Passes | {counts.get('passes', 0)} |",
        f"| Interceptions | {counts.get('interceptions', 0)} |",
        "",
        "## Honest limitations",
        "",
        "- Shot timing is approximate; the detector fires near the gather.",
        "- Team ID and possession can drift; the page shows what the run produced.",
        "- This uses local research-only broadcast footage and is not redistributed.",
        "- Handheld/tripod footage is out of scope for this cycle.",
        "",
        "## Open it",
        "",
        "```powershell",
        f'start "" "{out_dir / "index.html"}"',
        "# or serve it:",
        f'python -m http.server 8000 --directory "{out_dir}"',
        "```",
        "",
        "## Reproduce",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## What to improve next",
        "",
        "1. Look at the event timeline and mark which shots are real.",
        "2. If shots are early, fix localization before adding more event types.",
        "3. If the ball or team ID is the weak link, fix that before a bigger demo.",
        "",
    ]
    (out_dir / "HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=Path, default=None, help="Source broadcast clip")
    parser.add_argument("--frames", type=int, default=1200, help="Max frames (default 1200)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--title", default="Broadcast Analysis Demo")
    parser.add_argument("--max-width", type=int, default=1600, help="Video transcode width cap")
    parser.add_argument("--run-json", type=Path, default=None, help="Use an existing ClipResult JSON")
    parser.add_argument("--render", type=Path, default=None, help="Existing rendered MP4")
    parser.add_argument("--serve", action="store_true", help="Serve the demo and block")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--event-clips", action="store_true",
                        help="Cut short clips around each detected event")
    args = parser.parse_args(argv)

    if args.run_json is None and args.video is None:
        parser.error("pass --video or --run-json")

    if args.out_dir is None:
        if args.run_json is not None:
            args.out_dir = args.run_json.parent / "demo"
        else:
            assert args.video is not None
            args.out_dir = REPO_ROOT / "outputs" / "broadcast_demo" / args.video.stem
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if args.run_json is not None:
        run_json = args.run_json
        render = args.render
        stats = None
        command = (
            f'python scripts/broadcast_demo.py --run-json "{run_json}"'
            + (f' --render "{render}"' if render else "")
            + f' --out-dir "{out_dir}"'
        )
    else:
        assert args.video is not None
        if not args.video.is_file():
            print(f"[demo] ERROR: video not found: {args.video}", file=sys.stderr)
            return 2
        run_json = out_dir / "clip.json"
        render = out_dir / "render.mp4"
        stats = out_dir / "stats.csv"
        log_path = out_dir / "pipeline.log"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "inference" / "run_video.py"),
            "--video", str(args.video),
            "--frames", str(args.frames),
            "--out", str(render),
            "--json-out", str(run_json),
            "--stats-out", str(stats),
            "--events", "--pose", "--team-id", "--possession",
            "--court-view-gate", "--offline-events",
        ]
        command = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        print(f"[demo] running pipeline on {args.video} ({args.frames} frames max)")
        print(f"[demo] log: {log_path}")
        code = _run_pipeline(cmd, log_path)
        if code != 0 or not run_json.is_file():
            print(f"[demo] pipeline failed (exit {code}); see {log_path}", file=sys.stderr)
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print("\n".join(tail), file=sys.stderr)
            return 1

    runtime_s = time.perf_counter() - started
    if args.event_clips and render is not None and render.is_file():
        try:
            export_clips(run_json, render, out_dir / "clips", max_clips=12)
        except SystemExit as exc:
            print(f"[demo] event clips skipped: {exc}", file=sys.stderr)
    summary = build_demo(
        run_json,
        render if render and render.is_file() else None,
        out_dir,
        title=args.title,
        max_width=args.max_width,
        stats=stats if stats and stats.is_file() else None,
    )
    _handoff(out_dir, summary, runtime_s, command)

    print(f"[demo] wrote {out_dir / 'index.html'}")
    print(f"[demo] events: {summary['event_counts']}")
    print(f"[demo] shots: {len(summary['shots'])}")
    print(f"[demo] open: start \"\" \"{out_dir / 'index.html'}\"")
    print(f"[demo] serve: python -m http.server {args.port} --directory \"{out_dir}\"")

    if args.serve:
        import http.server
        import os

        os.chdir(out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"[demo] serving http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\n[demo] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
