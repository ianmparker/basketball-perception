"""Build a static broadcast-analysis demo page from a ClipResult JSON.

The page is self-contained: no CDN, no server required. It shows the annotated
video (H.264), summary cards, a shot chart, an event timeline with seek buttons,
and links to the JSON/CSV/report.

    python scripts/build_demo_page.py \
        --run-json data/results/kyrie/kyrie_fps.json \
        --video data/results/kyrie/kyrie_fps.mp4 \
        --out-dir outputs/broadcast_demo/kyrie

The script deliberately reads only the versioned ClipResult fields. Missing
fields become "not available" instead of crashing, because the schema is still
evolving and an old run should still be viewable.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from string import Template

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
HOOPS_FT = ((5.25, 25.0), (88.75, 25.0))
EVENT_KINDS = ("shots", "screens", "pnrs", "dunks", "passes", "interceptions")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def _time_s(frame: int, fps: float) -> float:
    return round(frame / fps, 2) if fps else 0.0


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"


def _location_valid(x_ft: float, y_ft: float) -> bool:
    """Reject court-boundary projections that mean the ball was off-camera."""
    return 0.5 < x_ft < COURT_LENGTH_FT - 0.5 and 0.5 < y_ft < COURT_WIDTH_FT - 0.5


def _event_rows(data: dict, fps: float) -> list[dict]:
    rows: list[dict] = []

    for shot in data.get("shots") or []:
        frame = int(shot.get("frame_start", 0) or 0)
        distance = shot.get("shot_distance_ft")
        detail = shot.get("shot_type") or "shot"
        if distance is not None:
            detail += f" · {float(distance):.1f} ft"
        if shot.get("shooter_track_id") is not None:
            detail += f" · track {shot['shooter_track_id']}"
        court = shot.get("ball_court_ft")
        if not court or not _location_valid(_f(court[0]), _f(court[1])):
            detail += " · location unavailable"
        rows.append({
            "kind": "Shot",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": _f(shot.get("confidence")),
            "cls": "shot",
        })

    for screen in data.get("screens") or []:
        frame = int(screen.get("frame", 0) or 0)
        detail = screen.get("screen_type") or "screen"
        if screen.get("screener_id") is not None and screen.get("cutter_id") is not None:
            detail += f" · {screen['screener_id']} → {screen['cutter_id']}"
        if screen.get("separation_ft") is not None:
            detail += f" · {float(screen['separation_ft']):.1f} ft"
        rows.append({
            "kind": "Screen",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": _f(screen.get("confidence")),
            "cls": "screen",
        })

    for pnr in data.get("pnrs") or []:
        frame = int(pnr.get("frame", 0) or 0)
        detail = pnr.get("outcome") or "pick-and-roll"
        if pnr.get("ball_handler_id") is not None:
            detail += f" · handler {pnr['ball_handler_id']}"
        rows.append({
            "kind": "PnR",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": _f(pnr.get("confidence")),
            "cls": "pnr",
        })

    for dunk in data.get("dunks") or []:
        frame = int(dunk.get("frame", 0) or 0)
        detail = f"hoop {dunk.get('hoop_side', '?')} · track {dunk.get('player_track_id', '?')}"
        rows.append({
            "kind": "Dunk",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": 1.0,
            "cls": "dunk",
        })

    for pass_event in data.get("passes") or []:
        frame = int(pass_event.get("frame_idx", 0) or 0)
        detail = f"team {pass_event.get('team_id', '?')} · {pass_event.get('from_track_id', '?')} → {pass_event.get('to_track_id', '?')}"
        rows.append({
            "kind": "Pass",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": 1.0,
            "cls": "pass",
        })

    for interception in data.get("interceptions") or []:
        frame = int(interception.get("frame_idx", 0) or 0)
        detail = f"team {interception.get('team_id', '?')} · {interception.get('from_track_id', '?')} → {interception.get('to_track_id', '?')}"
        rows.append({
            "kind": "Interception",
            "frame": frame,
            "time_s": _time_s(frame, fps),
            "detail": detail,
            "confidence": 1.0,
            "cls": "interception",
        })

    rows.sort(key=lambda row: (row["frame"], row["kind"]))
    return rows


def summarize(data: dict, title: str = "Broadcast Analysis Demo") -> dict:
    metadata = data.get("metadata") or {}
    frames = data.get("frames") or []
    frame_count = len(frames) or int(metadata.get("frame_count") or 0)
    fps = _f(metadata.get("fps"), 30.0) or 30.0
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)

    ball_tracked = 0
    ball_detected = 0
    ball_interpolated = 0
    court_view = 0
    possession_frames = 0
    players_per_frame: list[int] = []
    team_counts: dict[str, int] = {}
    homography_rmse: list[float] = []
    track_distances: dict[int, float] = {}

    for frame in frames:
        ball = frame.get("ball") or {}
        if ball.get("tracked"):
            ball_tracked += 1
        if ball.get("det_box") is not None:
            ball_detected += 1
        if ball.get("interpolated"):
            ball_interpolated += 1
        if frame.get("court_view"):
            court_view += 1
        if frame.get("possession_team_id") is not None:
            possession_frames += 1
        players = frame.get("players") or []
        players_per_frame.append(len(players))
        for player in players:
            team = player.get("team_id")
            key = "unassigned" if team is None else str(team)
            team_counts[key] = team_counts.get(key, 0) + 1
            track_id = player.get("track_id")
            if track_id is not None:
                track_distances[int(track_id)] = max(
                    track_distances.get(int(track_id), 0.0),
                    _f(player.get("distance_traveled_ft")),
                )
        quality = frame.get("homography_quality") or {}
        rmse = quality.get("rmse_ft")
        if rmse is not None:
            homography_rmse.append(_f(rmse))

    events = _event_rows(data, fps)
    event_counts = {kind: len(data.get(kind) or []) for kind in EVENT_KINDS}
    top_players = sorted(
        ({"track_id": track_id, "distance_ft": round(distance, 1)}
         for track_id, distance in track_distances.items() if distance > 0),
        key=lambda item: item["distance_ft"],
        reverse=True,
    )[:8]

    shots = []
    for shot in data.get("shots") or []:
        court = shot.get("ball_court_ft")
        x_ft = _f(court[0]) if court else float("nan")
        y_ft = _f(court[1]) if court else float("nan")
        valid = bool(court) and _location_valid(x_ft, y_ft)
        shots.append({
            "frame": int(shot.get("frame_start", 0) or 0),
            "time_s": _time_s(int(shot.get("frame_start", 0) or 0), fps),
            "x_ft": x_ft,
            "y_ft": y_ft,
            "location_valid": valid,
            "shot_type": shot.get("shot_type") or "unknown",
            "distance_ft": _f(shot.get("shot_distance_ft")),
            "confidence": _f(shot.get("confidence")),
            "shooter_track_id": shot.get("shooter_track_id"),
        })
    shots_located = sum(1 for shot in shots if shot["location_valid"])

    return {
        "title": title,
        "source": metadata.get("video") or "",
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": round(frame_count / fps, 1) if fps else 0.0,
        "schema_version": data.get("schema_version") or "unknown",
        "created_at": metadata.get("created_at") or "",
        "event_counts": event_counts,
        "events": events,
        "shots": shots,
        "shots_located": shots_located,
        "shots_unlocated": len(shots) - shots_located,
        "ball": {
            "tracked_frames": ball_tracked,
            "detected_frames": ball_detected,
            "interpolated_frames": ball_interpolated,
            "tracked_pct": _pct(ball_tracked, frame_count),
            "detected_pct": _pct(ball_detected, frame_count),
        },
        "players": {
            "mean_per_frame": round(sum(players_per_frame) / len(players_per_frame), 1) if players_per_frame else 0.0,
            "max_per_frame": max(players_per_frame) if players_per_frame else 0,
        },
        "teams": {
            "assignment_available": bool(data.get("team_assignment_available")),
            "counts": team_counts,
        },
        "possession": {
            "frames": possession_frames,
            "pct": _pct(possession_frames, frame_count),
        },
        "court_view": {
            "frames": court_view,
            "pct": _pct(court_view, frame_count),
        },
        "homography": {
            "median_rmse_ft": round(median(homography_rmse), 3) if homography_rmse else None,
        },
        "top_players": top_players,
    }


def _court_xy(x_ft: float, y_ft: float, width: int, height: int, pad: int = 24) -> tuple[float, float]:
    sx = pad + (x_ft / COURT_LENGTH_FT) * (width - 2 * pad)
    sy = pad + (y_ft / COURT_WIDTH_FT) * (height - 2 * pad)
    return sx, sy


def shot_chart_svg(shots: list[dict], width: int = 960, height: int = 520) -> str:
    pad = 24
    court_w = width - 2 * pad
    court_h = height - 2 * pad

    def pt(x_ft: float, y_ft: float) -> str:
        sx, sy = _court_xy(x_ft, y_ft, width, height, pad)
        return f"{sx:.1f},{sy:.1f}"

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Shot chart on a basketball court">',
        f'<rect x="{pad}" y="{pad}" width="{court_w}" height="{court_h}" rx="6" '
        f'fill="#0f1a24" stroke="#5b7083" stroke-width="2"/>',
        f'<line x1="{pad + court_w / 2:.1f}" y1="{pad}" x2="{pad + court_w / 2:.1f}" '
        f'y2="{pad + court_h}" stroke="#5b7083" stroke-width="1.5"/>',
        f'<circle cx="{pad + court_w / 2:.1f}" cy="{pad + court_h / 2:.1f}" '
        f'r="{6 / COURT_WIDTH_FT * court_h:.1f}" fill="none" stroke="#5b7083" stroke-width="1.5"/>',
    ]

    # Paint boxes and free-throw circles.
    for x0 in (0.0, 75.0):
        x1 = x0 + 19.0
        y0, y1 = 17.0, 33.0
        lines.append(
            f'<rect x="{_court_xy(x0, y0, width, height, pad)[0]:.1f}" '
            f'y="{_court_xy(x0, y0, width, height, pad)[1]:.1f}" '
            f'width="{(19.0 / COURT_LENGTH_FT) * court_w:.1f}" '
            f'height="{(16.0 / COURT_WIDTH_FT) * court_h:.1f}" '
            f'fill="none" stroke="#5b7083" stroke-width="1.5"/>'
        )
        cx, cy = _court_xy(x1, 25.0, width, height, pad)
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" '
            f'r="{6 / COURT_WIDTH_FT * court_h:.1f}" fill="none" stroke="#5b7083" stroke-width="1.5"/>'
        )

    # Three-point arcs: sample the arc and keep it inside the court.
    for hoop_x, direction in ((HOOPS_FT[0][0], 1.0), (HOOPS_FT[1][0], -1.0)):
        points = []
        for step in range(41):
            y = 3.25 + (46.75 - 3.25) * step / 40
            dx = math.sqrt(max(0.0, 23.75 ** 2 - (y - 25.0) ** 2))
            x = hoop_x + direction * dx
            if 0.0 <= x <= COURT_LENGTH_FT:
                points.append(pt(x, y))
        if len(points) >= 2:
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="#8aa0b4" '
                f'stroke-width="2" stroke-dasharray="7 5"/>'
            )
        # Baseline segments to the arc.
        lines.append(
            f'<line x1="{pt(hoop_x, 3.25).split(",")[0]}" y1="{pt(hoop_x, 3.25).split(",")[1]}" '
            f'x2="{pt(0 if direction > 0 else 94, 3.25).split(",")[0]}" '
            f'y2="{pt(0 if direction > 0 else 94, 3.25).split(",")[1]}" '
            f'stroke="#8aa0b4" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        lines.append(
            f'<line x1="{pt(hoop_x, 46.75).split(",")[0]}" y1="{pt(hoop_x, 46.75).split(",")[1]}" '
            f'x2="{pt(0 if direction > 0 else 94, 46.75).split(",")[0]}" '
            f'y2="{pt(0 if direction > 0 else 94, 46.75).split(",")[1]}" '
            f'stroke="#8aa0b4" stroke-width="2" stroke-dasharray="7 5"/>'
        )

    # Hoops and backboards.
    for hoop_x, hoop_y in HOOPS_FT:
        hx, hy = _court_xy(hoop_x, hoop_y, width, height, pad)
        lines.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="7" fill="none" stroke="#f59e0b" stroke-width="3"/>')
        bx = hoop_x - 1.25 if hoop_x < 47 else hoop_x + 1.25
        bx1, by1 = _court_xy(bx, hoop_y - 3, width, height, pad)
        bx2, by2 = _court_xy(bx, hoop_y + 3, width, height, pad)
        lines.append(f'<line x1="{bx1:.1f}" y1="{by1:.1f}" x2="{bx2:.1f}" y2="{by2:.1f}" stroke="#e5e7eb" stroke-width="3"/>')

    # Shots. Only plot locations that are inside the court; a boundary
    # projection means the ball was off-camera or the projection failed.
    plotted = [shot for shot in shots if shot.get("location_valid", True)]
    colors = {"3_pointer": "#38bdf8", "mid_range": "#f59e0b", "layup": "#34d399", "dunk": "#a78bfa", "unknown": "#cbd5e1"}
    for index, shot in enumerate(plotted, start=1):
        sx, sy = _court_xy(shot["x_ft"], shot["y_ft"], width, height, pad)
        color = colors.get(shot.get("shot_type"), colors["unknown"])
        radius = 7 + 6 * max(0.0, min(1.0, shot.get("confidence", 0.0)))
        label = html.escape(f"Shot {index}: {shot.get('shot_type', 'unknown')}, {shot.get('distance_ft', 0):.1f} ft")
        lines.append(f'<g><title>{label}</title>')
        lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="2"/>')
        lines.append(f'<text x="{sx:.1f}" y="{sy - radius - 5:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" font-family="Segoe UI, Arial, sans-serif">{index}</text>')
        lines.append('</g>')

    if not plotted:
        lines.append(
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            f'fill="#8aa0b4" font-size="18" font-family="Segoe UI, Arial, sans-serif">'
            f'No shot locations available in this clip</text>'
        )

    # Legend.
    legend_x = pad + 8
    legend_y = height - 14
    for i, (label, color) in enumerate((("3PT", colors["3_pointer"]), ("mid", colors["mid_range"]), ("layup", colors["layup"]))):
        lx = legend_x + i * 90
        lines.append(f'<circle cx="{lx}" cy="{legend_y - 4}" r="6" fill="{color}"/>')
        lines.append(f'<text x="{lx + 10}" y="{legend_y}" fill="#cbd5e1" font-size="12" font-family="Segoe UI, Arial, sans-serif">{label}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def _chip(label: str, value: str) -> str:
    return f'<span class="chip"><b>{html.escape(value)}</b> {html.escape(label)}</span>'


def _summary_cards(summary: dict) -> str:
    event_counts = summary["event_counts"]
    shots_total = event_counts.get("shots", 0)
    shots_located = summary.get("shots_located", shots_total)
    shots_note = (
        f"{shots_located} located"
        if shots_located == shots_total
        else f"{shots_located} located, {shots_total - shots_located} no location"
    )
    cards = [
        ("Shots", str(shots_total), shots_note),
        ("Screens", str(event_counts.get("screens", 0)), "detected screen events"),
        ("Passes", str(event_counts.get("passes", 0)), "detected pass events"),
        ("Interceptions", str(event_counts.get("interceptions", 0)), "detected interception events"),
        ("Ball detected", f'{summary["ball"]["detected_pct"]:.0f}%', f'track covers {summary["ball"]["tracked_pct"]:.0f}% after smoothing'),
        ("Players / frame", f'{summary["players"]["mean_per_frame"]:.1f}', "mean detections"),
    ]
    return "\n".join(
        f'<div class="card"><div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-note">{html.escape(note)}</div></div>'
        for label, value, note in cards
    )


def _event_table(events: list[dict]) -> str:
    if not events:
        return '<p class="muted">No events detected in this clip. The page still shows the tracking and tactical view.</p>'
    rows = []
    for event in events[:80]:
        confidence = event.get("confidence")
        conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
        rows.append(
            f'<tr class="event-{html.escape(event["cls"])}">'
            f'<td class="time">{_fmt_time(event["time_s"])}</td>'
            f'<td><span class="pill {html.escape(event["cls"])}">{html.escape(event["kind"])}</span></td>'
            f'<td>{html.escape(event["detail"])}</td>'
            f'<td class="num">{conf_text}</td>'
            f'<td><button class="seek" data-seek="{event["time_s"]:.2f}">Jump</button></td>'
            f'</tr>'
        )
    more = ""
    if len(events) > 80:
        more = f'<p class="muted">Showing the first 80 of {len(events)} events.</p>'
    return (
        '<table class="events"><thead><tr><th>Time</th><th>Event</th><th>Detail</th>'
        '<th>Conf.</th><th></th></tr></thead><tbody>' + "\n".join(rows) + "</tbody></table>" + more
    )


def _team_rows(summary: dict) -> str:
    counts = summary["teams"]["counts"]
    if not counts:
        return '<p class="muted">Team assignment was not available for this run.</p>'
    total = sum(counts.values()) or 1
    rows = []
    for team, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            f'<tr><td>{html.escape(team)}</td><td class="num">{count}</td>'
            f'<td class="num">{100 * count / total:.1f}%</td></tr>'
        )
    return (
        '<table class="mini"><thead><tr><th>Team</th><th>Player-frames</th><th>Share</th></tr></thead>'
        '<tbody>' + "\n".join(rows) + "</tbody></table>"
    )


def _player_rows(summary: dict) -> str:
    players = summary.get("top_players") or []
    if not players:
        return '<p class="muted">Distance was not available for this run.</p>'
    rows = "\n".join(
        f'<tr><td>#{player["track_id"]}</td><td class="num">{player["distance_ft"]:.1f} ft</td></tr>'
        for player in players
    )
    return (
        '<table class="mini"><thead><tr><th>Track</th><th>Max distance</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def build_report(summary: dict, run_json: Path, video: Path | None, out_dir: Path) -> str:
    event_counts = summary["event_counts"]
    lines = [
        f'# {summary["title"]}',
        "",
        f'**Source:** `{summary["source"] or "unknown"}`  ',
        f'**Duration:** {summary["duration_s"]:.1f} s ({summary["frame_count"]} frames at {summary["fps"]} fps)  ',
        f'**Resolution:** {summary["width"]}×{summary["height"]}  ',
        f'**Schema:** {summary["schema_version"]}',
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f'| Shots | {event_counts.get("shots", 0)} '
        f'({summary.get("shots_located", 0)} located, {summary.get("shots_unlocated", 0)} unavailable) |',
        f'| Screens | {event_counts.get("screens", 0)} |',
        f'| Passes | {event_counts.get("passes", 0)} |',
        f'| Interceptions | {event_counts.get("interceptions", 0)} |',
        f'| Ball tracked | {summary["ball"]["tracked_pct"]:.1f}% of frames |',
        f'| Ball detected | {summary["ball"]["detected_pct"]:.1f}% of frames |',
        f'| Players / frame | {summary["players"]["mean_per_frame"]:.1f} mean, {summary["players"]["max_per_frame"]} max |',
        f'| Court view | {summary["court_view"]["pct"]:.1f}% of frames |',
        f'| Possession | {summary["possession"]["pct"]:.1f}% of frames |',
        f'| Homography median RMSE | {summary["homography"]["median_rmse_ft"] if summary["homography"]["median_rmse_ft"] is not None else "n/a"} ft |',
        "",
    ]
    if summary["shots"]:
        lines += [
            "## Shots",
            "",
            "| # | Time | Type | Distance | Confidence | Location |",
            "|---|---|---|---|---|---|",
        ]
        for index, shot in enumerate(summary["shots"], start=1):
            location = "ok" if shot.get("location_valid", True) else "unavailable"
            lines.append(
                f'| {index} | {_fmt_time(shot["time_s"])} | {shot["shot_type"]} | '
                f'{shot["distance_ft"]:.1f} ft | {shot["confidence"]:.2f} | {location} |'
            )
        lines.append("")
    lines += [
        "## Known limitations",
        "",
        "- Event detection is heuristic; shots can be early, missed, or duplicated.",
        "- Team ID and possession can drift on broadcast footage.",
        "- The demo uses local research-only footage and is not redistributed.",
        "- This is a broadcast demo, not a handheld/tripod product build.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        f'python scripts/build_demo_page.py --run-json "{run_json}"'
        + (f' --video "{video}"' if video else "")
        + f' --out-dir "{out_dir}"',
        "```",
        "",
    ]
    return "\n".join(lines)


HTML_TEMPLATE = Template(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  :root {
    --bg: #080c11;
    --panel: #101923;
    --panel-2: #152231;
    --line: #243447;
    --text: #e6edf5;
    --muted: #8fa3b8;
    --accent: #f59e0b;
    --green: #34d399;
    --blue: #38bdf8;
    --red: #fb7185;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: radial-gradient(circle at 20% 0%, #142334 0%, var(--bg) 45%);
    color: var(--text); font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
    line-height: 1.5;
  }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 28px 22px 60px; }
  header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 18px; margin-bottom: 22px; }
  h1 { font-size: clamp(26px, 4vw, 44px); margin: 0 0 6px; letter-spacing: -0.02em; }
  h2 { font-size: 20px; margin: 0 0 12px; }
  .sub { color: var(--muted); margin: 0; max-width: 760px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chip { background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; padding: 6px 11px; font-size: 12px; color: var(--muted); }
  .chip b { color: var(--text); }
  .grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 18px; }
  .panel { background: linear-gradient(180deg, rgba(255,255,255,.025), transparent), var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }
  video { width: 100%; border-radius: 10px; background: #000; display: block; }
  .video-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
  #demo-video:fullscreen { width: 100vw; height: 100vh; object-fit: contain; background: #000; border-radius: 0; }
  #demo-video:-webkit-full-screen { width: 100vw; height: 100vh; object-fit: contain; background: #000; border-radius: 0; }
  body.theater { overflow: hidden; }
  body.theater .wrap { max-width: none; padding: 0; }
  body.theater header, body.theater .section, body.theater .two, body.theater footer { display: none !important; }
  body.theater .grid { display: block; margin: 0; }
  body.theater .video-panel { position: fixed; inset: 0; z-index: 9999; border: 0; border-radius: 0; padding: 0; background: #000; display: flex; align-items: center; justify-content: center; }
  body.theater .video-panel video { width: 100vw; height: 100vh; object-fit: contain; border-radius: 0; }
  body.theater .video-panel .video-toolbar { position: fixed; top: 10px; right: 10px; z-index: 10000; margin: 0; background: rgba(8, 12, 17, .78); padding: 6px 8px; border-radius: 8px; }
  body.theater .grid > .panel:not(.video-panel) { display: none; }
  .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .card { background: var(--panel-2); border: 1px solid var(--line); border-radius: 11px; padding: 12px; }
  .card-value { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .card-label { font-size: 13px; color: var(--text); }
  .card-note { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .section { margin-top: 18px; }
  .muted { color: var(--muted); }
  .chart { width: 100%; height: auto; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .time { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
  .pill { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
  .pill.shot { background: rgba(245,158,11,.16); color: var(--accent); }
  .pill.screen { background: rgba(56,189,248,.14); color: var(--blue); }
  .pill.pass { background: rgba(52,211,153,.14); color: var(--green); }
  .pill.interception { background: rgba(251,113,133,.14); color: var(--red); }
  .pill.dunk, .pill.pnr { background: rgba(167,139,250,.16); color: #c4b5fd; }
  .seek { background: transparent; border: 1px solid var(--line); color: var(--text); border-radius: 8px; padding: 4px 9px; cursor: pointer; font-size: 12px; }
  .seek:hover { border-color: var(--accent); color: var(--accent); }
  .two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .downloads a { color: var(--blue); text-decoration: none; margin-right: 14px; }
  .downloads a:hover { text-decoration: underline; }
  .clip-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
  .clip-card { background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
  .clip-card video { width: 100%; border-radius: 8px; background: #000; display: block; }
  .clip-meta { color: var(--muted); font-size: 12px; padding: 7px 2px 2px; }
  .clip-meta b { color: var(--text); }
  .callout { border-left: 3px solid var(--accent); padding: 10px 14px; background: rgba(245,158,11,.07); border-radius: 0 10px 10px 0; }
  footer { margin-top: 26px; color: var(--muted); font-size: 12px; }
  code { background: #0b121a; border: 1px solid var(--line); border-radius: 6px; padding: 2px 6px; font-size: 12px; }
  @media (max-width: 900px) { .grid, .two { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>$title</h1>
      <p class="sub">A broadcast clip pushed through the perception pipeline: ball and player tracking, court calibration, team/possession, event detection, and a tactical view.</p>
      <div class="chips">
        $chips
      </div>
    </div>
  </header>

  <div class="grid">
    <section class="panel video-panel">
      $video_toolbar
      $video_block
    </section>
    <section class="panel">
      <h2>At a glance</h2>
      <div class="cards">
        $cards
      </div>
      <p class="muted" style="font-size:12px;margin:12px 0 0;">Ball coverage counts a frame as tracked when the smoothed track has a position; detected means a raw detector box backed that frame.</p>
    </section>
  </div>

  <section class="panel section">
    <h2>Shot chart</h2>
    <p class="muted" style="margin-top:-6px;">Court coordinates in feet. Shots are placed from the ball's court position at the detected event frame; this is the gather, so timing and location are approximate. Only shots with a valid court location are plotted; boundary projections are called out in the timeline. Dashed arc is the three-point line.</p>
    $shot_chart
  </section>

  <section class="panel section">
    <h2>Event timeline</h2>
    <p class="muted" style="margin-top:-6px;">Click <b>Jump</b> to seek the video. Shot timing is approximate: the detector fires near the gather, not necessarily the release.</p>
    $events
  </section>

  $clips_section

  <div class="two section">
    <section class="panel">
      <h2>Team assignment</h2>
      $team_table
    </section>
    <section class="panel">
      <h2>Top tracks by distance</h2>
      $player_table
    </section>
  </div>

  <section class="panel section">
    <h2>What this is / what it is not</h2>
    <div class="callout">
      <p><b>This is a broadcast demo.</b> It is meant to show the pipeline end-to-end and to make its failure modes visible.</p>
      <p class="muted" style="margin-bottom:0;">It is <b>not</b> a handheld/tripod product build, a live system, or a validated scouting report. Shots can be early or missed; team ID and possession can drift. The demo uses local research-only broadcast footage and is not redistributed.</p>
    </div>
    <p class="downloads" style="margin-top:14px;">
      <a href="clip.json">ClipResult JSON</a>
      $stats_link
      $clips_link
      <a href="report.md">One-page report</a>
      <a href="shot_chart.svg">Shot chart SVG</a>
    </p>
  </section>

  <footer>
    Reproduce: <code>python scripts/build_demo_page.py --run-json "$run_json_name" --video "$video_name" --out-dir "$out_name"</code>
  </footer>
</div>
<script>
  const video = document.getElementById('demo-video');
  const fullscreenButton = document.getElementById('fs-button');
  let fullscreenFallbackTimer = null;

  function setTheater(on) {
    if (fullscreenFallbackTimer) {
      clearTimeout(fullscreenFallbackTimer);
      fullscreenFallbackTimer = null;
    }
    document.body.classList.toggle('theater', on);
    if (fullscreenButton) fullscreenButton.textContent = on ? 'Exit full screen' : 'Full screen';
  }

  function toggleFullscreen() {
    if (!video) return;
    // If the page is already in theater mode, leave it.
    if (document.body.classList.contains('theater')) {
      setTheater(false);
      return;
    }
    // Native fullscreen when the host allows it.
    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }
    let inIframe = false;
    try {
      inIframe = window.self !== window.top;
    } catch (error) {
      inIframe = true;
    }
    if (!inIframe && document.fullscreenEnabled && video.requestFullscreen) {
      try {
        const request = video.requestFullscreen();
        if (request && request.catch) {
          request.catch(() => setTheater(true));
        }
        // Some hosts silently swallow the request. If native fullscreen has not
        // appeared shortly after the click, fall back to the CSS theater mode.
        fullscreenFallbackTimer = setTimeout(() => {
          if (!document.fullscreenElement) setTheater(true);
        }, 120);
      } catch (error) {
        setTheater(true);
      }
      return;
    }
    // The Review pane can embed the page in an iframe and swallow the Fullscreen
    // API request. Use a CSS theater mode there instead.
    setTheater(true);
  }

  if (fullscreenButton) fullscreenButton.addEventListener('click', toggleFullscreen);
  if (video) video.addEventListener('dblclick', toggleFullscreen);
  document.addEventListener('fullscreenchange', () => {
    if (fullscreenFallbackTimer) {
      clearTimeout(fullscreenFallbackTimer);
      fullscreenFallbackTimer = null;
    }
    if (!fullscreenButton) return;
    fullscreenButton.textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
  });
  document.addEventListener('keydown', (event) => {
    const tag = document.activeElement ? document.activeElement.tagName : '';
    if (event.key === 'Escape' && document.body.classList.contains('theater')) {
      setTheater(false);
      return;
    }
    if (event.key.toLowerCase() === 'f' && !['INPUT', 'TEXTAREA'].includes(tag)) {
      toggleFullscreen();
    }
  });
  document.querySelectorAll('[data-seek]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!video) return;
      video.currentTime = parseFloat(button.dataset.seek);
      video.play();
      video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });
</script>
</body>
</html>
""")


def _clips_section(out_dir: Path) -> tuple[str, str]:
    manifest_path = out_dir / "clips" / "clips.json"
    if not manifest_path.is_file():
        return "", ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "", ""
    cards = []
    for clip in manifest.get("clips") or []:
        file_name = clip.get("file")
        if not file_name or not (out_dir / "clips" / file_name).is_file():
            continue
        kind = html.escape(str(clip.get("kind", "event")).title())
        detail = html.escape(str(clip.get("detail") or ""))
        time_s = _fmt_time(_f(clip.get("time_s")))
        cards.append(
            f'<div class="clip-card">'
            f'<video controls preload="metadata" src="clips/{html.escape(file_name)}"></video>'
            f'<div class="clip-meta"><b>{kind}</b> · {time_s}'
            + (f' · {detail}' if detail else '')
            + '</div></div>'
        )
    if not cards:
        return "", ""
    section = (
        '<section class="panel section">'
        '<h2>Event clips</h2>'
        '<p class="muted" style="margin-top:-6px;">Short clips around each detected event. '
        'Use these to judge whether the event is real before trusting the count.</p>'
        '<div class="clip-grid">' + "\n".join(cards) + '</div></section>'
    )
    link = '<a href="clips/clips.json">Event clips</a>'
    return section, link


def build_html(summary: dict, shot_svg: str, video_rel: str | None, poster_rel: str | None,
               stats_name: str | None, out_dir: Path) -> str:
    chips = "".join([
        _chip("source", Path(summary["source"]).name if summary["source"] else "unknown"),
        _chip("duration", f'{summary["duration_s"]:.1f} s'),
        _chip("fps", f'{summary["fps"]}'),
        _chip("resolution", f'{summary["width"]}×{summary["height"]}'),
        _chip("schema", str(summary["schema_version"])),
    ])
    if video_rel:
        video_block = (
            f'<video id="demo-video" controls playsinline preload="metadata" '
            f'poster="{html.escape(poster_rel or "")}">'
            f'<source src="{html.escape(video_rel)}" type="video/mp4">'
            'Your browser cannot play this video. Open the MP4 directly.</video>'
        )
        video_toolbar = (
            '<div class="video-toolbar">'
            '<span class="muted" style="font-size:12px;">Broadcast render</span>'
            '<span style="display:flex; gap:6px; align-items:center;">'
            f'<a class="seek" href="{html.escape(video_rel)}" target="_blank" rel="noopener">Open video</a>'
            '<button id="fs-button" class="seek" type="button" title="Full screen (F)">Full screen</button>'
            '</span></div>'
        )
    else:
        video_block = '<p class="muted">No rendered video was provided. The JSON and shot chart are still shown.</p>'
        video_toolbar = ""
    stats_link = f'<a href="{html.escape(stats_name)}">Stats CSV</a>' if stats_name else ""
    clips_section, clips_link = _clips_section(out_dir)
    return HTML_TEMPLATE.substitute(
        title=html.escape(summary["title"]),
        chips=chips,
        video_block=video_block,
        video_toolbar=video_toolbar,
        cards=_summary_cards(summary),
        shot_chart=shot_svg,
        events=_event_table(summary["events"]),
        team_table=_team_rows(summary),
        player_table=_player_rows(summary),
        stats_link=stats_link,
        clips_section=clips_section,
        clips_link=clips_link,
        run_json_name=html.escape(str(out_dir / "clip.json")),
        video_name=html.escape(str(out_dir / "assets" / "demo.mp4")),
        out_name=html.escape(str(out_dir)),
    )


def _transcode(src: Path, dst: Path, max_width: int = 1600) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        shutil.copy2(src, dst)
        return False
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"scale='min({max_width},iw)':-2"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _poster(video: Path, dst: Path, frame_idx: int | None = None) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = total // 10 if frame_idx is None else frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, target)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return bool(cv2.imwrite(str(dst), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86]))


def build_demo(run_json: Path, video: Path | None, out_dir: Path, title: str,
               max_width: int = 1600, stats: Path | None = None) -> dict:
    data = _load(run_json)
    summary = summarize(data, title=title)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)

    video_rel = None
    poster_rel = None
    if video is not None and video.is_file():
        dst = assets / "demo.mp4"
        if not dst.exists() or dst.stat().st_mtime < video.stat().st_mtime:
            ok = _transcode(video, dst, max_width=max_width)
            if not ok:
                print(f"[demo] note: ffmpeg transcode unavailable; copied {video.name}", file=sys.stderr)
        if dst.exists():
            video_rel = "assets/demo.mp4"
            poster = assets / "poster.jpg"
            # Prefer a frame the court-view gate accepted over the first frame,
            # which on a broadcast is often a crowd or close-up shot.
            poster_frame = None
            for frame in data.get("frames") or []:
                idx = int(frame.get("frame_idx", 0) or 0)
                if frame.get("court_view") and idx >= max(1, len(data.get("frames") or []) // 20):
                    poster_frame = idx
                    break
            if _poster(dst, poster, poster_frame):
                poster_rel = "assets/poster.jpg"
    else:
        print("[demo] note: no video supplied; the page will show the data only", file=sys.stderr)

    shot_svg = shot_chart_svg(summary["shots"])
    (out_dir / "shot_chart.svg").write_text(shot_svg, encoding="utf-8")
    (out_dir / "clip.json").write_text(run_json.read_text(encoding="utf-8"), encoding="utf-8")
    stats_name = None
    if stats is not None and stats.is_file():
        target = out_dir / "stats.csv"
        if target.resolve() != stats.resolve():
            shutil.copy2(stats, target)
        stats_name = "stats.csv"
    (out_dir / "report.md").write_text(build_report(summary, run_json, video, out_dir), encoding="utf-8")
    html_text = build_html(summary, shot_svg, video_rel, poster_rel, stats_name, out_dir)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-json", type=Path, required=True, help="ClipResult JSON")
    parser.add_argument("--video", type=Path, default=None, help="Rendered overlay MP4")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--title", default="Broadcast Analysis Demo")
    parser.add_argument("--max-width", type=int, default=1600, help="Transcode width cap")
    parser.add_argument("--stats", type=Path, default=None, help="Optional stats CSV")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (args.run_json.parent / "demo")
    summary = build_demo(args.run_json, args.video, out_dir, args.title,
                         max_width=args.max_width, stats=args.stats)
    print(f"[demo] wrote {out_dir / 'index.html'}")
    print(f"[demo] events: {summary['event_counts']}")
    print(f"[demo] shots: {len(summary['shots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
