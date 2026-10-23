"""Phase 15 eval harness driver.

Usage
─────
    python eval/score.py                                  # score all labeled clips
    python eval/score.py --clip broadcast_74              # score one clip by stem
    python eval/score.py --out-dir eval/results           # write reports to dir

For each label file under `eval/labels/*.json`:
  1. Find corresponding ClipResult JSON (e.g., eval/results/broadcast_74.clip.json)
  2. Load both label and ClipResult
  3. Score each event type (shots, passes, screens, PnRs, blocks, rebounds, dunks)
  4. Output human-readable report and JSON scorecard
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.diagnostics import (  # noqa: E402
    bootstrap_mean_ci, resolvable_f1_delta, shuffle_sanity_check, summarize_by_group,
)
from eval.ground_truth import GroundTruth  # noqa: E402
from eval.metrics import (  # noqa: E402
    EventScore, EvalReport, score_events_frame_tolerance, score_team_assignment,
)
from eval.metrics.tracking import score_player_count, track_fragmentation  # noqa: E402
from inference.schema import ClipResult  # noqa: E402


# Default frame tolerances (30 fps, ±frames for event detection)
TOL = {
    "shot": 5,       # 0.17 s / 5 frames
    "pass": 5,       # 0.17 s
    "screen": 8,     # 0.27 s
    "pnr": 10,       # 0.33 s
    "block": 5,      # 0.17 s
    "rebound": 8,    # 0.27 s
    "dunk": 5,       # 0.17 s
}




def _git_sha() -> str:
    """Get current git SHA (short form)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _find_video(
    stem: str,
    label_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """Locate the source video for a labelled clip.

    The label file records where its video lives (`video_path`), which is the only
    reliable source: ingested footage sits under `data/raw/<source>/<date>/` and
    the extension varies (.ogv, .webm). Searching by stem alone silently skipped
    every ingested clip, so `--refresh` could only ever rebuild the broadcast one.
    """
    root = repo_root or REPO_ROOT
    if label_path is not None:
        try:
            recorded = json.loads(label_path.read_text(encoding="utf-8")).get("video_path")
        except (OSError, ValueError):
            recorded = None
        if isinstance(recorded, str) and recorded:
            for cand in (root / recorded, Path(recorded)):
                if cand.is_file():
                    return cand

    for cand in [
        root / "eval" / "clips" / f"{stem}.mp4",
        root / "data" / "sample" / "videos" / f"{stem}.mp4",
    ]:
        if cand.exists():
            return cand

    # Last resort: any ingested asset whose name starts with the stem.
    raw = root / "data" / "raw"
    if raw.is_dir():
        for pattern in (".ogv", ".webm", ".mp4"):
            hits = sorted(raw.rglob(f"{stem}*{pattern}"))
            if hits:
                return hits[0]
    return None


def _artifact_age_days(path: Path) -> Optional[float]:
    """Age of a cached ClipResult in days, or None when it is unreadable."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (time.time() - mtime) / 86400.0

# Flags the committed labels are scored against. These are NOT generic
# detector defaults, and re-running without them silently changes the numbers.
# replay-v3's single-frame best-F1 point is 0.25; event detection may want a
# lower threshold for trajectory recall, but that must be measured.
#
# --offline-events is load-bearing: online detection runs before the ball track is
# smoothed, so it fires nothing on these clips. Replaying the detectors over the
# smoothed frames is what produces the benchmark's shot events.
EVAL_PIPELINE_ARGS = [
    "--ball-conf", "0.25",
    "--shot-no-court-required",
    "--kmeans-teams",
    "--offline-events",
    "--no-render",
]


def _run_pipeline(
    video: Path,
    clip_json: Path,
    max_frames: int,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Invoke inference/run_video.py to produce a ClipResult JSON."""
    clip_json.parent.mkdir(parents=True, exist_ok=True)
    mp4_out = clip_json.with_suffix(".mp4")
    cmd = [
        sys.executable, str(REPO_ROOT / "inference" / "run_video.py"),
        "--video", str(video),
        "--frames", str(max_frames),
        "--events",
        "--json-out", str(clip_json),
        "--out", str(mp4_out),
        *(extra_args if extra_args is not None else EVAL_PIPELINE_ARGS),
    ]
    print(f"[eval] running pipeline: {video.name} -> {clip_json.name}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def score_clip(
    label_path: str,
    clip_result_path: str,
    tolerance_f: int = 5,
    *,
    shuffle_check: bool = True,
    shuffle_seed: int = 0,
) -> EvalReport:
    """Score a single clip against ground truth labels."""
    
    # Load ground truth
    gt = GroundTruth.from_json(label_path)
    
    # Load pipeline output
    clip = ClipResult.from_json(clip_result_path)
    
    # Extract predictions (use correct field names from schema)
    pred_shots = [s.frame_start for s in clip.shots]  # ShotEventModel.frame_start
    pred_passes = [p.frame_idx for p in clip.passes]  # PassEventModel.frame_idx
    pred_screens = [s.frame for s in clip.screens]  # ScreenEventModel.frame
    pred_pnrs = [p.screen_frame for p in clip.pnrs]  # PnREventModel.screen_frame
    pred_rebounds = []  # Not yet implemented in pipeline
    pred_dunks = [d.frame for d in clip.dunks]  # Dunk event frame
    pred_blocks = []  # Not yet implemented in pipeline
    
    # Extract ground truth
    gt_shots = [s.frame_idx for s in gt.shots]
    gt_passes = [p.frame_start for p in gt.passes]
    gt_screens = [s.frame_idx for s in gt.screens]
    gt_pnrs = [p.frame_screen for p in gt.pnrs]
    gt_rebounds = [r.frame_idx for r in gt.rebounds]
    gt_dunks = [d.frame_idx for d in gt.dunks]
    gt_blocks = [b.frame_shot for b in gt.blocks]
    
    # Score events
    shots_score = score_events_frame_tolerance(pred_shots, gt_shots, tolerance_f=tolerance_f)
    passes_score = score_events_frame_tolerance(pred_passes, gt_passes, tolerance_f=tolerance_f)
    screens_score = score_events_frame_tolerance(pred_screens, gt_screens, tolerance_f=tolerance_f)
    pnrs_score = score_events_frame_tolerance(pred_pnrs, gt_pnrs, tolerance_f=tolerance_f)
    rebounds_score = score_events_frame_tolerance(pred_rebounds, gt_rebounds, tolerance_f=tolerance_f)
    dunks_score = score_events_frame_tolerance(pred_dunks, gt_dunks, tolerance_f=tolerance_f)
    blocks_score = score_events_frame_tolerance(pred_blocks, gt_blocks, tolerance_f=tolerance_f) if gt_blocks else None
    
    # Score team assignment
    pred_team_dict: Dict = {}
    possession_mode_counts: Dict[str, int] = {}
    players_per_frame: Dict[int, int] = {}
    ids_per_frame: List[List[int]] = []
    for frame_idx, frame_result in enumerate(clip.frames):
        players_per_frame[frame_result.frame_idx] = len(frame_result.players)
        ids_per_frame.append([p.track_id for p in frame_result.players])
        for player in frame_result.players:
            if player.team_id is not None and player.team_id in (1, 2):
                pred_team_dict[(frame_idx, player.track_id)] = player.team_id
        mode = getattr(frame_result, "possession_distance_mode", None)
        if mode is not None:
            possession_mode_counts[mode] = possession_mode_counts.get(mode, 0) + 1

    player_count_score = None
    if gt.tracking_checks:
        player_count_score = score_player_count(
            [(c.frame_idx, c.expected_num_players) for c in gt.tracking_checks],
            players_per_frame,
        )
    
    gt_team_dict: Dict = {}
    for ta in gt.team_assignments:
        gt_team_dict[(ta.frame_idx, ta.track_id)] = ta.team_id
    
    team_assignment_score = score_team_assignment(pred_team_dict, gt_team_dict)

    # B-08 — scrambled labels must destroy the score.
    shuffle_checks: Dict[str, dict] = {}
    if shuffle_check:
        for evt, pred, truth in [
            ("shot", pred_shots, gt_shots),
            ("pass", pred_passes, gt_passes),
            ("screen", pred_screens, gt_screens),
            ("pnr", pred_pnrs, gt_pnrs),
            ("block", pred_blocks, gt_blocks),
            ("rebound", pred_rebounds, gt_rebounds),
            ("dunk", pred_dunks, gt_dunks),
        ]:
            if truth:
                shuffle_checks[evt] = shuffle_sanity_check(
                    pred, truth, tolerance_f,
                    event_type=evt, seed=shuffle_seed,
                    frame_limit=max(1, len(clip.frames)),
                ).to_dict()
    
    # Build report
    return EvalReport(
        video_name=Path(label_path).stem,
        game_id=gt.game_id or Path(label_path).stem,
        footage_type=gt.footage_type,
        shots_score=shots_score,
        passes_score=passes_score,
        screens_score=screens_score,
        pnrs_score=pnrs_score,
        blocks_score=blocks_score,
        rebounds_score=rebounds_score,
        dunks_score=dunks_score,
        team_assignment_score=team_assignment_score,
        shuffle_checks=shuffle_checks,
        team_assignment_available=getattr(clip, "team_assignment_available", None),
        possession_mode_counts=possession_mode_counts,
        player_count_score=player_count_score,
        track_quality=track_fragmentation(ids_per_frame),
    )


def _markdown_table(rows: List[List[str]]) -> str:
    """Format rows as Markdown table."""
    if not rows:
        return "(no rows)"
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    def fmt(r): return "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    header = fmt(rows[0])
    sep    = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body   = "\n".join(fmt(r) for r in rows[1:])
    return "\n".join([header, sep, body])


def format_event_table(per_clip: Dict[str, EvalReport]) -> str:
    """Format event detection scores as Markdown table."""
    rows: List[List[str]] = [["clip", "event", "gt", "pred", "tp", "fp", "fn", "P", "R", "F1"]]
    for clip_stem, report in per_clip.items():
        for evt, score in [
            ("shot", report.shots_score),
            ("pass", report.passes_score),
            ("screen", report.screens_score),
            ("pnr", report.pnrs_score),
            ("block", report.blocks_score),
            ("rebound", report.rebounds_score),
            ("dunk", report.dunks_score),
        ]:
            if score is None:
                continue
            gt_count = score.tp + score.fn
            pred_count = score.tp + score.fp
            rows.append([
                clip_stem, evt,
                str(gt_count), str(pred_count),
                str(score.tp), str(score.fp), str(score.fn),
                f"{score.precision:.2f}", f"{score.recall:.2f}", f"{score.f1:.2f}",
            ])
    
    # Aggregate row
    agg: Dict[str, List[int]] = {}
    for report in per_clip.values():
        for evt, score in [
            ("shot", report.shots_score),
            ("pass", report.passes_score),
            ("screen", report.screens_score),
            ("pnr", report.pnrs_score),
            ("block", report.blocks_score),
            ("rebound", report.rebounds_score),
            ("dunk", report.dunks_score),
        ]:
            if score is None:
                continue
            if evt not in agg:
                agg[evt] = [0, 0, 0, 0, 0]  # [gt, pred, tp, fp, fn]
            agg[evt][0] += score.tp + score.fn
            agg[evt][1] += score.tp + score.fp
            agg[evt][2] += score.tp
            agg[evt][3] += score.fp
            agg[evt][4] += score.fn
    
    rows.append(["-"] * len(rows[0]))
    for evt, (gt, pred, tp, fp, fn) in sorted(agg.items()):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / gt if gt > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append([
            "ALL", evt,
            str(gt), str(pred), str(tp), str(fp), str(fn),
            f"{p:.2f}", f"{r:.2f}", f"{f:.2f}",
        ])
    
    return _markdown_table(rows)


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(description="Phase 15 eval harness")
    ap.add_argument("--labels-dir", default="eval/labels",
                    help="Directory containing <clip_stem>.json label files")
    ap.add_argument("--results-dir", default="eval/results",
                    help="Where to write ClipResult JSONs and score reports")
    ap.add_argument("--clip", default=None,
                    help="Only score this clip stem (e.g. 'broadcast_74')")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Reuse existing ClipResult JSONs; fail if missing")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-run the pipeline even when a ClipResult JSON already "
                         "exists. Without this, a cached artefact from an older "
                         "pipeline is silently re-scored.")
    ap.add_argument("--pipeline-arg", action="append", default=None,
                    metavar="ARG",
                    help="Extra argument passed to inference/run_video.py when the "
                         "pipeline runs. Repeatable. Defaults to EVAL_PIPELINE_ARGS, "
                         "the settings the committed scorecard was produced with; "
                         "pass an empty value to use the pipeline defaults instead.")
    ap.add_argument("--max-artifact-age", type=float, default=7.0,
                    help="Warn when a reused ClipResult is older than this many "
                         "days (default 7). Stale artefacts are how a pre-change "
                         "scorecard gets mistaken for a current one.")
    ap.add_argument("--tolerance", type=int, default=5,
                    help="Frame tolerance for event matching (default 5)")
    ap.add_argument("--tolerances", default="5,15,30",
                    help="Comma-separated frame tolerances for the localization "
                         "sweep (default 5,15,30). The first value is the primary "
                         "regression threshold; the others show how close the "
                         "detector is without loosening it.")
    ap.add_argument("--no-shuffle-check", dest="shuffle_check", action="store_false",
                    default=True,
                    help="Skip the B-08 shuffled-label leakage check")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for the shuffled-label check (default 0)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose output")
    args = ap.parse_args()
    args.tolerances = sorted({int(x) for x in args.tolerances.split(",") if x.strip()})

    labels_dir = REPO_ROOT / args.labels_dir
    results_dir = REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.json"))
    if args.clip:
        label_files = [p for p in label_files if p.stem == args.clip]

    # Benchmark construction uses templates marked unlabeled/draft. Do not
    # score them as zeros: that silently turns "not annotated yet" into a
    # failed detector. Explicitly reviewed labels are the only scoreable input.
    scoreable = []
    for p in label_files:
        try:
            status = json.loads(p.read_text(encoding="utf-8")).get("status", "reviewed")
        except (OSError, ValueError):
            status = "reviewed"
        if status in ("unlabeled", "draft"):
            print(f"[eval] SKIP {p.stem}: status={status}", file=sys.stderr)
            continue
        scoreable.append(p)
    label_files = scoreable

    if not label_files:
        print(f"[eval] no scoreable labels under {labels_dir}", file=sys.stderr)
        return 1

    per_clip: Dict[str, EvalReport] = {}
    per_tol: Dict[int, Dict[str, EvalReport]] = {}
    artifact_ages: Dict[str, float] = {}
    extra_tolerances = [t for t in args.tolerances if t != args.tolerance]

    for label_path in label_files:
        stem = label_path.stem
        
        # Find or generate ClipResult JSON
        clip_json = results_dir / f"{stem}.clip.json"

        needs_run = args.refresh or not clip_json.exists()

        if needs_run:
            if args.skip_pipeline and not args.refresh:
                print(f"[eval] ERROR: --skip-pipeline but {clip_json.name} missing",
                      file=sys.stderr)
                return 1

            # Find video
            video = _find_video(stem, label_path)
            if video is None:
                print(f"[eval] WARN: no video found for {stem}; skipping", file=sys.stderr)
                continue

            # Run pipeline
            try:
                with open(label_path) as f:
                    label = json.load(f)
                max_frames = int(label.get("total_frames", 1000))
                _run_pipeline(video, clip_json, max_frames, args.pipeline_arg)
            except Exception as e:
                print(f"[eval] ERROR running pipeline for {stem}: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return 1

        # A reused artefact may predate the current pipeline. Say so loudly:
        # a stale ClipResult silently scored as current is worse than no score.
        age_days = _artifact_age_days(clip_json)
        if age_days is not None:
            artifact_ages[stem] = age_days
            if not needs_run and age_days > args.max_artifact_age:
                print(
                    f"[eval] WARN: {clip_json.name} is {age_days:.1f} days old "
                    f"and was reused (pipeline config may have changed). "
                    f"Re-run with --refresh to regenerate.",
                    file=sys.stderr,
                )

        # Score the clip
        try:
            report = score_clip(
                label_path=str(label_path),
                clip_result_path=str(clip_json),
                tolerance_f=args.tolerance,
                shuffle_check=args.shuffle_check,
                shuffle_seed=args.shuffle_seed,
            )
            per_clip[stem] = report

            # Secondary tolerances answer "how close is the detector?" without
            # loosening the primary regression threshold.
            for tol in extra_tolerances:
                per_tol.setdefault(tol, {})[stem] = score_clip(
                    label_path=str(label_path),
                    clip_result_path=str(clip_json),
                    tolerance_f=tol,
                    shuffle_check=False,
                )
            
            if args.verbose:
                print(f"\n{report}")
        except Exception as e:
            print(f"[eval] ERROR scoring {stem}: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # ────────────────────────────────────────────────────────────────────────────
    # Output results
    
    print("\n" + format_event_table(per_clip))

    if per_tol:
        all_tols = [args.tolerance] + sorted(per_tol)
        header = ["event"] + [f"F1@{t}f" for t in all_tols]
        rows = [header]
        for evt, attr in [
            ("shot", "shots_score"), ("pass", "passes_score"),
            ("screen", "screens_score"), ("pnr", "pnrs_score"),
            ("block", "blocks_score"), ("rebound", "rebounds_score"),
            ("dunk", "dunks_score"),
        ]:
            row = [evt]
            any_value = False
            for tol in all_tols:
                reports = per_clip.values() if tol == args.tolerance else per_tol[tol].values()
                tp = fp = fn = 0
                found = False
                for report in reports:
                    score = getattr(report, attr)
                    if score is None:
                        continue
                    found = True
                    tp += score.tp
                    fp += score.fp
                    fn += score.fn
                if not found:
                    row.append("-")
                    continue
                any_value = True
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                row.append(f"{2 * p * r / (p + r):.2f}" if (p + r) else "0.00")
            if any_value:
                rows.append(row)
        print("\nLocalization sweep (how close, not a looser pass mark):")
        print(_markdown_table(rows))

    event_attrs = [
        ("shot", "shots_score"),
        ("pass", "passes_score"),
        ("screen", "screens_score"),
        ("pnr", "pnrs_score"),
        ("block", "blocks_score"),
        ("rebound", "rebounds_score"),
        ("dunk", "dunks_score"),
    ]

    # B-17 — how much F1 movement is even visible at this event count?
    agg: Dict[str, List[int]] = {}
    for report in per_clip.values():
        for evt, attr in event_attrs:
            score = getattr(report, attr)
            if score is None:
                continue
            counts = agg.setdefault(evt, [0, 0, 0])  # [gt, tp, fp]
            counts[0] += score.tp + score.fn
            counts[1] += score.tp
            counts[2] += score.fp

    noise_lines = []
    for evt, (n_gt, tp, fp) in sorted(agg.items()):
        if n_gt <= 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / n_gt
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        delta = resolvable_f1_delta(n_gt, f1)
        if delta is not None:
            if delta >= 1.0:
                msg = "no F1 difference is measurable at this sample size (95%)"
            else:
                msg = (f"differences under {100 * delta:.1f} F1 points "
                       "are not measurable (95%)")
            noise_lines.append(
                f"  {evt:<7} n={n_gt:<4} F1={f1:.3f}  -> {msg}"
            )
    if noise_lines:
        print("\nNoise floor (B-17):")
        print("\n".join(noise_lines))

    # B-35 — a disabled/failed team warmup must not look like a good score.
    unavailable = [stem for stem, r in per_clip.items()
                   if r.team_assignment_available is False]
    if unavailable:
        print("\n[team-id] WARNING: team assignment unavailable for "
              + ", ".join(unavailable)
              + " — possession/pass team metrics are not comparable.",
              file=sys.stderr)

    tracked = {stem: r for stem, r in per_clip.items() if r.player_count_score}
    if tracked:
        print("\nPerson-count spot checks (observed = ALL tracked persons, "
              "including referees/bench):")
        for stem, r in sorted(tracked.items()):
            s = r.player_count_score
            obs = ", ".join(
                f"f{f}: {o} vs {e}" for f, e, o in s.observations
            ) or "none"
            print(f"  {stem:<20} checks={s.checks} exact={s.exact} "
                  f"acc={s.accuracy:.2f} mean|err|={s.mean_abs_error:.1f} worst={s.worst}")
            print(f"  {'':<20} [frame: observed vs expected players] {obs}")

    quality = {stem: r.track_quality for stem, r in sorted(per_clip.items()) if r.track_quality}
    if quality:
        print("\nTrack-ID churn (long = kept ID >=50% of clip, fragments = <10%):")
        for stem, q in quality.items():
            print(f"  {stem:<20} unique={q['unique_tracks']:>3} "
                  f"long={q['long_tracks']:>3} fragments={q['fragments']:>3}")

    modes = {m for r in per_clip.values() for m in r.possession_mode_counts}
    if modes:
        print("\nPossession distance mode (B-10):")
        for stem, r in sorted(per_clip.items()):
            parts = ", ".join(f"{m}={n}"
                              for m, n in sorted(r.possession_mode_counts.items()))
            print(f"  {stem:<16} {parts or '(none)'}")

    # B-08 — shuffled-label sanity check.
    shuffled_rows = []
    for clip_stem, report in per_clip.items():
        for evt, check in report.shuffle_checks.items():
            shuffled_rows.append(
                f"  {clip_stem:<16} {evt:<7} observed={check['observed_f1']:.2f} "
                f"shuffled={check['shuffled_f1']:.2f}  [{check['status']}]"
                + (f" {check['reason']}" if check.get("reason") else "")
            )
    if shuffled_rows:
        print("\nShuffled-label check (B-08):")
        print("\n".join(shuffled_rows))
        if any(c["status"] == "warning"
               for r in per_clip.values() for c in r.shuffle_checks.values()):
            print("[eval] WARNING: shuffled labels still score above chance — "
                  "inspect the matcher before trusting this run.", file=sys.stderr)

    # B-03 — error bars and per-group scorecards are over CLIPS, not frames.
    if per_clip:
        avg_f1_values = [r.average_event_f1() for r in per_clip.values()]
        mean_f1 = sum(avg_f1_values) / len(avg_f1_values)
        lo, hi = bootstrap_mean_ci(avg_f1_values)
        if lo is None:
            print(f"\nAverage F1 across {len(avg_f1_values)} clip(s): {mean_f1:.3f} "
                  "(no error bar: need >=2 clips)")
        else:
            print(f"\nAverage F1 across {len(avg_f1_values)} clips: {mean_f1:.3f} "
                  f"(95% CI {lo:.3f}–{hi:.3f}, bootstrap over clips)")

        for label, attr in [("game", "game_id"), ("footage type", "footage_type")]:
            groups = summarize_by_group(per_clip, attr)
            if not groups:
                continue
            print(f"\nGrouped by {label} (B-03):")
            for g in groups:
                ci = (f"95% CI {g.ci_low:.3f}–{g.ci_high:.3f}"
                      if g.ci_low is not None else "no CI (n<2 clips)")
                print(f"  {g.group:<24} clips={g.n_clips:<3} mean F1={g.mean_f1:.3f}  {ci}")
    
    # Write JSON report
    sha = _git_sha()
    report_path = results_dir / f"{sha}.json"
    serialized = {
        "git_sha": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerances": TOL,
        "artifact_ages_days": {k: round(v, 2) for k, v in artifact_ages.items()},
        "clips": {}
    }
    
    for clip_stem, report in per_clip.items():
        serialized["clips"][clip_stem] = {
            "game_id": report.game_id,
            "footage_type": report.footage_type,
            "team_assignment_available": report.team_assignment_available,
            "player_count": (
                report.player_count_score.to_dict()
                if report.player_count_score else None
            ),
            "track_quality": report.track_quality,
            "possession_distance_modes": report.possession_mode_counts,
            "shuffle_checks": report.shuffle_checks,
            "shots": {
                "p": report.shots_score.precision if report.shots_score else None,
                "r": report.shots_score.recall if report.shots_score else None,
                "f1": report.shots_score.f1 if report.shots_score else None,
                "tp": report.shots_score.tp if report.shots_score else None,
                "fp": report.shots_score.fp if report.shots_score else None,
                "fn": report.shots_score.fn if report.shots_score else None,
            } if report.shots_score else None,
            "passes": {
                "p": report.passes_score.precision if report.passes_score else None,
                "r": report.passes_score.recall if report.passes_score else None,
                "f1": report.passes_score.f1 if report.passes_score else None,
            } if report.passes_score else None,
            "screens": {
                "p": report.screens_score.precision if report.screens_score else None,
                "r": report.screens_score.recall if report.screens_score else None,
                "f1": report.screens_score.f1 if report.screens_score else None,
            } if report.screens_score else None,
            "pnrs": {
                "p": report.pnrs_score.precision if report.pnrs_score else None,
                "r": report.pnrs_score.recall if report.pnrs_score else None,
                "f1": report.pnrs_score.f1 if report.pnrs_score else None,
            } if report.pnrs_score else None,
            "blocks": {
                "p": report.blocks_score.precision if report.blocks_score else None,
                "r": report.blocks_score.recall if report.blocks_score else None,
                "f1": report.blocks_score.f1 if report.blocks_score else None,
            } if report.blocks_score else None,
            "rebounds": {
                "p": report.rebounds_score.precision if report.rebounds_score else None,
                "r": report.rebounds_score.recall if report.rebounds_score else None,
                "f1": report.rebounds_score.f1 if report.rebounds_score else None,
            } if report.rebounds_score else None,
            "dunks": {
                "p": report.dunks_score.precision if report.dunks_score else None,
                "r": report.dunks_score.recall if report.dunks_score else None,
                "f1": report.dunks_score.f1 if report.dunks_score else None,
            } if report.dunks_score else None,
            "team_assignment": {
                "accuracy": report.team_assignment_score.accuracy if report.team_assignment_score else None,
                "correct": report.team_assignment_score.num_correct if report.team_assignment_score else None,
                "total": report.team_assignment_score.num_total if report.team_assignment_score else None,
            } if report.team_assignment_score else None,
            "avg_event_f1": report.average_event_f1(),
        }

    # B-03 — clip-level groups and error bars.
    serialized["groups"] = {
        "by_game": [
            {
                "game_id": g.group,
                "n_clips": g.n_clips,
                "mean_f1": g.mean_f1,
                "ci_low": g.ci_low,
                "ci_high": g.ci_high,
            }
            for g in summarize_by_group(per_clip, "game_id")
        ],
        "by_footage_type": [
            {
                "footage_type": g.group,
                "n_clips": g.n_clips,
                "mean_f1": g.mean_f1,
                "ci_low": g.ci_low,
                "ci_high": g.ci_high,
            }
            for g in summarize_by_group(per_clip, "footage_type")
        ],
    }
    
    with open(report_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\n[eval] wrote {report_path.relative_to(REPO_ROOT)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
