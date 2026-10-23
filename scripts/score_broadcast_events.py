"""Score labeled broadcast shot candidates.

The labels file is a template until a human fills ``label`` (1 = real shot, 0 =
false positive), optionally ``release_frame`` and ``shooter_track_id_gt``, and
adds any missed shots with ``predicted_frame: null``. This script then reports
event-level precision/recall/F1 at one or more frame tolerances.

    python scripts/score_broadcast_events.py \
        --labels eval/broadcast_demo/labels.json \
        --out eval/broadcast_demo/event_scorecard.md \
        --json-out eval/broadcast_demo/event_scorecard.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _predictions(run_json: Path) -> list[int]:
    data = _load(run_json)
    return sorted(int(shot.get("frame_start", 0)) for shot in data.get("shots") or [])


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_run(run: dict, predictions: list[int], tolerance: int) -> dict:
    events = run.get("events") or []
    labeled = [event for event in events if event.get("label") in (0, 1)]
    gather_only = [event for event in events if event.get("label") == "gather"]
    positives = [event for event in labeled if int(event.get("label")) == 1]
    negatives = [event for event in labeled if int(event.get("label")) == 0]

    used: set[int] = set()
    tp = 0
    offsets: list[int] = []
    for event in positives:
        target = event.get("release_frame")
        if target is None:
            target = event.get("predicted_frame")
        if target is None:
            continue
        target = int(target)
        best: tuple[int, int] | None = None
        for index, prediction in enumerate(predictions):
            if index in used:
                continue
            distance = abs(prediction - target)
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, index)
        if best is not None:
            used.add(best[1])
            tp += 1
            if event.get("release_frame") is not None:
                offsets.append(predictions[best[1]] - int(event["release_frame"]))

    fp = 0
    for event in negatives:
        target = event.get("predicted_frame")
        if target is None:
            continue
        target = int(target)
        if any(abs(prediction - target) <= tolerance for prediction in predictions):
            fp += 1

    fn = len(positives) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_lo, p_hi = wilson_interval(tp, tp + fp)
    r_lo, r_hi = wilson_interval(tp, tp + fn)
    offsets_sorted = sorted(offsets)
    return {
        "run": run.get("id"),
        "regime": run.get("regime", "unknown"),
        "tolerance": tolerance,
        "labeled": len(labeled),
        "gather_only": len(gather_only),
        "unlabeled": len(events) - len(labeled) - len(gather_only),
        "gt": len(positives),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "precision_ci": [round(p_lo, 3), round(p_hi, 3)],
        "recall_ci": [round(r_lo, 3), round(r_hi, 3)],
        "offset_median": offsets_sorted[len(offsets_sorted) // 2] if offsets_sorted else None,
        "offset_mean": round(sum(offsets_sorted) / len(offsets_sorted), 2) if offsets_sorted else None,
        "offset_range": [min(offsets_sorted), max(offsets_sorted)] if offsets_sorted else None,
    }


def score_labels(labels: dict, tolerances: list[int]) -> dict:
    rows = []
    for run in labels.get("runs") or []:
        run_json = Path(run.get("run_json", ""))
        if not run_json.is_file():
            rows.append({"run": run.get("id"), "error": f"missing run json: {run_json}"})
            continue
        predictions = _predictions(run_json)
        for tolerance in tolerances:
            rows.append(score_run(run, predictions, tolerance))
    return {"tolerances": tolerances, "rows": rows}


def _regime_label(value: str) -> str:
    return {
        "nba_broadcast": "NBA broadcast",
        "hs_top_down": "HS top-down",
        "unknown": "Unknown",
    }.get(value or "unknown", value or "unknown")


def _aggregate_row(label: str, subset: list[dict]) -> str:
    tp = sum(row["tp"] for row in subset)
    fp = sum(row["fp"] for row in subset)
    fn = sum(row["fn"] for row in subset)
    judged = sum(row["labeled"] for row in subset)
    gather = sum(row.get("gather_only", 0) for row in subset)
    gt = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_lo, p_hi = wilson_interval(tp, tp + fp)
    r_lo, r_hi = wilson_interval(tp, tp + fn)
    return (
        f"| **{label}** | — | {judged} | {gather} | {gt} | {tp} | {fp} | {fn} | "
        f"{precision:.2f} ({p_lo:.2f}-{p_hi:.2f}) | "
        f"{recall:.2f} ({r_lo:.2f}-{r_hi:.2f}) | {f1:.2f} | — |"
    )


def _table(rows: list[dict], tolerance: int) -> list[str]:
    lines = [
        f"### Tolerance ±{tolerance} frames",
        "",
        "| Run | Regime | Judged | Gather only | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    valid = []
    for row in rows:
        if row.get("tolerance") != tolerance:
            continue
        if "error" in row:
            lines.append(f"| {row['run']} | — | — | — | — | — | — | — | — | — | — | {row['error']} |")
            continue
        valid.append(row)
        regime = _regime_label(row.get("regime", "unknown"))
        if row["labeled"] == 0:
            lines.append(
                f"| {row['run']} | {regime} | 0 | {row.get('gather_only', 0)} | 0 | 0 | 0 | 0 | "
                "— | — | — | — |"
            )
            continue
        lines.append(
            f"| {row['run']} | {regime} | {row['labeled']} | {row.get('gather_only', 0)} | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.2f} ({row['precision_ci'][0]:.2f}-{row['precision_ci'][1]:.2f}) | "
            f"{row['recall']:.2f} ({row['recall_ci'][0]:.2f}-{row['recall_ci'][1]:.2f}) | "
            f"{row['f1']:.2f} | {row['offset_median'] if row['offset_median'] is not None else '—'} |"
        )
    if valid:
        for regime in sorted({row.get("regime", "unknown") for row in valid}):
            subset = [row for row in valid if row.get("regime", "unknown") == regime]
            if sum(row["labeled"] for row in subset) == 0:
                continue
            lines.append(_aggregate_row(_regime_label(regime), subset))
        lines.append(_aggregate_row("All runs (mixed regimes)", valid))
    lines.append("")
    return lines


def render_scorecard(result: dict, labels: dict) -> str:
    rows = result["rows"]
    events = [
        event
        for run in labels.get("runs") or []
        for event in run.get("events") or []
    ]
    judged = sum(1 for event in events if event.get("label") in (0, 1))
    gather_only = sum(1 for event in events if event.get("label") == "gather")
    unlabeled = len(events) - judged - gather_only
    lines = [
        "# Broadcast Event Scorecard",
        "",
        f"**Labels:** `{labels.get('status', 'unknown')}`<br>",
        f"**Runs:** {len(labels.get('runs') or [])}<br>",
        f"**Judged candidates:** {judged}<br>",
        f"**Gather only:** {gather_only}<br>",
        f"**Unlabeled:** {unlabeled}",
        "",
    ]
    if judged == 0:
        if gather_only:
            lines += [
                "**No judged shots yet.** These candidates are marked gather-only: the",
                "clip shows the start of a shot but not the release. Extend the clips",
                "(or use the full demo video) and relabel them as 1 (real release) or 0",
                "(no shot).",
                "",
            ]
        else:
            lines += [
                "**No labels yet.** Open each event clip and set `label` to 1 (real shot",
                "with a visible release/attempt) or 0 (no shot). Use `gather` when the clip",
                "stops before the release, and add missed shots with `predicted_frame: null`.",
                "",
            ]
        for run in labels.get("runs") or []:
            run_events = run.get("events") or []
            run_judged = sum(1 for event in run_events if event.get("label") in (0, 1))
            run_gather = sum(1 for event in run_events if event.get("label") == "gather")
            lines.append(
                f"- `{run.get('id')}`: {run_judged} judged, {run_gather} gather-only, "
                f"{len(run_events) - run_judged - run_gather} unlabeled"
            )
        lines.append("")
        return "\n".join(lines)

    for tolerance in result["tolerances"]:
        lines += _table(rows, tolerance)
    lines += [
        "## Notes",
        "",
        "- A true positive matches a labeled real shot to a prediction within the tolerance.",
        "- A false positive is a labeled false-positive candidate that still has a prediction.",
        "- A false negative is a labeled real shot with no prediction in tolerance.",
        "- Gather-only candidates are excluded from precision/recall: the clip shows the",
        "  start of a shot but not the release.",
        "- The label set currently contains only predicted candidates. A real shot the",
        "  detector missed is not in `labels.json`, so **recall is optimistic**. Add missed",
        "  shots with `predicted_frame: null` to measure it.",
        "- Do **not** compare the NBA broadcast aggregate with the HS top-down probe.",
        "  They are different camera regimes; the top-down result is a transfer check.",
        "- Offsets use `release_frame` only; candidates labeled without a release frame count",
        "  as matches but do not contribute to the offset distribution.",
        "- With this few events, the confidence intervals are wide; report the counts, not",
        "  just F1.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path("eval/broadcast_demo/labels.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/broadcast_demo/event_scorecard.md"))
    parser.add_argument("--json-out", type=Path, default=Path("eval/broadcast_demo/event_scorecard.json"))
    parser.add_argument("--tolerances", default="5,15,30")
    args = parser.parse_args(argv)

    tolerances = [int(part) for part in args.tolerances.split(",") if part.strip()]
    labels = _load(args.labels)
    result = score_labels(labels, tolerances)
    args.out.write_text(render_scorecard(result, labels), encoding="utf-8")
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[score] wrote {args.out} and {args.json_out}")
    for row in result["rows"]:
        if "error" in row:
            print(f"[score] {row['run']}: {row['error']}")
        elif row["tolerance"] == tolerances[-1] and row.get("labeled", 0) == 0:
            print(f"[score] {row['run']} @ ±{row['tolerance']}f: no judged candidates")
        elif row["tolerance"] == tolerances[-1]:
            print(f"[score] {row['run']} @ ±{row['tolerance']}f: "
                  f"P {row['precision']:.2f} R {row['recall']:.2f} F1 {row['f1']:.2f} "
                  f"(TP {row['tp']} FP {row['fp']} FN {row['fn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
