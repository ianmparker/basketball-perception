"""Event-level shot evaluation against PBP labels from a full NBA game.

Positives: 19 scorebug-aligned PBP field-goal attempts after the alignment audit.
Negatives: 7 clean turnover clips with no preceding shot attempt.
Each clip is 10 s at 30 fps; the PBP/turnover frame is the ground-truth event
frame. The detector emits shot events; we match them within a tolerance. Pass
--no-audit to score the pre-audit 20 + 10 set.

This is event-level, not clip-level: a positive clip with two predicted shots and
one match is TP=1, FP=1. A negative clip with any predicted shot is FP.

    python scripts/eval_pbp_shots.py --tolerances 1,2,3

The first run takes a few minutes because it invokes the video pipeline on 30
clips. Later runs reuse the cached per-clip JSON unless --refresh is passed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIVES = REPO_ROOT / "outputs" / "shot-events" / "401810030" / "validation_ocr.json"
DEFAULT_NEGATIVES = REPO_ROOT / "outputs" / "shot-events" / "401810030" / "negatives_clean" / "negatives.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "pbp_shots"
DEFAULT_SCORECARD = REPO_ROOT / "eval" / "pbp_shots" / "scorecard.md"
DEFAULT_AUDIT = REPO_ROOT / "eval" / "pbp_shots" / "alignment_audit.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_pipeline(clip: Path, out_json: Path, timeout: float) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "inference" / "run_video.py"),
        "--video", str(clip),
        "--no-render", "--events", "--offline-events", "--shot-no-court-required",
        "--json-out", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-800:]
    return True, ""


def predictions(result_path: Path) -> list[int]:
    data = load_json(result_path)
    return sorted(int(shot.get("frame_start", 0)) for shot in data.get("shots") or [])


def clip_records(positives: dict, negatives: dict, results_dir: Path,
                 refresh: bool, timeout: float, limit: int = 0,
                 audit: dict | None = None) -> tuple[list[dict], dict]:
    records = []
    pos_root = Path(positives.get("_root", ""))
    neg_root = Path(negatives.get("_root", ""))
    excluded = {"positives": [], "negatives": []}
    for event in positives.get("events") or []:
        verdict = ((audit or {}).get(event["event_id"]) or {}).get("verdict")
        if verdict == "misaligned":
            excluded["positives"].append(event["event_id"])
            continue
        records.append({
            "kind": "positive",
            "event_id": event["event_id"],
            "clip": pos_root / event["video"],
            "result": results_dir / f"{event['event_id']}.json",
            "gt_offset": int(event["frame"]) - int(event["clip_start_frame"]),
            "made": bool(event.get("made")),
            "shot_type": event.get("shot_type"),
            "shot_value": event.get("shot_value"),
            "text": event.get("text", ""),
        })
    for event in negatives.get("items") or []:
        verdict = ((audit or {}).get(event["event_id"]) or {}).get("verdict")
        if verdict in ("misaligned", "adjacent"):
            excluded["negatives"].append(event["event_id"])
            continue
        records.append({
            "kind": "negative",
            "event_id": event["event_id"],
            "clip": neg_root / event["video"],
            "result": results_dir / f"{event['event_id']}.json",
            "gt_offset": None,
            "made": None,
            "shot_type": None,
            "shot_value": None,
            "text": event.get("text", ""),
        })
    if limit:
        records = [r for r in records if r["kind"] == "positive"][:limit] + [
            r for r in records if r["kind"] == "negative"][:limit]
    for i, record in enumerate(records, start=1):
        if not record["clip"].is_file():
            record["error"] = f"missing clip: {record['clip']}"
            record["predicted_frames"] = []
            continue
        if refresh or not record["result"].is_file():
            print(f"[pbp-eval] {i}/{len(records)} {record['event_id']} ...")
            ok, error = run_pipeline(record["clip"], record["result"], timeout)
            if not ok:
                record["error"] = error
                record["predicted_frames"] = []
                continue
        record["predicted_frames"] = predictions(record["result"])
    return records, {"audited": len(audit or {}), **excluded}


def score(records: list[dict], tolerance_frames: int) -> dict:
    tp = fp = fn = 0
    offsets = []
    for record in records:
        if "error" in record:
            continue
        preds = record["predicted_frames"]
        if record["kind"] == "negative":
            fp += len(preds)
            continue
        used = set()
        target = record["gt_offset"]
        best = None
        for i, pred in enumerate(preds):
            distance = abs(pred - target)
            if distance <= tolerance_frames and (best is None or distance < best[0]):
                best = (distance, i)
        if best is None:
            fn += 1
        else:
            used.add(best[1])
            tp += 1
            offsets.append(preds[best[1]] - target)
        fp += len(preds) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_lo, p_hi = wilson_interval(tp, tp + fp)
    r_lo, r_hi = wilson_interval(tp, tp + fn)
    return {
        "tolerance_frames": tolerance_frames,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "precision_ci": [round(p_lo, 3), round(p_hi, 3)],
        "recall_ci": [round(r_lo, 3), round(r_hi, 3)],
        "offset_median": sorted(offsets)[len(offsets) // 2] if offsets else None,
        "offset_mean": round(sum(offsets) / len(offsets), 2) if offsets else None,
        "offset_range": [min(offsets), max(offsets)] if offsets else None,
    }


def render(records: list[dict], summaries: dict[int, dict], fps: float,
           audit_summary: dict | None = None,
           event_id: str | None = None) -> str:
    positives = [r for r in records if r["kind"] == "positive"]
    negatives = [r for r in records if r["kind"] == "negative"]
    event_id = str(event_id or "game")
    lines = [
        "# PBP Shot Event Evaluation",
        "",
        f"PBP event set: `{event_id}`.",
        f"Positives: {len(positives)} PBP field-goal attempts; negatives: {len(negatives)} non-shot clips.",
        f"Clip fps: {fps}. Match tolerance is reported in seconds.",
    ]
    if audit_summary and audit_summary.get("audited"):
        excluded_pos = ", ".join(audit_summary.get("positives") or []) or "none"
        excluded_neg = ", ".join(audit_summary.get("negatives") or []) or "none"
        lines.append(
            f"Audit: {audit_summary['audited']} clips reviewed; excluded positives: {excluded_pos}; "
            f"excluded negatives: {excluded_neg}."
        )
    lines += [
        "",
        "**Candidate-level timing caveat:** human labels in `candidate_audit.md` show real-shot",
        "candidates a median 3.2 s before the PBP event. This symmetric event scorecard is a",
        "misaligned view, not the final release metric.",
    ]
    lines += [
        "",
        "## Event-level scorecard",
        "",
        "| Tolerance | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median | Offset mean |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for tolerance, summary in summaries.items():
        seconds = tolerance / fps
        lines.append(
            f"| ±{seconds:g}s ({tolerance}f) | {summary['tp']} | {summary['fp']} | {summary['fn']} | "
            f"{summary['precision']:.2f} ({summary['precision_ci'][0]:.2f}-{summary['precision_ci'][1]:.2f}) | "
            f"{summary['recall']:.2f} ({summary['recall_ci'][0]:.2f}-{summary['recall_ci'][1]:.2f}) | "
            f"{summary['f1']:.2f} | {summary['offset_median'] if summary['offset_median'] is not None else '—'} | "
            f"{summary['offset_mean'] if summary['offset_mean'] is not None else '—'} |"
        )
    lines += ["", "## Clip-level check", ""]
    pos_detected = sum(1 for r in positives if r.get("predicted_frames"))
    neg_detected = sum(1 for r in negatives if r.get("predicted_frames"))
    lines += [
        f"- Positive clips with any shot event: {pos_detected}/{len(positives)}",
        f"- Negative clips with any shot event: {neg_detected}/{len(negatives)}",
        "",
    ]
    lines += ["## Per-event", "", "| Event | Kind | Truth | GT frame | Pred frames | Error |", "|---|---|---|---:|---|---|"]
    for record in records:
        truth = "MADE" if record.get("made") else ("MISS" if record["kind"] == "positive" else "NO SHOT")
        gt = record.get("gt_offset")
        preds = record.get("predicted_frames") or []
        lines.append(
            f"| {record['event_id']} | {record['kind']} | {truth} | "
            f"{gt if gt is not None else '—'} | {preds} | {record.get('error', '')} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The PBP frame is the event clock, not the release frame; scorebug alignment corrects the video offset.",
        "- Event-level metrics count extra predicted shots as false positives.",
        f"- The {len(negatives)} negatives are non-shot clips with no PBP field goal in the window.",
        "- This is one full game; the next step is to expand to all 186 PBP events and a second game.",
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
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT,
                        help="alignment-audit JSON; excluded clips are removed from the benchmark")
    parser.add_argument("--no-audit", action="store_true", help="ignore the alignment audit")
    parser.add_argument("--tolerances", default="1,2,3", help="seconds, comma-separated")
    parser.add_argument("--refresh", action="store_true", help="re-run the pipeline")
    parser.add_argument("--limit", type=int, default=0, help="limit positives and negatives for a smoke run")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    positives = load_json(args.positives)
    negatives = load_json(args.negatives)
    positives["_root"] = str(args.positives.resolve().parent)
    negatives["_root"] = str(args.negatives.resolve().parent)
    fps = float(positives.get("fps") or 30.0)
    results_dir = args.out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit = {}
    if not args.no_audit and args.audit and args.audit.is_file():
        audit = load_json(args.audit).get("verdicts", {})

    records, audit_summary = clip_records(
        positives, negatives, results_dir,
        refresh=args.refresh, timeout=args.timeout, limit=args.limit, audit=audit,
    )
    tolerances = [int(round(float(part) * fps)) for part in args.tolerances.split(",") if part.strip()]
    summaries = {tolerance: score(records, tolerance) for tolerance in tolerances}

    json_records = []
    for record in records:
        item = dict(record)
        item["clip"] = str(item.get("clip"))
        item["result"] = str(item.get("result"))
        json_records.append(item)
    payload = {
        "fps": fps,
        "tolerances": tolerances,
        "summaries": summaries,
        "audit": audit_summary,
        "records": json_records,
    }
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.scorecard.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard.write_text(
        render(records, summaries, fps, audit_summary,
               event_id=str(positives.get("event_id") or "game")),
        encoding="utf-8",
    )
    summary_payload = {
        "fps": fps,
        "tolerances": tolerances,
        "summaries": summaries,
        "audit": audit_summary,
        "positives": len([r for r in records if r["kind"] == "positive"]),
        "negatives": len([r for r in records if r["kind"] == "negative"]),
    }
    args.scorecard.with_suffix(".json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[pbp-eval] wrote {args.scorecard}")
    for tolerance, summary in summaries.items():
        print(f"[pbp-eval] ±{tolerance / fps:g}s: P {summary['precision']:.2f} "
              f"R {summary['recall']:.2f} F1 {summary['f1']:.2f} "
              f"(TP {summary['tp']} FP {summary['fp']} FN {summary['fn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
