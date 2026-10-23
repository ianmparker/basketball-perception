"""Build a broadcast screen/PnR benchmark from MultiSports.

MultiSports annotates 25 fps action tubes. We map each screen and pick-and-roll
defense tube to an event at its start frame. This is an event-start benchmark,
not a full temporal-boundary benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tarfile
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent.parent
ROOT = Path(os.environ.get("MULTISPORTS_ROOT", REPO / "data" / "multisports"))
GT = ROOT / "data" / "trainval" / "multisports_GT.pkl"
TEST_TAR = ROOT / "data" / "test" / "basketball.tar"
TRAINVAL_TAR = ROOT / "data" / "trainval" / "basketball.tar"
CLIPS = ROOT / "clips"
OUT = REPO / "eval" / "multisports_v1"


def load_gt():
    with open(GT, "rb") as fh:
        return pickle.load(fh)


def screen_events(gt, vid):
    labels = gt["labels"]
    out = []
    for label_idx, tubes in gt["gttubes"].get(vid, {}).items():
        if labels[label_idx] == "basketball screen":
            for tube in tubes:
                frame = max(0, int(tube[0, 0]) - 1)
                out.append({
                    "frame_idx": frame,
                    "screener_id": 0,
                    "cutter_id": 0,
                    "screen_type": "off_ball",
                    "team_id": None,
                })
    return out


def pnr_events(gt, vid):
    labels = gt["labels"]
    out = []
    for label_idx, tubes in gt["gttubes"].get(vid, {}).items():
        if labels[label_idx] == "basketball pick-and-roll defensive":
            for tube in tubes:
                frame = max(0, int(tube[0, 0]) - 1)
                out.append({
                    "frame_screen": frame,
                    "frame_roll": frame,
                    "screener_id": 0,
                    "ball_handler_id": 0,
                    "pnr_type": "drag",
                    "outcome": "unknown",
                    "team_id": None,
                })
    return out


def shots_and_passes(gt, vid):
    labels = gt["labels"]
    shots, passes = [], []
    for label_idx, tubes in gt["gttubes"].get(vid, {}).items():
        name = labels[label_idx]
        for tube in tubes:
            frame = max(0, int(tube[0, 0]) - 1)
            if name == "basketball 3-point shot":
                shots.append({"frame_idx": frame, "track_id": 0,
                              "shot_type": "three_point", "outcome": "unknown"})
            elif name == "basketball 2-point shot":
                shots.append({"frame_idx": frame, "track_id": 0,
                              "shot_type": "two_point", "outcome": "unknown"})
            elif name == "basketball free throw":
                shots.append({"frame_idx": frame, "track_id": 0,
                              "shot_type": "free_throw", "outcome": "unknown"})
            elif name == "basketball pass":
                passes.append({"frame_start": frame, "frame_end": frame,
                               "from_track_id": 0, "to_track_id": 0,
                               "team_id": None, "outcome": "complete"})
    return shots, passes


def blocks_and_rebounds(gt, vid):
    labels = gt["labels"]
    blocks, rebounds = [], []
    for label_idx, tubes in gt["gttubes"].get(vid, {}).items():
        name = labels[label_idx]
        for tube in tubes:
            frame = max(0, int(tube[0, 0]) - 1)
            if name == "basketball block":
                blocks.append({"frame_shot": frame, "frame_block": frame,
                               "shooter_id": 0, "blocker_id": 0,
                               "block_type": "contested"})
            elif name == "basketball offensive rebound":
                rebounds.append({"frame_idx": frame, "rebounder_id": 0,
                                 "rebound_type": "offensive", "shot_frame": None,
                                 "team_id": None})
            elif name == "basketball defensive rebound":
                rebounds.append({"frame_idx": frame, "rebounder_id": 0,
                                 "rebound_type": "defensive", "shot_frame": None,
                                 "team_id": None})
    return blocks, rebounds


def selected_clips(gt, limit):
    rows = []
    for vid in gt["test_videos"][0]:
        if not vid.startswith("basketball/"):
            continue
        screens = screen_events(gt, vid)
        pnrs = pnr_events(gt, vid)
        if not screens and not pnrs:
            continue
        rows.append({
            "clip": vid.split("/", 1)[1],
            "video_id": vid,
            "screens": len(screens),
            "pnrs": len(pnrs),
        })
    rows.sort(key=lambda r: r["clip"])
    return rows[:limit] if limit else rows


def extract_clips(tar_path, selected, out_dir):
    wanted = {r["video_id"] + ".mp4" for r in selected}
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            name = member.name.replace("\\", "/")
            if not member.isfile() or name not in wanted:
                continue
            dest = out_dir / Path(name).name
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            extracted.add(name)
    return extracted


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return frames, width, height, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10,
                    help="Number of test clips to build (0 = all)")
    ap.add_argument("--all", action="store_true", help="Build all test event clips")
    args = ap.parse_args()
    limit = 0 if args.all else args.limit

    gt = load_gt()
    selected = selected_clips(gt, limit)
    print(f"selected {len(selected)} clips")
    missing = [r for r in selected if not (CLIPS / f"{r['clip']}.mp4").exists()]
    if missing:
        print(f"extracting {len(missing)} clips from {TRAINVAL_TAR.name}")
        extract_clips(TRAINVAL_TAR, missing, CLIPS)

    labels_dir = OUT / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_id": "multisports_v1",
        "source": "MultiSports validation basketball (GT test_videos; no test labels)",
        "license": "CC-BY-NC-4.0",
        "fps": 25.0,
        "clips": [],
    }
    for row in selected:
        vid = row["video_id"]
        clip = row["clip"]
        mp4 = CLIPS / f"{clip}.mp4"
        if not mp4.is_file():
            print(f"WARN missing {mp4}")
            continue
        frames, width, height, fps = video_meta(mp4)
        screens = screen_events(gt, vid)
        pnrs = pnr_events(gt, vid)
        shots, passes = shots_and_passes(gt, vid)
        blocks, rebounds = blocks_and_rebounds(gt, vid)
        label = {
            "video_path": str(mp4),
            "fps": fps,
            "total_frames": frames,
            "game_id": clip.split("_c")[0],
            "footage_type": "competition_broadcast",
            "annotator": "MultiSports official action tubes",
            "annotation_date": "2026-10-21",
            "status": "reviewed",
            "notes": (
                "Official MultiSports action tubes (screen, pick-and-roll defense, "
                "pass, shots, blocks, rebounds) mapped to their start frames. "
                "Start-frame benchmark, not full boundary."
            ),
            "shots": shots,
            "passes": passes,
            "screens": screens,
            "pnrs": pnrs,
            "blocks": blocks,
            "rebounds": rebounds,
            "dunks": [],
            "team_assignments": [],
            "tracking_checks": [],
        }
        (labels_dir / f"{clip}.json").write_text(
            json.dumps(label, indent=2) + "\n", encoding="utf-8")
        manifest["clips"].append({
            **row, "mp4": str(mp4), "frames": frames,
            "width": width, "height": height, "fps": fps,
            "screen_events": len(screens), "pnr_events": len(pnrs),
            "shot_events": len(shots), "pass_events": len(passes),
            "block_events": len(blocks), "rebound_events": len(rebounds),
        })
        print(f"{clip}: {len(screens)} screens, {len(pnrs)} pnrs, "
              f"{frames} frames {width}x{height} @ {fps:.1f}fps")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# MultiSports broadcast screen/PnR benchmark v1\n\n"
        "Source: MultiSports validation split (`test_videos` in the GT pickle), basketball, CC-BY-NC-4.0. Test-split labels are withheld.\n\n"
        "Official 25 fps action tubes for screens, pick-and-roll defense, passes, "
        "shots, blocks, and rebounds are mapped to their start frames. "
        "This is a start-frame benchmark, not a full temporal-boundary benchmark.\n\n"
        "Labels are scoreable (`status: reviewed`) because they are official "
        "dataset labels, not model predictions.\n\n"
        f"Clips: {len(manifest['clips'])}. Build script: `eval/multisports_v1/build.py`.\n",
        encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


