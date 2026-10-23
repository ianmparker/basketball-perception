"""Export tracking rows in MOTChallenge format for external evaluation.

Writes the standard layout TrackEval expects:

    <out>/<sequence>/gt/gt.txt
    <out>/<sequence>/seqinfo.ini
    <out>/<sequence>/pred/pred.txt   (optional; same columns)

Columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

Usage:
    python scripts/export_mot.py --clip-json eval/results/wikimedia_s008.clip.json \
        --out data/mot --sequence wikimedia_s008
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.schema import ClipResult  # noqa: E402


def write_pred(path: Path, clip: ClipResult) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", encoding="utf-8") as f:
        for frame in clip.frames:
            for player in frame.players:
                x1, y1, x2, y2 = player.box_xyxy
                f.write(
                    f"{frame.frame_idx + 1},{player.track_id},"
                    f"{x1},{y1},{x2 - x1},{y2 - y1},"
                    f"{player.conf:.4f},1,1\n"
                )
                rows += 1
    return rows


def write_seqinfo(path: Path, clip: ClipResult) -> None:
    path.write_text(
        "[Sequence]\n"
        f"name={path.parent.name}\n"
        "imDir=img1\n"
        f"frameRate={int(round(clip.metadata.fps or 30))}\n"
        f"seqLength={len(clip.frames)}\n"
        f"imWidth={clip.metadata.width}\n"
        f"imHeight={clip.metadata.height}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export MOTChallenge-format tracking")
    ap.add_argument("--clip-json", required=True)
    ap.add_argument("--out", default="data/mot")
    ap.add_argument("--sequence", default=None)
    args = ap.parse_args()

    clip = ClipResult.from_json(args.clip_json)
    stem = args.sequence or Path(args.clip_json).stem.replace(".clip", "")
    seq_dir = Path(args.out) / stem

    rows = write_pred(seq_dir / "pred" / "pred.txt", clip)
    write_seqinfo(seq_dir / "seqinfo.ini", clip)
    print(f"[mot] {seq_dir}: {rows} rows, {len(clip.frames)} frames")
    print("[mot] add gt/gt.txt from labels, then point TrackEval at this directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
