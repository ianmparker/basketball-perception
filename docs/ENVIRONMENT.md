# Environment

**Verified:** 2026-10-21 on a CUDA 12.8 machine with an RTX 50-series GPU.

| Component | Version |
|---|---|
| Python | 3.11 |
| torch | 2.11.0+cu128 |
| CUDA | 12.8, `torch.cuda.is_available() == True` |
| OpenCV | 4.12 |
| Ultralytics | 8.4 |
| pydantic | 2.14 |

## The gotcha

Conda's `pytorch` package can resolve to a **CPU-only** build on Blackwell (RTX 50xx).
The symptom is a `c10_cuda.dll` load error or `torch.cuda.is_available() == False`.
The verified path is Python 3.11 plus the pip CUDA 12.8 wheels:

```bash
conda create -n basketball-perception python=3.11 -y
conda activate basketball-perception
pip install -r requirements.txt
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
python scripts/check_env.py
```

`requirements.txt` installs CPU torch first; the second command replaces it with the
CUDA build. Do not skip it on a GPU machine. For older GPUs, use the matching CUDA
wheel index from <https://pytorch.org/get-started/locally/>.

## Verify

```bash
python scripts/check_env.py
```

It prints the Python, torch, CUDA, GPU, OpenCV, Ultralytics, and pydantic status. Exit
code 1 means a required import is missing. A CPU-only torch is a warning, not a hard
failure, but it should be fixed for training or inference.

## Notes

- `open_clip_torch` is optional; install it only for the CLIP team-ID path.
- `easyocr`, `wandb`, `roboflow`, and `matplotlib` are optional extras:
  `pip install -r requirements-optional.txt`.
- The court keypoint weights are expected at
  `runs/pose/court_kp_v18_resume/weights/best.pt`. `runs/` is gitignored; a fresh clone
  needs the artifact supplied separately.
- `scripts/calibrate_from_points.py` imports `inference.pipeline`, so it needs torch
  even though the DLT solve itself is pure numpy.
