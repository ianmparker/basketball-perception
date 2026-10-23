Training and model guidelines.

- Use src/train.py to run experiments.
- Log runs to Weights & Biases and include DVC dataset hash.

## Ball detector weights

- `ball_detector_ebard_v3.pt` - replay-v3 YOLO11s, 4-class ball/hoop/player/referee,
  trained at 1280. This is the current target-domain production candidate.
- `ball_detector_model.pt` - legacy 7-class YOLOv5l6u, kept as the fallback.

Both `.pt` files are gitignored; provision them locally. They are not distributed with
this repository because the training data includes research-only footage.
