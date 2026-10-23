#!/bin/bash
python models/src/train.py --export-onnx --ckpt models/experiments/latest.pt --out inference/model.onnx
