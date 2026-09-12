# GMED-YOLO + ByteTrack + STTAC

This repository releases the computational workflow supporting the manuscript:

```text
video frames → GMED-YOLO detections → ByteTrack trajectory segments → STTAC association → defect counts
```

## Released files

- `train_gmed_yolo.py`: detector training entry point.
- `seeds.txt`: random seeds used for all detector experiments.
- `models/gmed-yolo.yaml`: executable GMED-YOLO architecture.
- `modules/gmed.py`: GMED custom modules, including MSEIG, SobelConv, ConvEdgeFusion, Skip-CEF, SPDConv, and
  CSPOmniKernel.
- `modules/gmed_registry.py`: GMED module registration and model-parser integration.
- `postprocess.py`: GMED-YOLO inference, ByteTrack tracking, and trajectory CSV generation.
- `sttac_count.py`: STTAC trajectory association and category-specific defect counting.
- `configs/train.yaml`: detector-training configuration consumed by `train_gmed_yolo.py`.
- `configs/apple-defect.yaml`: detection-dataset format and class configuration.
- `configs/bytetrack.yaml`: ByteTrack parameter configuration.

## Dataset splits

The directory membership under `datasets/` defines the exact released data splits.

For defect detection, the released image-label pairs are organized as follows:

- `datasets/images/train` and `datasets/labels/train`: 6,237 pairs.
- `datasets/images/val` and `datasets/labels/val`: 891 pairs.
- `datasets/images/test` and `datasets/labels/test`: 1,782 pairs.

For defect counting, the 150 videos are organized as follows:

- `datasets/counting/calibration`: 60 calibration videos.
- `datasets/counting/locked_test`: 90 locked-test videos.

## Environment

The experiments used Ultralytics 8.4.41 with Python 3.11.14, PyTorch 2.9.1, and CUDA 12.8. Install the required
packages in an isolated Python environment. ByteTrack also requires `lap`.

```bash
python -m pip install "ultralytics==8.4.41"
python -m pip install "lap>=0.5.12"
```

## Detector training

The released training settings are an input size of 640 pixels, 200 epochs, batch size 16, SGD, an initial learning
rate of 0.01, momentum of 0.937, and weight decay of 0.0005. All detector models are trained from scratch without
pretrained weights. The five random seeds are listed in `seeds.txt`.

Before training, set `path` in `configs/apple-defect.yaml` to the absolute path of the repository's `datasets`
directory.

```bash
python train_gmed_yolo.py --config configs/train.yaml
```

## ByteTrack trajectory generation

Trajectory generation uses the ByteTrack implementation bundled with Ultralytics 8.4.41. The exact tracker
parameters are provided in `configs/bytetrack.yaml`.

```bash
python postprocess.py \
  --weights /path/to/best.pt \
  --source /path/to/input.avi \
  --output-dir runs/track/sample
```

## STTAC counting

```bash
python sttac_count.py \
  --input runs/track/sample/input_trajectories.csv \
  --output runs/track/sample/input_sttac_counts.csv
```

The default STTAC parameters are `L_min=5`, `alpha=0.1`, `N_roller=120 rpm`, `R_roller=25 mm`,
`R_apple=40.25 mm`, `s=0.144`, `F=90.1 FPS`, `delta=15 frames`, and `Delta_y_th=20 pixels`. They give
`N_half=42` and the inclusive temporal window `[27, 57]` frames. Counting uses the three defect classes `puncture`,
`bruise`, and `bitter pit`.

## License

This code is released under AGPL-3.0; see `LICENSE`.
