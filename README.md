# ML Climate Dentifier

IP Camera **weather** and **time-of-day** classifier using a dual-head Convolutional Neural Network.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)](https://opencv.org)

---

## Architecture

### Dual-Head CNN

The model outputs **two independent probability distributions** — one for time-of-day, one for weather — instead of the incorrect single 6-way softmax found in v1:

```
Input (150x150x3)
  → Conv2D(32) → BN → MaxPool
  → Conv2D(64) → BN → MaxPool
  → Conv2D(128) → BN → MaxPool
  → Conv2D(256) → BN → MaxPool
  → Flatten → Dense(256) → Dropout → Dense(128) → Dropout
  → [Dense(3, softmax): time_head]   ← Day / Evening / Night
  → [Dense(3, softmax): weather_head] ← Clear / Cloudy / Partly Cloudy
```

Training uses **sample-weight masking**: images from time directories (day/evening/night) only contribute to the time loss; images from weather directories (clear/cloudy/partly_cloudy) only contribute to the weather loss. Both heads can be trained simultaneously with partially-labeled data.

Optional backbone: `--backbone mobilenetv2` for transfer learning.

---

## Project Structure

```
ml-climate-dentifier/
├── app.py                  # Entry point: real-time IP camera classifier
├── config.yaml             # All config (camera, model, display, recording)
├── inference/              # Runtime inference package
│   ├── config.py           # YAML + .env config loader
│   ├── predictor.py        # Model loading, predict, temporal smoothing
│   ├── streamer.py         # Camera capture with exponential backoff
│   └── visualizer.py       # OpenCV overlay and recording
├── training/               # Training package
│   ├── model.py            # CNN / MobileNetV2 dual-head architecture
│   ├── dataset.py          # DualDataGenerator with sample-weight masking
│   ├── train.py            # Training loop with callbacks
│   └── evaluate.py         # Confusion matrices, classification reports
├── scripts/
│   ├── build_dataset.py    # Combine time + weather dirs into 9 classes
│   ├── label_captures.py   # Label captured images by EXIF timestamp
│   └── capture.sh          # Periodic frame capture via ffmpeg
├── models/                 # Trained .h5 files (gitignored)
├── dataset/                # Raw training images (gitignored)
│   ├── day/                #   628 images
│   ├── evening/            #   132
│   ├── night (Nightvision)/#   585
│   ├── clear/              #    65
│   ├── cloudy/             #    10
│   └── partly_cloudy/      #    10
├── captures/               # Periodic captures (gitignored)
├── tests/                  # Pytest test suite
├── Dockerfile
├── Makefile
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Setup

```bash
git clone https://github.com/bi0punk/ml-climate-dentifier
cd ml-climate-dentifier
pip install -r requirements.txt
```

Copy and edit the environment file:
```bash
cp .env.example .env
# Set your camera URL and model path
```

---

## Usage

### Real-time Classification
```bash
python app.py                # Default config
python app.py --record       # Save video to output.avi
python app.py --config my_config.yaml
```

### Test Camera Stream
```bash
# Set RTSP_URL in .env first
python test.py
```

### Train a New Model
```bash
python -m training.train                              # CNN backbone
python -m training.train --backbone mobilenetv2        # Transfer learning
python -m training.train --evaluate                    # Train + evaluate
python -m training.train --epochs 50 --batch-size 64   # Custom params
```

### Build Dataset
```bash
python scripts/build_dataset.py --analyze   # Show class distribution
python scripts/build_dataset.py             # Create combined 9-class dataset
python scripts/label_captures.py            # Label captures/ by timestamp
```

### Capture Periodically
```bash
./scripts/capture.sh          # Every 3 minutes
./scripts/capture.sh 600      # Every 10 minutes
```

---

## Configuration

All settings in `config.yaml`. Environment variables in `.env` override camera URL and model path:

| Variable | Description |
|----------|-------------|
| `IP_CAMERA_URL` | Camera stream URL (overrides config.yaml) |
| `MODEL_PATH` | Path to .h5 model (overrides config.yaml) |
| `RTSP_URL` | Full RTSP URL with credentials |

---

## Docker

```bash
make docker-build
make docker-run
```

---

## Tests

```bash
make test
# or
python -m pytest tests/ -v
```

---

## v1 → v2 Migration

| Issue | v1 | v2 |
|-------|----|----|
| **Architecture** | Single 6-way softmax (Day, Evening, Night, Clear, Cloudy, Partly Cloudy compete) | Dual-head: independent time (3) + weather (3) softmax |
| **Training** | No validation split, `batch_size=1`, dead augmentation code | Train/val/test split, `batch_size=32`, active augmentation, callbacks |
| **Inference** | Predicts every frame, no smoothing | Frame skipping + temporal mode filter |
| **Config** | Hardcoded in source | `config.yaml` + `.env` overrides |
| **Security** | Credentials in `test.py` | `.env`-only, `.env.example` provided |
| **Structure** | Monolithic `.py` files | Modular packages (`inference/`, `training/`, `scripts/`, `tests/`) |

---

## License

MIT
