# AI Hand Tracking

Easy Hand Tracking with OpenCV and Mediapipe via macOS Metal GPU

[//]: # ([![codecov]&#40;https://codecov.io/gh/faunris/ai-hand-tracking/graph/badge.svg?token=YIAPH2CQB5&#41;]&#40;https://codecov.io/gh/faunris/ai-hand-tracking&#41;)

[//]: # ([![CI]&#40;https://github.com/faunris/ai-hand-tracking/actions/workflows/main.yml/badge.svg&#41;]&#40;https://github.com/faunris/ai-hand-tracking/actions/workflows/main.yml&#41;)

HandHolder is a Python-based AI hand tracking application that uses OpenCV and Mediapipe to detect and visualize hand landmarks in real-time.

## Features

- Real-time hand tracking using Mediapipe.
- Visualizes hand landmarks and handedness (left or right hand) on the video feed.
- GPU acceleration support for faster processing.

## Installation

1. Install UV:
   ```bash
   brew install uv
   ```

2. Run the application:
   ```bash
   uv run src/main.py
   ```
   For exit, press `ESC`.
