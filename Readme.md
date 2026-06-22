# specter

Real-time invisibility cloak powered by Python, OpenCV, and MediaPipe.

**Specter** captures your background, segments your body, and replaces you with that background when you trigger a hand gesture — a live "ghost mode" effect through your webcam.

---

## Requirements

- Python 3.8+
- Webcam with OS camera permission granted
- macOS / Windows / Linux

---

## Setup (first time only)

```bash
cd /path/to/Invisibility-Computer-Vision-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install opencv-python numpy mediapipe
```

On first run, Specter auto-downloads AI models into `models/` (~8 MB).

---

## Run

```bash
source .venv/bin/activate
python main.py
```

Alternate camera:

```bash
python main.py 1
```

---

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| `R` | Recalibrate background (stand still 3s) |
| `S` | Save screenshot (`specter_0000.png`, …) |
| `Q` / `ESC` | Quit |

### Hand gestures

| Gesture | What it does |
|---------|----------------|
| Stand still (startup) | Captures clean background for 3 seconds |
| Both hands spread apart | Opens the portal frame between index fingers |
| Pinch thumb + index (either hand) | Toggle invisibility on / off |
| Pinch again | Turn invisibility off |

---

## Tech stack

- Python · OpenCV · NumPy · MediaPipe Tasks API
