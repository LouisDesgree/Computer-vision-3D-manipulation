# Computer-vision-3D-manipulation

Bootstrap for hand tracking and future 3D manipulation experiments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the hand tracking demo

```bash
./run_hand_tracking.py --flip
```

Press `q` or `Esc` to quit.
The first run downloads the MediaPipe hand landmarker model into `models/`.

## Troubleshooting

- If the video window is black on macOS, ensure Terminal/Python has camera access in
  System Settings > Privacy & Security > Camera, then try `./run_hand_tracking.py --camera-index 1`
  or `./run_hand_tracking.py --auto-camera`.

## Structure

- `src/cv3d/hand_tracking.py`: hand landmark extraction utilities.
- `run_hand_tracking.py`: camera demo loop.

## Next steps

- Add pose smoothing and a stable hand coordinate frame.
- Export landmarks to a 3D engine (e.g., Blender, Unity, or Three.js).
- Calibrate camera intrinsics for better 3D reconstruction.
