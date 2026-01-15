# Computer-vision-3D-manipulation

Bootstrap for hand tracking and future 3D manipulation experiments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
#bonne commande : python run_hand_cube.py --flip --camera-index 1
## Run the hand tracking demo

```bash
./run_hand_tracking.py --flip
```

Press `q` or `Esc` to quit.
The first run downloads the MediaPipe hand landmarker model into `models/`.

## Control a 3D cube with your hand

```bash
./run_hand_cube.py --flip
```

The cube is drawn on top of the camera feed. Pinch (thumb + index) to grab the cube
and drag to rotate. Close the window or press `q`/`Esc` to exit. Use
`--cube-anchor center` to keep the cube fixed at screen center instead of following
your palm.
If the video is black, try `--auto-camera` or a different `--camera-index`.
Use `--cube-size 0.5` or `--cube-distance 6.5` if the cube feels too large, and
`--rotation-smoothing 0.08` / `--anchor-smoothing 0.08` for smoother motion.

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
