"""Built-in gestures recognised with simple geometry, so the app works before any training."""
import numpy as np

from .features import normalize

# (tip, pip) joints for index, middle, ring and little finger
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def finger_states(pts):
    """Return [thumb, index, middle, ring, pinky] as True when the finger is extended."""
    states = []
    # Thumb: extended when its tip is far from the index finger's base knuckle.
    palm = np.linalg.norm(pts[9, :2] - pts[0, :2])
    states.append(np.linalg.norm(pts[4, :2] - pts[5, :2]) > 0.55 * palm)
    for tip, pip in FINGERS:
        states.append(np.linalg.norm(pts[tip, :2] - pts[0, :2]) > np.linalg.norm(pts[pip, :2] - pts[0, :2]) * 1.1)
    return states


def recognise(landmarks, handedness="Right"):
    """Return (label, confidence) for a built-in gesture, or (None, 0.0)."""
    pts = normalize(landmarks, handedness)
    thumb, index, middle, ring, pinky = finger_states(pts)
    others = [index, middle, ring, pinky]

    if thumb and index and pinky and not middle and not ring:
        return "I_LOVE_YOU", 1.0
    if thumb and not any(others):
        # Image y grows downwards, so a negative y means the thumb points up.
        dy = pts[4, 1] - pts[2, 1]
        if dy < -0.15:
            return "GOOD", 1.0
        if dy > 0.15:
            return "BAD", 1.0
    count = int(thumb) + sum(others)
    return str(count), 0.9
