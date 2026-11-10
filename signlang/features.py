"""Turn 21 MediaPipe hand landmarks into a position- and size-independent feature vector."""
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
N_FEATURES = 63 + 10


def normalize(landmarks, handedness="Right"):
    """Return a (21, 3) array centred on the wrist, scaled to unit size.

    Left hands are mirrored so a sign made with either hand looks the same.
    Rotation is kept on purpose: some letters (G/Q, K/P) differ only by direction.
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    pts -= pts[WRIST]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts


def extract(landmarks, handedness="Right"):
    """Feature vector: normalised coordinates plus distances between fingertips."""
    pts = normalize(landmarks, handedness)
    tips = pts[TIPS, :2]
    dists = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)]
    return np.concatenate([pts.ravel(), np.asarray(dists, dtype=np.float32)])


def augment(landmarks, rng, n=4):
    """Make slightly rotated, scaled and jittered copies of a sample for training."""
    base = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    out = []
    for _ in range(n):
        a = rng.uniform(-0.15, 0.15)  # about +/- 9 degrees
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
        p = base.copy()
        centre = p[WRIST, :2].copy()
        p[:, :2] = (p[:, :2] - centre) @ rot.T * rng.uniform(0.9, 1.1) + centre
        p += rng.normal(0, 0.004, p.shape).astype(np.float32)
        out.append(p)
    return out
