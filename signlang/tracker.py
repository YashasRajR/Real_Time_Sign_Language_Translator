"""Hand tracking with MediaPipe's Hand Landmarker (Tasks API)."""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config

# Pairs of landmark indices that form the hand skeleton, for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model(path=config.HAND_MODEL):
    """Download the hand landmark model (about 8 MB) the first time the app runs."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading the MediaPipe hand model (first run only)...")
    tmp = path + ".part"
    urllib.request.urlretrieve(config.HAND_MODEL_URL, tmp)
    os.replace(tmp, path)
    return path


class HandTracker:
    def __init__(self, max_hands=1, min_confidence=0.6):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr, timestamp_ms):
        """Return (landmarks as 21 (x, y, z) tuples, handedness) or (None, None)."""
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # must strictly increase
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        lms = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        hand = result.handedness[0][0].category_name if result.handedness else "Right"
        return lms, hand

    def close(self):
        self.landmarker.close()


def draw_hand(frame, landmarks, color=(255, 255, 255), joint=(0, 170, 255)):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, joint, -1, cv2.LINE_AA)
