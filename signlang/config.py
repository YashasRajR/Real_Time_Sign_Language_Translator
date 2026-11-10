"""Paths, sign sets and tuning constants."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
TRANSCRIPTS_DIR = os.path.join(ROOT, "transcripts")
SAMPLES_CSV = os.path.join(DATA_DIR, "samples.csv")
SIGN_MODEL = os.path.join(MODELS_DIR, "sign_model.joblib")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# J and Z are drawn in the air in ASL, so a single hand pose can't capture them.
ALPHABET = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]
CONTROLS = ["SPACE", "DELETE"]
WORDS = ["I_LOVE_YOU", "GOOD"]
SIGN_SETS = {
    "alphabet": ALPHABET + CONTROLS,
    "words": WORDS + CONTROLS,
    "all": ALPHABET + WORDS + CONTROLS,
}

# How each sign is written into the sentence.
DISPLAY = {"I_LOVE_YOU": "I love you", "GOOD": "good", "BAD": "bad"}

# Short reminders shown while recording each sign.
HINTS = {
    "A": "Fist, thumb resting against the side of your index finger",
    "B": "Flat hand, fingers together and up, thumb folded across the palm",
    "C": "Curve your fingers and thumb into a C",
    "D": "Index finger up, other fingertips touching the thumb",
    "E": "Fingers bent down, tips resting on the tucked thumb",
    "F": "Thumb and index finger touch, other three fingers up",
    "G": "Index finger and thumb point sideways, parallel",
    "H": "Index and middle fingers point sideways together",
    "I": "Little finger up, others in a fist",
    "K": "Index and middle up in a V, thumb touching the middle finger",
    "L": "Index finger up, thumb out, making an L",
    "M": "Thumb tucked under the first three fingers",
    "N": "Thumb tucked under the first two fingers",
    "O": "All fingertips touch the thumb, making an O",
    "P": "Like K, but pointing down",
    "Q": "Like G, but pointing down",
    "R": "Index and middle fingers crossed",
    "S": "Fist with the thumb across the front of the fingers",
    "T": "Thumb tucked between index and middle fingers",
    "U": "Index and middle fingers up, together",
    "V": "Index and middle fingers up, spread apart",
    "W": "Index, middle and ring fingers up, spread",
    "X": "Index finger bent like a hook",
    "Y": "Thumb and little finger out",
    "SPACE": "Open palm, all five fingers spread",
    "DELETE": "Thumb pointing down, other fingers in a fist",
    "I_LOVE_YOU": "Thumb, index and little finger out",
    "GOOD": "Thumbs up",
}

# Recognition tuning
MIN_CONFIDENCE = 0.55   # ignore predictions below this probability
HOLD_FRAMES = 12        # frames a sign must be held before it is typed (~0.5 s)
RELEASE_FRAMES = 6      # frames without a hand that count as "letting go"
