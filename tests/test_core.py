"""Tests for everything except the live camera, using synthetic hand landmarks."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signlang import classifier, gestures, ui  # noqa: E402
from signlang.features import N_FEATURES, extract, normalize  # noqa: E402
from signlang.sentence import Sentence  # noqa: E402
from signlang.stabilizer import Stabilizer  # noqa: E402

MCP_X = [0.44, 0.48, 0.52, 0.56]


def make_hand(thumb="folded", fingers=(False, False, False, False), dx=0.0, dy=0.0, s=1.0):
    """Synthetic right hand in image coordinates (x right, y down)."""
    p = [(0.5, 0.8)]  # wrist
    thumb_pts = {
        "folded": [(0.56, 0.75), (0.58, 0.70), (0.55, 0.66), (0.50, 0.64)],
        "out": [(0.56, 0.75), (0.60, 0.70), (0.65, 0.66), (0.70, 0.63)],
        "up": [(0.56, 0.75), (0.58, 0.68), (0.58, 0.58), (0.58, 0.50)],
        "down": [(0.56, 0.75), (0.58, 0.72), (0.58, 0.82), (0.58, 0.90)],
    }[thumb]
    p += thumb_pts
    for x, ext in zip(MCP_X, fingers):
        if ext:
            p += [(x, 0.62), (x, 0.54), (x, 0.49), (x, 0.44)]
        else:
            p += [(x, 0.62), (x, 0.56), (x, 0.60), (x, 0.62)]
    return [(0.5 + (x - 0.5) * s + dx, 0.5 + (y - 0.5) * s + dy, 0.0) for x, y in p]


OPEN = make_hand("out", (True, True, True, True))
FIST = make_hand("folded")
ILY = make_hand("out", (True, False, False, True))


# Features ---------------------------------------------------------------

def test_features_ignore_position_and_size():
    a = extract(OPEN)
    b = extract(make_hand("out", (True, True, True, True), dx=0.2, dy=-0.1, s=0.6))
    assert a.shape == (N_FEATURES,)
    assert np.allclose(a, b, atol=1e-5)


def test_left_hand_is_mirrored_to_match_right():
    mirrored = [(1 - x, y, z) for x, y, z in ILY]
    assert np.allclose(normalize(ILY, "Right"), normalize(mirrored, "Left"), atol=1e-5)


# Built-in gestures --------------------------------------------------------

@pytest.mark.parametrize("hand,expected", [
    (OPEN, "5"),
    (FIST, "0"),
    (make_hand("folded", (True, False, False, False)), "1"),
    (make_hand("folded", (True, True, False, False)), "2"),
    (ILY, "I_LOVE_YOU"),
    (make_hand("up"), "GOOD"),
    (make_hand("down"), "BAD"),
])
def test_builtin_gestures(hand, expected):
    assert gestures.recognise(hand)[0] == expected


# Stabilizer ---------------------------------------------------------------

def feed(stab, label, n, conf=0.9):
    return [t for t in (stab.update(label, conf) for _ in range(n)) if t]


def test_sign_typed_once_while_held():
    stab = Stabilizer(hold=5, release=3)
    assert feed(stab, "A", 4) == []
    assert feed(stab, "A", 20) == ["A"]


def test_same_sign_again_after_hand_leaves():
    stab = Stabilizer(hold=5, release=3)
    feed(stab, "A", 10)
    feed(stab, None, 3)
    assert feed(stab, "A", 10) == ["A"]


def test_next_sign_without_lowering_hand():
    stab = Stabilizer(hold=5, release=3)
    feed(stab, "A", 10)
    assert feed(stab, "B", 15) == ["B"]


def test_low_confidence_is_ignored():
    stab = Stabilizer(hold=5, release=3, min_conf=0.6)
    assert feed(stab, "A", 20, conf=0.3) == []


def test_brief_flicker_does_not_retype():
    stab = Stabilizer(hold=5, release=3)
    feed(stab, "A", 10)
    feed(stab, "B", 1)
    assert feed(stab, "A", 10) == []


# Sentence -----------------------------------------------------------------

def test_sentence_building():
    s = Sentence()
    for sign in ["H", "I", "SPACE", "I_LOVE_YOU", "Y", "O", "U", "U", "DELETE"]:
        s.add(sign)
    assert s.text == "Hi I love you you"


# Classifier ---------------------------------------------------------------

def noisy(hand, rng, n):
    return [np.asarray(hand) + rng.normal(0, 0.006, (21, 3)) for _ in range(n)]


def test_classifier_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    lms, hands, labels = [], [], []
    for label, hand in [("B", OPEN), ("A", FIST), ("I_LOVE_YOU", ILY)]:
        samples = noisy(hand, rng, 40)
        lms += samples
        hands += ["Right"] * 40
        labels += [label] * 40
    model, report = classifier.train(lms, hands, labels, n_aug=2)
    assert report["accuracy"] > 0.95
    assert classifier.predict(model, ILY, "Right")[0] == "I_LOVE_YOU"

    path = tmp_path / "m.joblib"
    classifier.save(model, str(path))
    assert classifier.predict(classifier.load(str(path)), FIST, "Right")[0] == "A"


def test_samples_csv_round_trip(tmp_path):
    path = str(tmp_path / "s.csv")
    classifier.append_samples("A", "Right", [FIST, FIST], path)
    classifier.append_samples("B", "Left", [OPEN], path)
    lms, hands, labels = classifier.load_samples(path)
    assert labels == ["A", "A", "B"] and hands == ["Right", "Right", "Left"]
    assert np.allclose(lms[2], np.asarray(OPEN), atol=1e-4)


# UI -------------------------------------------------------------------------

def test_ui_renders_full_window():
    frame = np.zeros((480, 640, 3), np.uint8)
    s = Sentence()
    s.add("H")
    for kwargs in [dict(pending="I", progress=0.5, confidence=0.8, alternatives=[("I", .8), ("Y", .15)], landmarks=OPEN),
                   dict(paused=True), dict(show_help=True), dict(show_help=True, mode="Built-in gestures"),
                   dict(history=["H", "I", "SPACE"], flash=0.5, landmarks=OPEN)]:
        out = ui.render(ui.fit_video(frame), sentence=s, **kwargs)
        assert out.shape == (ui.H, ui.W, 3)
    out = ui.message_screen(ui.fit_video(frame), "A", "Hint", footer="Footer", progress=0.3,
                            recording=True, counter=(40, 150), step=(3, 26), landmarks=OPEN)
    assert out.shape == (ui.H, ui.W, 3)


def test_hand_guidance():
    assert ui.hand_guidance(None) is None
    assert ui.hand_guidance(OPEN) is None
    tiny = [(0.5 + (x - 0.5) * 0.2, 0.5 + (y - 0.5) * 0.2, z) for x, y, z in OPEN]
    assert "closer" in ui.hand_guidance(tiny)
    edge = [(x + 0.5, y, z) for x, y, z in OPEN]
    assert "view" in ui.hand_guidance(edge)


def test_top_predictions_sorted():
    rng = np.random.default_rng(0)
    lms = noisy(OPEN, rng, 30) + noisy(FIST, rng, 30)
    model, _ = classifier.train(lms, ["Right"] * 60, ["B"] * 30 + ["A"] * 30, n_aug=1)
    top = classifier.predict_top(model, OPEN, "Right", k=2)
    assert top[0][0] == "B" and top[0][1] >= top[1][1]
