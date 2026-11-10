"""Train, save and use the sign classifier (scikit-learn)."""
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import augment, extract


def append_samples(label, handedness, samples, path=config.SAMPLES_CSV):
    """Append raw landmark samples (each 21x3) for one sign to the CSV dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "hand"] + [f"{a}{i}" for i in range(21) for a in "xyz"])
        for s in samples:
            w.writerow([label, handedness] + [f"{v:.5f}" for v in np.asarray(s).ravel()])


def load_samples(path=config.SAMPLES_CSV):
    """Return (landmarks list, handedness list, labels list)."""
    lms, hands, labels = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"])
            hands.append(row["hand"])
            lms.append(np.array([float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"]).reshape(21, 3))
    return lms, hands, labels


def build_xy(lms, hands, labels, rng=None, n_aug=0):
    X, y = [], []
    for lm, hand, label in zip(lms, hands, labels):
        X.append(extract(lm, hand))
        y.append(label)
        if rng is not None and n_aug:
            for a in augment(lm, rng, n_aug):
                X.append(extract(a, hand))
                y.append(label)
    return np.array(X), np.array(y)


def train(lms, hands, labels, seed=42, n_aug=4):
    """Train with a held-out test split. Returns (fitted model, report dict)."""
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    Xtr, ytr = build_xy([lms[i] for i in tr], [hands[i] for i in tr], [labels[i] for i in tr], rng, n_aug)
    Xte, yte = build_xy([lms[i] for i in te], [hands[i] for i in te], [labels[i] for i in te])

    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    classes = list(model.classes_)
    report = {
        "accuracy": float(accuracy_score(yte, pred)),
        "classes": classes,
        "text": classification_report(yte, pred, zero_division=0),
        "confusion": confusion_matrix(yte, pred, labels=classes),
        "n_train": len(tr), "n_test": len(te),
    }
    # Refit on all data (with augmentation) for the final model.
    Xall, yall = build_xy(lms, hands, labels, rng, n_aug)
    model.fit(Xall, yall)
    return model, report


def save(model, path=config.SIGN_MODEL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path=config.SIGN_MODEL):
    return joblib.load(path) if os.path.exists(path) else None


def predict(model, landmarks, handedness):
    """Return (label, probability) for one hand."""
    proba = model.predict_proba(extract(landmarks, handedness).reshape(1, -1))[0]
    i = int(np.argmax(proba))
    return model.classes_[i], float(proba[i])


def predict_top(model, landmarks, handedness, k=3):
    """Return the k most likely (label, probability) pairs, best first."""
    proba = model.predict_proba(extract(landmarks, handedness).reshape(1, -1))[0]
    order = np.argsort(proba)[::-1][:k]
    return [(model.classes_[i], float(proba[i])) for i in order]
