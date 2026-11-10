"""Train the sign classifier on your recorded samples.

    python train.py

Prints the accuracy on held-out samples and saves:
  models/sign_model.joblib       the trained model used by translate.py
  models/confusion_matrix.png    which signs get mixed up
"""
import os
from collections import Counter

import cv2
import numpy as np

from signlang import classifier, config


def draw_confusion(cm, classes, path):
    """Save the confusion matrix as an image (rows = true sign, columns = predicted)."""
    n = len(classes)
    cell, pad = 28, 70
    img = np.full((pad + n * cell + 10, pad + n * cell + 10, 3), 255, np.uint8)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            color = (int(255 - 180 * v), int(255 - 140 * v), int(255 - 60 * v)) if i == j else \
                    (int(255 - 40 * v), int(255 - 160 * v), int(255 - 200 * v))
            x, y = pad + j * cell, pad + i * cell
            cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), color, -1)
            if cm[i, j]:
                cv2.putText(img, str(cm[i, j]), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
        short = classes[i][:3]
        cv2.putText(img, short, (5, pad + i * cell + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, short, (pad + i * cell + 2, pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    if not os.path.exists(config.SAMPLES_CSV):
        raise SystemExit("No samples yet. Record some first:  python collect.py")
    lms, hands, labels = classifier.load_samples()
    counts = Counter(labels)
    print(f"Loaded {len(labels)} samples for {len(counts)} signs.")
    too_few = [s for s, c in counts.items() if c < 10]
    if too_few:
        raise SystemExit(f"These signs need at least 10 samples: {', '.join(too_few)}")
    if len(counts) < 2:
        raise SystemExit("Record at least two different signs before training.")

    model, report = classifier.train(lms, hands, labels)
    classifier.save(model)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    draw_confusion(report["confusion"], report["classes"], cm_path)

    print(f"\nAccuracy on held-out samples: {report['accuracy']:.1%} "
          f"({report['n_train']} train / {report['n_test']} test)\n")
    print(report["text"])
    print(f"Model saved to {config.SIGN_MODEL}")
    print(f"Confusion matrix saved to {cm_path}")
    print("\nNow run:  python translate.py")


if __name__ == "__main__":
    main()
