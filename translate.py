"""Live sign language translator.

    python translate.py            # uses your trained model if there is one
    python translate.py --builtin  # built-in gestures only (no training needed)
"""
import argparse
import os
import time
from datetime import datetime

import cv2

from signlang import classifier, config, gestures, ui
from signlang.sentence import Sentence
from signlang.speech import speak
from signlang.stabilizer import Stabilizer
from signlang.tracker import HandTracker, draw_hand

WINDOW = "Sign translator"
FLASH_SECONDS = 0.6


def save_transcript(text):
    os.makedirs(config.TRANSCRIPTS_DIR, exist_ok=True)
    path = os.path.join(config.TRANSCRIPTS_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--builtin", action="store_true", help="use built-in gestures only")
    args = ap.parse_args()

    model = None if args.builtin else classifier.load()
    use_model = model is not None
    tracker = HandTracker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    sentence, stab = Sentence(), Stabilizer()
    history = []
    paused = False
    # Show the help screen automatically until the user has trained their own signs.
    show_help = model is None
    notice, notice_until = None, 0.0
    flash_until = 0.0
    fps, last, t0 = 0.0, time.time(), time.time()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ui.W, ui.H)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, so it feels like a mirror
            lms, hand = tracker.detect(frame, (time.time() - t0) * 1000)

            label, conf, alternatives = None, 0.0, []
            if lms:
                if use_model:
                    alternatives = classifier.predict_top(model, lms, hand)
                    label, conf = alternatives[0]
                else:
                    label, conf = gestures.recognise(lms, hand)
                draw_hand(frame, lms)
            # Badly placed hands give unreliable predictions, so don't type from them.
            if ui.hand_guidance(lms) or paused or show_help:
                label = None
            typed = stab.update(label, conf)
            now = time.time()
            if typed:
                sentence.add(typed)
                history.append(typed)
                flash_until = now + FLASH_SECONDS

            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-3))
            last = now
            screen = ui.render(
                ui.fit_video(frame), sentence=sentence,
                pending=stab.candidate, progress=stab.progress, confidence=conf,
                alternatives=alternatives, history=history,
                mode="Trained signs" if use_model else "Built-in gestures",
                landmarks=lms, paused=paused,
                flash=max(0.0, flash_until - now) / FLASH_SECONDS,
                show_help=show_help, notice=notice if now < notice_until else None, fps=fps,
            )
            cv2.imshow(WINDOW, screen)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:
                continue
            if show_help:
                show_help = False  # any key closes help
                if key not in (ord("q"), 27):
                    continue
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                show_help = True
            elif key == ord("p"):
                paused = not paused
                stab.reset()
            elif key == ord(" "):
                sentence.space()
            elif key == 8:
                sentence.backspace()
            elif key == ord("c"):
                sentence.clear()
                history.clear()
            elif key == ord("s"):
                speak(sentence.text)
                notice, notice_until = "Reading aloud", now + 1.5
            elif key == ord("t"):
                if sentence.text.strip():
                    path = save_transcript(sentence.text)
                    notice, notice_until = f"Saved to transcripts/{os.path.basename(path)}", now + 3
                else:
                    notice, notice_until = "Nothing to save yet", now + 2
            elif key == ord("m"):
                if model is None:
                    notice, notice_until = "Record and train your signs first", now + 3
                else:
                    use_model = not use_model
                    stab.reset()
                    notice, notice_until = "Trained signs" if use_model else "Built-in gestures", now + 2
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
