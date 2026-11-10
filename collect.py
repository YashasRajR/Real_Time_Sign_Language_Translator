"""Record your own sign samples with the webcam.

    python collect.py                     # ASL alphabet + Space and Delete
    python collect.py --set words         # I love you, Good, Space, Delete
    python collect.py --set all
    python collect.py --signs A,B,C       # only these signs (e.g. to re-record)
    python collect.py --samples 200       # more samples per sign = better accuracy

For each sign: press SPACE when your hand is ready, then hold the sign and move
it slightly (closer, further, tilt a little) while it records. S skips, Q quits.
"""
import argparse
import time
from collections import Counter

import cv2

from signlang import classifier, config, ui
from signlang.tracker import HandTracker, draw_hand

WINDOW = "Record signs"


def existing_counts():
    try:
        _, _, labels = classifier.load_samples()
        return Counter(labels)
    except FileNotFoundError:
        return Counter()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=config.SIGN_SETS, default="alphabet")
    ap.add_argument("--signs", help="comma-separated signs, e.g. A,B,SPACE")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    signs = [s.strip().upper() for s in args.signs.split(",")] if args.signs else config.SIGN_SETS[args.set]
    have = existing_counts()
    tracker = HandTracker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ui.W, ui.H)
    t0 = time.time()

    def frame_and_hand():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Camera stopped.")
        frame = cv2.flip(frame, 1)
        lms, hand = tracker.detect(frame, (time.time() - t0) * 1000)
        if lms:
            draw_hand(frame, lms)
        return frame, lms, hand

    try:
        for n, sign in enumerate(signs, 1):
            name = config.DISPLAY.get(sign, sign.replace("_", " ").title() if len(sign) > 1 else sign)
            hint = config.HINTS.get(sign, "Hold the sign steady.")
            already = f" You already have {have[sign]} samples; these will be added." if have[sign] else ""
            # Wait until ready
            while True:
                frame, lms, _ = frame_and_hand()
                status = "Hand found. Press Space to start recording." if lms else "Raise your hand inside the frame."
                cv2.imshow(WINDOW, ui.message_screen(ui.fit_video(frame), name, f"{hint}. {status}{already}",
                                                     footer="Space record, S skip, Q quit",
                                                     step=(n, len(signs)), landmarks=lms))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("s"):
                    break
                if key == ord(" "):
                    # Countdown
                    end = time.time() + 2
                    while time.time() < end:
                        frame, lms, _ = frame_and_hand()
                        left = int(end - time.time()) + 1
                        cv2.imshow(WINDOW, ui.message_screen(ui.fit_video(frame), str(left), f"Get ready to sign {name}.",
                                                             step=(n, len(signs)), landmarks=lms))
                        cv2.waitKey(1)
                    # Record
                    samples, hand_used = [], "Right"
                    while len(samples) < args.samples:
                        frame, lms, hand = frame_and_hand()
                        if lms:
                            samples.append(lms)
                            hand_used = hand
                        cv2.imshow(WINDOW, ui.message_screen(
                            ui.fit_video(frame), name,
                            "Keep signing. Move your hand a little closer, further and tilt it slightly." if lms
                            else "Hand lost. Bring it back into view.",
                            progress=len(samples) / args.samples, counter=(len(samples), args.samples),
                            recording=True, step=(n, len(signs)), landmarks=lms))
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            return
                    classifier.append_samples(sign, hand_used, samples)
                    print(f"Saved {len(samples)} samples for {sign}")
                    break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print("\nDone. Now run:  python train.py")


if __name__ == "__main__":
    main()
