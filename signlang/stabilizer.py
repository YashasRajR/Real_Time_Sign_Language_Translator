"""Turn noisy per-frame predictions into deliberate, one-at-a-time sign entries."""
from collections import deque

from . import config


class Stabilizer:
    """A sign is typed once it has been held steadily for `hold` frames.

    The same sign is not typed again until the hand changes shape or leaves the
    frame, so holding "L" types one L, not LLLLL.
    """

    def __init__(self, hold=config.HOLD_FRAMES, release=config.RELEASE_FRAMES,
                 min_conf=config.MIN_CONFIDENCE, window=8):
        self.hold, self.release, self.min_conf = hold, release, min_conf
        self.recent = deque(maxlen=window)
        self.candidate, self.streak = None, 0
        self.last_committed, self.missing = None, 0

    def update(self, label, conf):
        """Feed one frame. Returns the sign to type now, or None."""
        if label is None or conf < self.min_conf:
            self.missing += 1
            self.recent.append(None)
            if self.missing >= self.release:
                self.last_committed = None  # hand left: the same sign may be typed again
            self.candidate, self.streak = None, 0
            return None
        self.missing = 0
        self.recent.append(label)
        top = max(set(self.recent), key=self.recent.count)
        if top != self.candidate:
            self.candidate, self.streak = top, 0
        self.streak += 1
        if self.candidate is None:
            return None
        if self.streak == self.hold and self.candidate != self.last_committed:
            self.last_committed = self.candidate
            return self.candidate
        return None

    @property
    def progress(self):
        """0..1 for how close the current sign is to being typed (for the UI)."""
        if self.candidate is None or self.candidate == self.last_committed:
            return 0.0
        return min(1.0, self.streak / self.hold)

    def reset(self):
        self.__init__(self.hold, self.release, self.min_conf, self.recent.maxlen)
