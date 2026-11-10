"""Draws the translator window: camera on the left, live captions, and a side panel."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

W, H = 1280, 720
VIDEO_W = 920
PANEL_X = VIDEO_W
PAD = 32

# Colours (RGB). Deep signage blue with one amber accent for "what you're signing now".
INK = (19, 41, 75)
INK_2 = (36, 62, 104)
INK_3 = (58, 88, 136)
PAPER = (244, 246, 250)
MUTED = (150, 168, 196)
AMBER = (242, 169, 0)
RED = (226, 75, 74)
CAPTION_BG = (8, 14, 26)

_FONT_CANDIDATES = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
_cache = {}

HELP_KEYS = [("H", "Show or hide this help"), ("P", "Pause, so resting hands don't type"),
             ("Space", "Add a space"), ("Bksp", "Delete the last letter"), ("C", "Clear everything"),
             ("S", "Read the sentence aloud"), ("T", "Save the sentence as a text file"),
             ("M", "Switch between trained signs and built-in gestures"), ("Q", "Quit")]
BUILTIN_GESTURES = [("0 to 5", "Hold up that many fingers"), ("Good", "Thumbs up"),
                    ("Bad", "Thumbs down"), ("I love you", "Thumb, index and little finger out")]


def font(size, bold=False):
    key = (size, bold)
    if key not in _cache:
        for path in _FONT_CANDIDATES[bold]:
            if os.path.exists(path):
                _cache[key] = ImageFont.truetype(path, size)
                break
        else:
            _cache[key] = ImageFont.load_default()
    return _cache[key]


def _wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split(" "):
        test = (line + " " + word).strip() if line else word
        if draw.textlength(test, font=fnt) <= width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    lines.append(line)
    return lines


def sign_name(sign):
    if sign in config.DISPLAY:
        return config.DISPLAY[sign][0].upper() + config.DISPLAY[sign][1:]
    if sign == "SPACE":
        return "Space"
    if sign == "DELETE":
        return "Delete"
    return f"Letter {sign}" if len(sign) == 1 and sign.isalpha() else sign.replace("_", " ").title()


def sign_glyph(sign):
    return {"SPACE": "Space", "DELETE": "Del", "I_LOVE_YOU": "ILY", "GOOD": "Good", "BAD": "Bad"}.get(sign, sign)


def hand_guidance(landmarks):
    """A short tip about hand position, or None when the hand is well placed."""
    if not landmarks:
        return None
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    if min(xs) < 0.02 or max(xs) > 0.98 or min(ys) < 0.02 or max(ys) > 0.98:
        return "Keep your whole hand in view"
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    if size < 0.14:
        return "Move your hand a little closer"
    if size > 0.7:
        return "Move your hand back a little"
    return None


def fit_video(frame):
    """Crop/scale the camera frame to fill the video area."""
    h, w = frame.shape[:2]
    target = VIDEO_W / H
    if w / h > target:
        nw = int(h * target)
        x0 = (w - nw) // 2
        frame = frame[:, x0:x0 + nw]
    else:
        nh = int(w / target)
        y0 = (h - nh) // 2
        frame = frame[y0:y0 + nh]
    return cv2.resize(frame, (VIDEO_W, H))


def _dashed_rect(d, box, color, dash=18, gap=12, width=3):
    x0, y0, x1, y1 = box
    for a, b, horizontal, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(p + dash, b)
            d.line([(p, fixed), (q, fixed)] if horizontal else [(fixed, p), (fixed, q)], fill=color, width=width)
            p += dash + gap


def _pill(d, center_x, y, text, fill, text_fill, size=20):
    f = font(size, True)
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([center_x - tw / 2 - 18, y, center_x + tw / 2 + 18, y + size + 20], radius=(size + 20) // 2, fill=fill)
    d.text((center_x - tw / 2, y + 9), text, font=f, fill=text_fill)


def _captions(d, sentence, pending, progress, paused):
    cap_font = font(34, True)
    shown = sentence.text
    ghost = ""
    if pending and progress > 0 and not paused and pending not in ("SPACE", "DELETE"):
        ghost = config.DISPLAY.get(pending, pending)
        if len(ghost) == 1 and shown.strip():
            ghost = ghost.lower()
        elif len(ghost) > 1 and shown and not shown.endswith(" "):
            ghost = " " + ghost
    lines = _wrap(d, (shown + ghost) or " ", cap_font, VIDEO_W - 2 * 40 - 30)[-2:]
    bar_h = 40 + 46 * len(lines)
    d.rectangle([0, H - bar_h, VIDEO_W, H], fill=CAPTION_BG + (215,))
    y = H - bar_h + 20
    if not shown and not ghost:
        d.text((40, y + 2), "Start signing. Your words appear here.", font=font(28), fill=MUTED)
        return
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        x = 40
        if last and ghost and line.endswith(ghost.strip()):
            head = line[: len(line) - len(ghost.strip())]
            d.text((x, y), head, font=cap_font, fill=PAPER)
            x += d.textlength(head, font=cap_font)
            g = ghost.strip()
            d.text((x, y), g, font=cap_font, fill=AMBER + (int(90 + 165 * progress),))
            gw = d.textlength(g, font=cap_font)
            d.line([x, y + 44, x + gw * progress, y + 44], fill=AMBER, width=4)
            x += gw
        else:
            d.text((x, y), line, font=cap_font, fill=PAPER)
            x += d.textlength(line, font=cap_font)
        if last:  # text cursor
            d.rectangle([x + 4, y + 6, x + 7, y + 40], fill=AMBER if not paused else MUTED)
        y += 46


def _help_overlay(canvas, builtin_mode):
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rectangle([0, 0, W, H], fill=(8, 14, 26, 225))
    x, y = 120, 56
    d.text((x, y), "How to use the translator", font=font(34, True), fill=PAPER)
    close = "Press any key to close"
    d.text((W - 120 - d.textlength(close, font=font(18)), y + 12), close, font=font(18), fill=MUTED)
    y += 60
    steps = [
        "Raise one hand inside the frame, about an arm's length from the camera.",
        "Make a sign and hold it still. The ring fills up, and the sign is typed.",
        "To type the same letter twice, lower your hand briefly between them.",
    ]
    for n, s in enumerate(steps, 1):
        d.ellipse([x, y, x + 30, y + 30], fill=AMBER)
        d.text((x + 15 - d.textlength(str(n), font=font(18, True)) / 2, y + 3), str(n), font=font(18, True), fill=INK)
        d.text((x + 46, y + 2), s, font=font(21), fill=PAPER)
        y += 42
    y += 18
    col2 = 700
    d.text((x, y), "Keys", font=font(22, True), fill=AMBER)
    d.text((col2, y), "Built-in gestures" if builtin_mode else "Signs you can train", font=font(22, True), fill=AMBER)
    y += 38
    ky = y
    for k, v in HELP_KEYS:
        kf = font(15, True)
        d.rounded_rectangle([x, ky, x + 64, ky + 28], radius=6, outline=MUTED, width=1)
        d.text((x + 32 - d.textlength(k, font=kf) / 2, ky + 4), k, font=kf, fill=PAPER)
        d.text((x + 80, ky + 3), v, font=font(18), fill=PAPER)
        ky += 35
    gy = y
    if builtin_mode:
        for name, how in BUILTIN_GESTURES:
            d.text((col2, gy), name, font=font(18, True), fill=PAPER)
            d.text((col2 + 120, gy), how, font=font(18), fill=MUTED)
            gy += 38
        gy += 10
        for line in _wrap(d, "Record and train the ASL alphabet to spell words: run collect.py, then train.py.",
                          font(17), W - col2 - 80):
            d.text((col2, gy), line, font=font(17), fill=MUTED)
            gy += 26
    else:
        for line in _wrap(d, "ASL letters A to Y (not J or Z, which use movement), plus I love you, Good, "
                          "Space and Delete. Re-record any sign that gets confused: python collect.py --signs A,E",
                          font(18), W - col2 - 80):
            d.text((col2, gy), line, font=font(18), fill=PAPER)
            gy += 30


def render(video_bgr, *, sentence, pending=None, progress=0.0, confidence=0.0, alternatives=None,
           history=None, mode="Trained signs", landmarks=None, paused=False, flash=0.0,
           show_help=False, notice=None, fps=0.0):
    """Compose the full window and return it as a BGR image."""
    canvas = Image.new("RGB", (W, H), INK)
    canvas.paste(Image.fromarray(cv2.cvtColor(video_bgr, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas, "RGBA")
    hand_seen = landmarks is not None

    # Video overlays: hand guide frame, position tips, pause state
    if paused:
        d.rectangle([0, 0, VIDEO_W, H], fill=(8, 14, 26, 150))
        _pill(d, VIDEO_W / 2, H / 2 - 60, "Paused. Press P to continue signing.", PAPER, INK, 22)
    elif not hand_seen:
        box = [VIDEO_W * 0.3, H * 0.14, VIDEO_W * 0.7, H * 0.62]
        _dashed_rect(d, box, PAPER + (170,))
        msg = "Raise your hand here"
        d.text((VIDEO_W / 2 - d.textlength(msg, font=font(22, True)) / 2, box[3] + 16), msg, font=font(22, True), fill=PAPER)
    else:
        tip = hand_guidance(landmarks)
        if tip:
            _pill(d, VIDEO_W / 2, 28, tip, AMBER, INK)
    if notice:
        _pill(d, VIDEO_W / 2, 28 if not (hand_seen and hand_guidance(landmarks)) else 82, notice, PAPER, INK, 18)
    _captions(d, sentence, pending, progress, paused)

    # Side panel
    x0, x1 = PANEL_X + PAD, W - PAD
    d.text((x0, 30), "Sign translator", font=font(26, True), fill=PAPER)
    d.text((x0, 66), mode, font=font(17), fill=MUTED)
    if fps:
        ft = f"{fps:.0f} fps"
        d.text((x1 - d.textlength(ft, font=font(17)), 66), ft, font=font(17), fill=MUTED)

    cx, cy, r = PANEL_X + (W - PANEL_X) // 2, 206, 86
    ring_col = AMBER if flash > 0 else INK_2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ring_col, width=10)
    if progress > 0 and not paused:
        d.arc([cx - r, cy - r, cx + r, cy + r], start=-90, end=-90 + 360 * progress, fill=AMBER, width=10)
    shown_sign = pending if not paused else None
    if flash > 0 and history:
        shown_sign = history[-1]
    if shown_sign:
        label = sign_glyph(shown_sign)
        size = 96 if len(label) == 1 else 50 if len(label) <= 3 else 36
        f = font(size, True)
        bbox = d.textbbox((0, 0), label, font=f)
        d.text((cx - d.textlength(label, font=f) / 2, cy - (bbox[3] + bbox[1]) / 2), label, font=f,
               fill=AMBER if (progress > 0 or flash > 0) else PAPER)
        caption = "Typed" if flash > 0 else sign_name(shown_sign)
        nf = font(20, flash > 0)
        d.text((cx - d.textlength(caption, font=nf) / 2, cy + r + 14), caption, font=nf, fill=AMBER if flash > 0 else PAPER)
    else:
        msg = "Paused" if paused else ("Show your hand" if not hand_seen else "Hold a sign steady")
        nf = font(19)
        d.text((cx - d.textlength(msg, font=nf) / 2, cy - 12), msg, font=nf, fill=MUTED)

    # Confidence and other possibilities
    y = cy + r + 56
    conf = confidence if (hand_seen and not paused) else 0.0
    d.text((x0, y), "Confidence", font=font(17), fill=MUTED)
    ct = f"{conf:.0%}" if conf else ""
    d.text((x1 - d.textlength(ct, font=font(17, True)), y), ct, font=font(17, True), fill=PAPER)
    d.rounded_rectangle([x0, y + 28, x1, y + 36], radius=4, fill=INK_2)
    if conf:
        d.rounded_rectangle([x0, y + 28, x0 + (x1 - x0) * conf, y + 36], radius=4, fill=PAPER)
    y += 56
    alts = [(s, p) for s, p in (alternatives or []) if s != pending and p >= 0.05][:2]
    if alts and conf and not paused:
        d.text((x0, y), "Also possible", font=font(17), fill=MUTED)
        y += 28
        for s, p in alts:
            d.text((x0, y), sign_glyph(s), font=font(17, True), fill=PAPER)
            d.rounded_rectangle([x0 + 70, y + 8, x1, y + 14], radius=3, fill=INK_2)
            d.rounded_rectangle([x0 + 70, y + 8, x0 + 70 + (x1 - x0 - 70) * p, y + 14], radius=3, fill=INK_3)
            y += 28
    y = max(y + 8, 520)

    # Recently typed
    d.text((x0, y), "Recently typed", font=font(17), fill=MUTED)
    y += 30
    cx0 = x0
    if history:
        f = font(16, True)
        chips, used = [], 0
        for s in reversed(history):  # newest first, keep as many as fit
            w = d.textlength(sign_glyph(s), font=f) + 20
            if used + w > x1 - x0:
                break
            chips.insert(0, (sign_glyph(s), w))
            used += w + 8
        for i, (t, w) in enumerate(chips):
            newest = i == len(chips) - 1
            d.rounded_rectangle([cx0, y, cx0 + w, y + 30], radius=6, fill=AMBER if newest and flash > 0 else INK_2)
            d.text((cx0 + 10, y + 5), t, font=f, fill=INK if newest and flash > 0 else PAPER)
            cx0 += w + 8
    else:
        d.text((x0, y + 4), "Nothing yet", font=font(16), fill=INK_3)

    # Footer hint
    hint_f = font(15, True)
    fx = x0
    for k, v in (("H", "Help"), ("P", "Pause" if not paused else "Resume"), ("S", "Speak")):
        d.rounded_rectangle([fx, H - 52, fx + 28, H - 26], radius=5, outline=MUTED, width=1)
        d.text((fx + 14 - d.textlength(k, font=hint_f) / 2, H - 49), k, font=hint_f, fill=PAPER)
        d.text((fx + 36, H - 50), v, font=font(16), fill=MUTED)
        fx += 36 + d.textlength(v, font=font(16)) + 18

    if show_help:
        _help_overlay(canvas, builtin_mode=mode != "Trained signs")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def message_screen(video_bgr, title, body, footer=None, progress=None, recording=False,
                   counter=None, step=None, landmarks=None):
    """Window used by the recording tool. `step` is (current, total) across all signs."""
    canvas = Image.new("RGB", (W, H), INK)
    canvas.paste(Image.fromarray(cv2.cvtColor(video_bgr, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas, "RGBA")

    if recording:
        d.rectangle([0, 0, VIDEO_W - 1, H - 1], outline=RED, width=6)
        d.ellipse([28, 30, 46, 48], fill=RED)
        d.text((56, 26), "REC", font=font(20, True), fill=PAPER)
    elif landmarks is None:
        box = [VIDEO_W * 0.3, H * 0.14, VIDEO_W * 0.7, H * 0.62]
        _dashed_rect(d, box, PAPER + (170,))
    tip = hand_guidance(landmarks)
    if tip:
        _pill(d, VIDEO_W / 2, 28, tip, AMBER, INK)

    x0, x1 = PANEL_X + PAD, W - PAD
    d.text((x0, 30), "Record signs", font=font(26, True), fill=PAPER)
    if step:
        n, total = step
        d.text((x0, 66), f"Sign {n} of {total}", font=font(17), fill=MUTED)
        seg_w = (x1 - x0 - (total - 1) * 3) / total
        for i in range(total):
            sx = x0 + i * (seg_w + 3)
            d.rectangle([sx, 96, sx + seg_w, 101], fill=AMBER if i < n - 1 else (PAPER if i == n - 1 else INK_2))
    tf = font(120 if len(title) == 1 else 38, True)
    d.text((x0, 116), title, font=tf, fill=AMBER)
    y = 280 if len(title) == 1 else 180
    for line in _wrap(d, body, font(20), x1 - x0):
        d.text((x0, y), line, font=font(20), fill=PAPER)
        y += 30
    if progress is not None:
        y += 20
        d.rounded_rectangle([x0, y, x1, y + 12], radius=6, fill=INK_2)
        d.rounded_rectangle([x0, y, x0 + (x1 - x0) * progress, y + 12], radius=6, fill=AMBER)
        if counter:
            d.text((x0, y + 22), f"{counter[0]} of {counter[1]} samples", font=font(17), fill=MUTED)
    if footer:
        lines = _wrap(d, footer, font(17), x1 - x0)
        fy = H - 40 - 26 * len(lines)
        for line in lines:
            d.text((x0, fy), line, font=font(17), fill=MUTED)
            fy += 26
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
