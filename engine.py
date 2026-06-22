import cv2
import numpy as np
import time
import collections
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}

def _ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODEL_DIR / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"[MODEL] Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[MODEL] Saved {path}")

class BackgroundModel:
    def __init__(self, h, w, n_frames=90):
        self.buf   = collections.deque(maxlen=n_frames)
        self.bg    = None
        self.ready = False
        self.h, self.w = h, w
        self._tick = 0

    def update(self, frame_f32):
        self.buf.append(frame_f32)
        self._tick += 1
        if len(self.buf) >= 15 and self._tick % 6 == 0:
            self.bg    = np.mean(self.buf, axis=0).astype(np.float32)
            self.ready = True

    def get(self):
        return self.bg if self.ready else None


SEG_W, SEG_H = 320, 180

class SegmentationEngine:
    def __init__(self):
        _ensure_models()
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODEL_DIR / "selfie_segmenter.tflite")
            ),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
        )
        self.seg = vision.ImageSegmenter.create_from_options(options)
        self._prev_mask = None
        self._frame_idx = 0
        self._ts_ms = 0

    def get_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_idx += 1
        if self._frame_idx % 2 == 0 and self._prev_mask is not None:
            return self._prev_mask
        small = cv2.resize(frame_bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        res = self.seg.segment_for_video(mp_img, self._ts_ms)
        if not res.confidence_masks:
            return self._prev_mask if self._prev_mask is not None else \
                   np.zeros((h, w), dtype=np.float32)
        mask = np.squeeze(res.confidence_masks[0].numpy_view()).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if self._prev_mask is not None:
            mask = 0.6 * mask + 0.4 * self._prev_mask
        _, hard = cv2.threshold(mask, 0.25, 1.0, cv2.THRESH_BINARY)
        hard8 = (hard * 255).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        hard8 = cv2.morphologyEx(hard8, cv2.MORPH_CLOSE, kernel)
        hard8 = cv2.dilate(hard8, kernel, iterations=1)
        hard = hard8.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(hard, (9, 9), 0)
        mask = np.clip(mask * 1.3, 0, 1).astype(np.float32)
        self._prev_mask = mask
        return mask


HAND_W = 480
PINCH_RATIO_THRESHOLD = 0.45

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

class HandTracker:
    def __init__(self):
        _ensure_models()
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODEL_DIR / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process(self, frame_bgr):
        h, w  = frame_bgr.shape[:2]
        sw    = min(w, HAND_W)
        sh    = int(h * sw / w)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        return self.hands.detect_for_video(mp_img, self._ts_ms)

    def get_info(self, results, w, h):
        out = {"hand_count": 0, "tips": None, "fingers_touching": False, "all_points": []}
        if not results or not results.hand_landmarks:
            return out

        out["hand_count"] = len(results.hand_landmarks)
        index_tips = []
        any_pinch  = False

        for lms in results.hand_landmarks:
            thumb_tip = lms[4]
            index_tip = lms[8]
            wrist     = lms[0]
            mid_mcp   = lms[9]

            index_tips.append((int(index_tip.x * w), int(index_tip.y * h)))
            out["all_points"].append([(int(p.x * w), int(p.y * h)) for p in lms])

            palm_size  = ((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)**0.5
            pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

            if palm_size > 1e-6:
                ratio = pinch_dist / palm_size
                if ratio < PINCH_RATIO_THRESHOLD:
                    any_pinch = True

        out["fingers_touching"] = any_pinch

        if len(index_tips) >= 2:
            out["tips"] = (index_tips[0], index_tips[1])

        return out


def _rounded_rect_mask(h, w, radius):
    mask = np.zeros((h, w), dtype=np.float32)
    r = min(radius, h // 2, w // 2)
    cv2.rectangle(mask, (r, 0), (w - r, h), 1.0, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 1.0, -1)
    cv2.circle(mask, (r, r), r, 1.0, -1)
    cv2.circle(mask, (w - r, r), r, 1.0, -1)
    cv2.circle(mask, (r, h - r), r, 1.0, -1)
    cv2.circle(mask, (w - r, h - r), r, 1.0, -1)
    return cv2.GaussianBlur(mask, (5, 5), 0)


class GlassUI:
    """Frosted / liquid-glass overlay helpers."""

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    MONO  = cv2.FONT_HERSHEY_DUPLEX
    ICE   = (255, 248, 242)
    MINT  = (210, 255, 220)
    LILAC = (255, 190, 235)
    SKY   = (255, 220, 180)
    WHITE = (255, 255, 255)
    MUTED = (195, 205, 220)
    DIM   = (120, 130, 150)

    def __init__(self):
        self._phase = 0.0

    def tick(self):
        self._phase = time.time()

    def _shimmer(self, speed=2.2, lo=0.55, hi=1.0):
        return lo + (hi - lo) * (0.5 + 0.5 * np.sin(self._phase * speed))

    def glass_panel(self, frame, x, y, pw, ph, radius=18, blur=31,
                    fill=0.38, tint=None, border=True):
        if tint is None:
            tint = np.array(self.ICE, dtype=np.float32)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + pw), min(h, y + ph)
        if x2 <= x1 or y2 <= y1:
            return
        pw, ph = x2 - x1, y2 - y1
        roi = frame[y1:y2, x1:x2].copy()
        k = blur | 1
        frosted = cv2.GaussianBlur(roi, (k, k), 0).astype(np.float32)
        frosted = frosted * (1.0 - fill) + tint * fill
        spec = np.linspace(0.42, 0.0, ph, dtype=np.float32)[:, None, None]
        frosted += np.array(self.WHITE, dtype=np.float32) * spec * 0.22
        mask = _rounded_rect_mask(ph, pw, radius)[:, :, None]
        base = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = np.clip(
            base * (1.0 - mask) + frosted * mask, 0, 255
        ).astype(np.uint8)
        if border:
            self._liquid_border(frame, x1, y1, x2, y2, radius)

    def _liquid_border(self, frame, x1, y1, x2, y2, radius):
        pulse = self._shimmer(3.0, 0.45, 1.0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      (int(255 * pulse), int(210 * pulse), int(255 * pulse)), 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2),
                      (int(180 * pulse), int(235 * pulse), int(255 * pulse)), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def pill(self, frame, text, cx, cy, scale=0.46, accent=None):
        if accent is None:
            accent = self.LILAC
        (tw, th), _ = cv2.getTextSize(text, self.FONT, scale, 1)
        pad_x, pad_y = 18, 10
        px = cx - tw // 2 - pad_x
        py = cy - th // 2 - pad_y
        self.glass_panel(frame, px, py, tw + pad_x * 2, th + pad_y * 2, radius=14, fill=0.44)
        cv2.putText(frame, text, (cx - tw // 2 + 1, cy + th // 2 + 1),
                    self.FONT, scale, (40, 40, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                    self.FONT, scale, accent, 1, cv2.LINE_AA)

    def draw_calibration(self, frame, countdown, total):
        self.tick()
        h, w = frame.shape[:2]
        pw, ph = min(520, w - 80), 220
        px, py = (w - pw) // 2, (h - ph) // 2
        self.glass_panel(frame, px, py, pw, ph, radius=24, blur=35, fill=0.48)
        self.pill(frame, "CALIBRATING BACKGROUND", w // 2, py + 52, 0.58, self.SKY)
        cv2.putText(frame, "Stand completely still", (px + 42, py + 98),
                    self.FONT, 0.52, self.MUTED, 1, cv2.LINE_AA)
        prog_w = pw - 80
        prog_x = px + 40
        prog_y = py + ph - 52
        cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + 8),
                      (90, 95, 110), -1, cv2.LINE_AA)
        done = 1.0 - (countdown / max(total, 1))
        fill = int(prog_w * done)
        if fill > 0:
            cv2.rectangle(frame, (prog_x, prog_y), (prog_x + fill, prog_y + 8),
                          self.MINT, -1, cv2.LINE_AA)
        cv2.putText(frame, str(countdown), (w // 2 - 28, py + 158),
                    self.FONT, 2.8, self.WHITE, 4, cv2.LINE_AA)
        cv2.putText(frame, str(countdown), (w // 2 - 30, py + 156),
                    self.FONT, 2.8, self.LILAC, 2, cv2.LINE_AA)


def draw_hand_mesh(frame, all_points):
    if not all_points:
        return frame
    for pts in all_points:
        if len(pts) < 21:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], (255, 210, 240), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 7 if i in (4, 8) else 4
            cv2.circle(frame, p, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, r, (255, 170, 230) if i in (4, 8) else (220, 240, 255), -1, cv2.LINE_AA)
    return frame


class PortalBox:
    def __init__(self):
        self.tl           = None
        self.br           = None
        self.active       = False
        self.invisible    = False
        self._alpha       = 0.0
        self._scan_offset = 0
        self._touch_cd    = 0
        self._glass       = GlassUI()

    def update(self, info: dict):
        tips             = info["tips"]
        fingers_touching = info["fingers_touching"]
        hand_count       = info["hand_count"]

        if tips is not None:
            p1, p2 = tips
            self.tl = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            self.br = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            w_box = self.br[0] - self.tl[0]
            h_box = self.br[1] - self.tl[1]
            self.active = w_box > 80 and h_box > 80
        else:
            self.active = False

        if self._touch_cd > 0:
            self._touch_cd -= 1

        if fingers_touching and hand_count >= 1 and self._touch_cd == 0:
            self.invisible = not self.invisible
            self._touch_cd = 20

    def update_alpha(self):
        target = 1.0 if self.invisible else 0.0
        speed  = 0.10
        if self._alpha < target:
            self._alpha = min(target, self._alpha + speed)
        else:
            self._alpha = max(target, self._alpha - speed)

    def render(self, frame, seg_mask, bg, all_points=None):
        h, w  = frame.shape[:2]
        alpha = self._alpha
        self._glass.tick()

        if alpha > 0.01 and bg is not None:
            roi_f    = frame.astype(np.float32)
            roi_bg   = bg
            eff_mask = seg_mask if alpha < 0.97 else np.minimum(seg_mask * 1.6, 1.0)
            roi_mask = eff_mask[:, :, np.newaxis]
            blend    = roi_f * (1 - roi_mask * alpha) + roi_bg * (roi_mask * alpha)
            np.clip(blend, 0, 255, out=blend)
            frame[:, :] = blend.astype(np.uint8)
            if alpha > 0.05:
                self._draw_scanlines(frame, seg_mask, 0, 0, w, h, alpha)
                self._draw_specter_glow(frame, seg_mask, alpha)

        if all_points:
            draw_hand_mesh(frame, all_points)

        if self.active and self.tl and self.br:
            x1, y1 = self.tl
            x2, y2 = self.br
            pulse  = self._glass._shimmer(2.6, 0.5, 1.0)
            pad    = 10
            self._glass.glass_panel(
                frame, x1 - pad, y1 - pad,
                (x2 - x1) + pad * 2, (y2 - y1) + pad * 2,
                radius=20, blur=27, fill=0.28,
                tint=np.array([255, 235, 250], dtype=np.float32),
            )
            for thick, blend_str, color in [
                (10, 0.10, (int(255 * pulse), int(200 * pulse), int(255 * pulse))),
                (5,  0.28, (int(255 * pulse), int(230 * pulse), int(255 * pulse))),
                (2,  0.62, (int(220 * pulse), int(245 * pulse), int(255 * pulse))),
                (1,  1.00, (255, 255, 255)),
            ]:
                ov = frame.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
                cv2.addWeighted(ov, blend_str, frame, 1 - blend_str, 0, frame)

            l = max(22, min(44, (x2 - x1) // 6, (y2 - y1) // 6))
            ccol = (int(255 * pulse), int(220 * pulse), int(255 * pulse))
            for (cx, cy), (ddx1, ddy1), (ddx2, ddy2) in [
                ((x1, y1), (l, 0), (0, l)),
                ((x2, y1), (-l, 0), (0, l)),
                ((x1, y2), (l, 0), (0, -l)),
                ((x2, y2), (-l, 0), (0, -l)),
            ]:
                cv2.line(frame, (cx, cy), (cx + ddx1, cy + ddy1), ccol, 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx + ddx2, cy + ddy2), ccol, 3, cv2.LINE_AA)

            msg = "Pinch to vanish" if not self.invisible else "Pinch to reappear"
            self._glass.pill(frame, msg.upper(), (x1 + x2) // 2, (y1 + y2) // 2, 0.48)

        self._scan_offset = (self._scan_offset + 3) % max(1, frame.shape[0])
        return frame

    def _draw_specter_glow(self, frame, seg_mask, alpha):
        glow = cv2.GaussianBlur((seg_mask * 255).astype(np.uint8), (31, 31), 0)
        tint = np.array([255, 180, 240], dtype=np.float32)
        g = glow.astype(np.float32)[:, :, None] / 255.0
        roi = frame.astype(np.float32)
        frame[:, :] = np.clip(roi + tint * g * (0.18 * alpha), 0, 255).astype(np.uint8)

    def _draw_scanlines(self, frame, seg_mask, x1, y1, x2, y2, alpha):
        scan_col = np.array([255, 210, 245], dtype=np.float32)
        roi_h = y2 - y1
        if roi_h <= 0:
            return
        rows = (np.arange(0, roi_h, 6) + self._scan_offset) % roi_h + y1
        rows = rows[rows < y2].astype(int)
        if rows.size == 0:
            return
        sm    = seg_mask[rows, x1:x2]
        strip = frame[rows, x1:x2].astype(np.float32)
        t     = 0.10 * alpha
        strip = strip*(1 - t*sm[:,:,None]) + scan_col*t*sm[:,:,None]
        frame[rows, x1:x2] = strip.astype(np.uint8)


class HUD:
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    MONO  = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, h, w, dev_str):
        self.h, self.w = h, w
        self.dev_str   = dev_str
        self._fps_buf  = collections.deque(maxlen=30)
        self._last_t   = time.time()
        self._glass    = GlassUI()

    def tick(self):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now
        self._glass.tick()

    @property
    def fps(self):
        return np.mean(self._fps_buf) if self._fps_buf else 0.0

    def draw(self, frame, portal, info: dict):
        h, w       = self.h, self.w
        alpha      = portal._alpha
        hand_count = info["hand_count"]
        touching   = info["fingers_touching"]
        g          = self._glass

        g.glass_panel(frame, 12, 10, min(340, w - 24), 56, radius=16, blur=25, fill=0.42)
        g.glass_panel(frame, w - 250, 10, 238, 56, radius=16, blur=25, fill=0.42)
        g.glass_panel(frame, 12, h - 66, min(w - 24, 520), 54, radius=16, blur=25, fill=0.42)

        cv2.putText(frame, "SPECTER", (28, 46),
                    g.FONT, 0.78, g.LILAC, 2, cv2.LINE_AA)
        cv2.putText(frame, "live invisibility", (148, 46),
                    g.FONT, 0.42, g.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.fps:05.1f} fps", (w - 220, 46),
                    g.MONO, 0.58, g.MINT, 1, cv2.LINE_AA)
        cv2.putText(frame, self.dev_str, (w - 118, 46),
                    g.MONO, 0.40, g.DIM, 1, cv2.LINE_AA)

        status_x = w - 228
        if alpha > 0.05:
            pulse = int(abs(np.sin(time.time() * 5)) * 70 + 185)
            cv2.circle(frame, (status_x, 30), 7, (pulse, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "ACTIVE", (status_x + 14, 36),
                        g.FONT, 0.42, g.LILAC, 1, cv2.LINE_AA)
        elif portal.active:
            cv2.circle(frame, (status_x, 30), 7, g.MINT, -1, cv2.LINE_AA)
            cv2.putText(frame, "READY", (status_x + 14, 36),
                        g.FONT, 0.42, g.MINT, 1, cv2.LINE_AA)

        bx, by, bw2 = 28, h - 34, 180
        cv2.rectangle(frame, (bx, by), (bx + bw2, by + 8), (80, 85, 100), -1, cv2.LINE_AA)
        filled = int(bw2 * alpha)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + 8), g.LILAC, -1, cv2.LINE_AA)
        cv2.putText(frame, f"cloak {int(alpha * 100):3d}%", (bx, by - 10),
                    g.FONT, 0.44, g.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"hands {hand_count}", (bx + 210, by + 6),
                    g.FONT, 0.44, g.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (w - 118, h - 28),
                    g.MONO, 0.46, g.DIM, 1, cv2.LINE_AA)

        if touching:
            g.pill(frame, "PINCH DETECTED", w // 2, h - 92, 0.52, g.MINT)
        elif alpha < 0.3 and not portal.active:
            hint = ("Spread both hands to open portal"
                    if hand_count < 2 else "Pinch thumb + index to vanish")
            g.pill(frame, hint, w // 2, h - 92, 0.40, g.SKY)

        return frame