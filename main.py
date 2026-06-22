import subprocess, sys, importlib.util

REQUIRED = {"cv2":"opencv-python","numpy":"numpy","mediapipe":"mediapipe"}

def _auto_install():
    missing = [pkg for mod,pkg in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"\n[AUTO-INSTALL] Installing: {', '.join(missing)}\n")
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet"]+missing)
        print("[AUTO-INSTALL] Done.\n")

_auto_install()

import cv2, numpy as np, time

def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "GPU/"+torch.cuda.get_device_name(0).split()[0]
    except ImportError:
        pass
    return "CPU"

from engine import BackgroundModel, SegmentationEngine, HandTracker, PortalBox, HUD, GlassUI

BANNER = """
  ███████╗██████╗ ███████╗ ██████╗████████╗███████╗██████╗
  ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗
  ███████╗██████╔╝█████╗  ██║        ██║   █████╗  ██████╔╝
  ╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══╝  ██╔══██╗
  ███████║██║     ███████╗╚██████╗   ██║   ███████╗██║  ██║
  ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

  Real-time invisibility · gesture-controlled

  1. Stand still 3s     ->  background captured
  2. Spread both hands  ->  portal opens
  3. Pinch thumb+index  ->  vanish
  4. Pinch again        ->  reappear

  R = recalibrate  |  S = screenshot  |  Q = quit
"""
WINDOW = "Specter"

def run_calibration(cap, bg_model, w, h, seconds=3):
    print(f"[CAL] Stand still {seconds}s ...")
    glass = GlassUI()
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        bg_model.update(frame.astype(np.float32))
        rem  = max(1, int(seconds - (time.time() - start)) + 1)
        disp = frame.copy()
        glass.draw_calibration(disp, rem, seconds)
        cv2.imshow(WINDOW, disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            return False
    print("[CAL] Done\n")
    return True

def _camera_backends():
    if sys.platform == "win32":
        return [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "Auto")]
    if sys.platform == "darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        return [(avf, "AVFoundation"), (cv2.CAP_ANY, "Auto")]
    return [(cv2.CAP_ANY, "Auto")]

def _read_probe_frame(cap, attempts=15):
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        time.sleep(0.05)
    return None

def _open_camera(cam_idx):
    for backend, name in _camera_backends():
        print(f"[CAM] Trying index {cam_idx} via {name} ...")
        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        probe = _read_probe_frame(cap)
        if probe is not None:
            print(f"[CAM] OK — index {cam_idx}, backend {name}")
            return cap, probe
        cap.release()
    return None, None

def main():
    print(BANNER)
    dev_str = _detect_device()
    print(f"[INFO] Device: {dev_str}")

    cam_idx  = int(sys.argv[1]) if len(sys.argv) > 1 else None
    indices  = [cam_idx] if cam_idx is not None else [0, 1]
    cap, probe = None, None
    for idx in indices:
        cap, probe = _open_camera(idx)
        if cap is not None:
            break

    if cap is None:
        print("[ERROR] Could not open webcam.")
        print("  1. Close Zoom, Teams, FaceTime, or any app using the camera")
        if sys.platform == "darwin":
            print("  2. macOS: System Settings -> Privacy & Security -> Camera")
            print("     Enable access for Terminal or Cursor (whichever you run from)")
        print("  3. Retry with another camera index: python main.py 1")
        sys.exit(1)

    h, w = probe.shape[:2]
    print(f"[INFO] Resolution: {w}x{h}")
    print("[INIT] Loading models ...")

    seg      = SegmentationEngine()
    tracker  = HandTracker()
    bg_model = BackgroundModel(h, w, n_frames=90)
    portal   = PortalBox()
    hud      = HUD(h, w, dev_str)

    print("[INIT] Ready!\n")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w,1280), min(h,720))

    if not run_calibration(cap, bg_model, w, h, seconds=3):
        cap.release(); cv2.destroyAllWindows(); return

    if bg_model.buf:
        bg_model.bg    = np.mean(bg_model.buf, axis=0).astype(np.float32)
        bg_model.ready = True

    sc_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        hud.tick()

        seg_mask = seg.get_mask(frame)
        results  = tracker.process(frame)
        info     = tracker.get_info(results, w, h)

        portal.update(info)
        portal.update_alpha()

        out = portal.render(frame, seg_mask, bg_model.get(), info["all_points"])
        out = hud.draw(out, portal, info)

        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fn = f"specter_{sc_idx:04d}.png"
            cv2.imwrite(fn, out)
            print(f"[SCREENSHOT] {fn}")
            sc_idx += 1
        elif key == ord('r'):
            bg_model.__init__(h, w)
            portal.invisible = False
            portal._alpha    = 0.0
            if not run_calibration(cap, bg_model, w, h, 3):
                break
            if bg_model.buf:
                bg_model.bg    = np.mean(bg_model.buf, axis=0).astype(np.float32)
                bg_model.ready = True

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE]")

if __name__ == "__main__":
    main()