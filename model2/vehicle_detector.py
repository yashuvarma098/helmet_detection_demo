

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL1 = HERE.parent / "models" / "best.pt"
WEIGHTS_DIR = HERE / "weights"                  # Model 2 weights live here, away from Model 1

MODEL2_WEIGHTS = "yolo11n.pt"                   # pretrained COCO model, auto-downloaded once
BICYCLE_CONF = 0.35
M1_CONF = 0.20                                  # same as the live script
IMGSZ = 640
FALLBACK_IDS = {"driver": 1, "helmet": 2, "no_helmet": 3}   # ids used by the live script

NO_HELMET_CONFIRM_FRAMES = 3
HELMET_CONFIRM_FRAMES = 2
HISTORY_SIZE = 8

GREEN, RED, ORANGE, CYAN = (0, 255, 0), (0, 0, 255), (0, 165, 255), (200, 200, 0)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model2(weights):
    from ultralytics import YOLO
    if Path(weights).exists():
        return YOLO(weights)
    target = WEIGHTS_DIR / weights                # keep the download next to Model 2
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(target))
        except Exception as exc:
            print(f"[Model 2] could not download into {WEIGHTS_DIR}: {exc}")
    return YOLO(str(target)) if target.exists() else YOLO(weights)


class BicycleDetector:
    """MODEL 2: finds bicycles only."""

    def __init__(self, weights, conf, device, model=None):    # `model` = test hook
        self.model = model if model is not None else load_model2(weights)
        self.conf, self.device = conf, device
        self.ids = [int(i) for i, n in self.model.names.items() if str(n).lower() == "bicycle"]
        if not self.ids:
            raise RuntimeError(f"Model 2 has no 'bicycle' class. Classes: {dict(self.model.names)}")
        print(f"[Model 2] {weights} device={device} conf={conf} bicycle class ids={self.ids}")

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=IMGSZ, device=self.device,
                               classes=self.ids, verbose=False)[0]
        out = []
        for b in (r.boxes or []):
            out.append((tuple(float(v) for v in b.xyxy[0].cpu().numpy()), float(b.conf[0])))
        return out                                  # [(box, conf), ...]


def resolve_model1_classes(names):
    """driver / helmet / no-helmet class ids, read from Model 1's own class names."""
    roles = {"driver": set(), "helmet": set(), "no_helmet": set()}
    for cid, raw in names.items():
        n = re.sub(r"[^a-z0-9]+", " ", str(raw).lower()).strip()
        if "helmet" in n:
            t = n.split()
            bad = n.startswith("no") or "without" in t or "not" in t or "missing" in t or n.endswith("less")
            roles["no_helmet" if bad else "helmet"].add(int(cid))
        elif "driver" in n or "rider" in n or "motorcyclist" in n:
            roles["driver"].add(int(cid))
    fallback = False
    if len(names) == 4 and not all(roles.values()):
        for k, cid in FALLBACK_IDS.items():
            roles[k] = roles[k] or {cid}
        fallback = True
    return roles, fallback


class Model1Helmet:
    """MODEL 1: your best.pt, loaded from a local path. Same track() call as the live script."""

    def __init__(self, path, device, model=None):              # `model` = test hook
        if model is None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"\nModel 1 weights not found: {path.resolve()}\n"
                    "Use --model1 D:\\path\\to\\best.pt, or set HELMET_MODEL_PATH, "
                    "or copy best.pt to models\\best.pt\n")
            from ultralytics import YOLO
            model = YOLO(str(path))
        self.model, self.device = model, device
        names = {int(k): v for k, v in self.model.names.items()}
        self.roles, fallback = resolve_model1_classes(names)
        print(f"[Model 1] device={device} class names: {names}")
        print(f"[Model 1] resolved: driver={sorted(self.roles['driver'])} "
              f"helmet={sorted(self.roles['helmet'])} no_helmet={sorted(self.roles['no_helmet'])}")
        if fallback:
            print("[Model 1] WARNING: names not recognised, using the live script's ids "
                  "(1=driver, 2=helmet, 3=no-helmet). Check they are right.")
        missing = [k for k, v in self.roles.items() if not v]
        if missing:
            raise RuntimeError(f"Model 1 has no class for {missing}. Class names: {names}")

    def detect(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=M1_CONF,
                             imgsz=IMGSZ, device=self.device, verbose=False)[0]
        riders, heads = [], []                      # riders: (track_id, box)  heads: (label, box, conf)
        for b in (r.boxes or []):
            cid, conf = int(b.cls[0]), float(b.conf[0])
            box = tuple(float(v) for v in b.xyxy[0].cpu().numpy())
            if cid in self.roles["driver"]:
                if b.id is not None:                # riders need a track id, as in the live script
                    riders.append((int(b.id[0]), box))
            elif cid in self.roles["helmet"]:
                heads.append(("helmet", box, conf))
            elif cid in self.roles["no_helmet"]:
                heads.append(("no-helmet", box, conf))
        return riders, heads


# ----------------------------------------------------------------------------
# The one new rule: is this rider on a bicycle?
# ----------------------------------------------------------------------------

def rides_bicycle(rider, bicycles):
    """True if the rider box sits on top of a detected bicycle box."""
    rx1, ry1, rx2, ry2 = rider
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 0 or rh <= 0:
        return False
    for (vx1, vy1, vx2, vy2), _ in bicycles:
        vw, vh = vx2 - vx1, vy2 - vy1
        if vw <= 0 or vh <= 0:
            continue
        inter = max(0, min(rx2, vx2) - max(rx1, vx1)) * max(0, min(ry2, vy2) - max(ry1, vy1))
        rcx = (rx1 + rx2) / 2
        if (inter / (rw * rh) >= 0.25                              # rider overlaps the bicycle
                and vx1 - 0.15 * vw <= rcx <= vx2 + 0.15 * vw      # centred over it
                and vy1 + 0.25 * vh <= ry2 <= vy2 + 0.25 * vh      # feet within its height
                and 0.5 <= rh / vh <= 3.5):                        # plausible size
            return True
    return False


# ----------------------------------------------------------------------------
# Helmet logic (same rules as the live script) + the bicycle skip
# ----------------------------------------------------------------------------

class HelmetMonitor:
    def __init__(self, bicycle_detector, helmet_model):
        self.model2, self.model1 = bicycle_detector, helmet_model
        self.history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
        self.fined, self.helmet_ids = set(), set()

    def process(self, frame):
        bicycles = self.model2.detect(frame)                  # Model 2 first
        riders, heads = self.model1.detect(frame)             # then Model 1
        views = []
        for tid, box in riders:
            if rides_bicycle(box, bicycles):                  # bicycle: no helmet detection
                self.history.pop(tid, None)
                views.append((tid, box, "HELMET SKIPPED (BICYCLE)", CYAN))
                continue
            views.append((tid, box) + self._vote(tid, box, heads))
        return bicycles, views

    def _vote(self, tid, box, heads):
        def inside(h):
            cx, cy = (h[1][0] + h[1][2]) / 2, (h[1][1] + h[1][3]) / 2
            return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

        mine = [h for h in heads if inside(h)]
        nh = [h for h in mine if h[0] == "no-helmet"]
        hh = [h for h in mine if h[0] == "helmet"]
        self.history[tid].append("no-helmet" if nh else "helmet" if hh else "unknown")
        nh_votes = sum(s == "no-helmet" for s in self.history[tid])
        h_votes = sum(s == "helmet" for s in self.history[tid])

        if tid in self.fined:
            return "NO HELMET - COUNTED", RED
        if nh_votes >= NO_HELMET_CONFIRM_FRAMES:
            self.fined.add(tid)
            self.helmet_ids.discard(tid)
            print(f"VIOLATION | rider #{tid} | NO HELMET")
            return "NO HELMET", RED
        if h_votes >= HELMET_CONFIRM_FRAMES and h_votes > nh_votes:
            if tid not in self.helmet_ids:
                self.helmet_ids.add(tid)
                print(f"HELMET    | rider #{tid}")
            return "HELMET", GREEN
        return "CHECKING", ORANGE


# ----------------------------------------------------------------------------
# Drawing + webcam loop
# ----------------------------------------------------------------------------

def label_above(frame, text, x, y, color, scale=0.65, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    top = y - th - 12 if y - th - 12 >= 0 else y + 2
    cv2.rectangle(frame, (x, top), (min(frame.shape[1] - 1, x + tw + 10), top + th + 10), color, -1)
    cv2.putText(frame, text, (x + 5, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255) if color == RED else (0, 0, 0), thick, cv2.LINE_AA)


def draw(frame, bicycles, views, monitor, fps):
    for box, conf in bicycles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2)
        label_above(frame, f"Bicycle {conf:.2f}", x1, y1, CYAN, 0.5, 1)
    for tid, box, status, color in views:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        label_above(frame, f"RIDER #{tid}: {status}", x1, y1, color)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
    cv2.putText(frame, f"HELMET RIDERS: {len(monitor.helmet_ids)}   NO HELMET: {len(monitor.fined)}"
                       f"   FPS {fps:.1f}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description="Helmet detection, bicycles skipped")
    p.add_argument("--model1", nargs="?", const="", default=None,
                   help="path to Model 1 best.pt (default: env HELMET_MODEL_PATH, else models/best.pt)")
    p.add_argument("--weights", default=MODEL2_WEIGHTS, help="Model 2 (bicycle) weights")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="video file instead of the webcam")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    m1_path = args.model1 or os.environ.get("HELMET_MODEL_PATH") or DEFAULT_MODEL1

    monitor = HelmetMonitor(BicycleDetector(args.weights, BICYCLE_CONF, device),
                            Model1Helmet(m1_path, device))

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:                                            # DirectShow = faster webcam open on Windows
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" \
            else cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open {args.video or 'camera ' + str(args.camera)}. "
              "Try another --camera index and close apps that use the webcam.")
        return 1

    print("Running. Press Q in the video window to quit.")
    fps, prev = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed / end of video.")
                break
            bicycles, views = monitor.process(frame)
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-3) if fps == 0 else 0.9 * fps + 0.1 / max(now - prev, 1e-3)
            prev = now
            draw(frame, bicycles, views, monitor, fps)
            cv2.imshow("HELMET DETECTION (bicycles skipped)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stopped. Helmet riders: {len(monitor.helmet_ids)} | "
              f"No-helmet violations: {len(monitor.fined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())