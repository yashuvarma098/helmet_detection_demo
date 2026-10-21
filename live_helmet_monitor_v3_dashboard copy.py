
import cv2
import numpy as np
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 0

# Put your trained checkpoint here:
MODEL_PATH = Path("models/best.pt")

CONFIDENCE = 0.20
IMGSZ = 640
DEVICE = "mps"

# Dataset classes:
# 0 = bicyclist
# 1 = driver
# 2 = helmet
# 3 = no-helmet
DRIVER_CLASS = 1
HELMET_CLASS = 2
NO_HELMET_CLASS = 3

NO_HELMET_CONFIRM_FRAMES = 3
HELMET_CONFIRM_FRAMES = 2
HISTORY_SIZE = 8

DEMO_FINE = 500

# Camera request. Continuity Camera may negotiate a nearby mode.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# UI
FINE_ANIMATION_STEP = 25

FINE_POPUP_SECONDS = 1.1

# Separate challan collection window
SHOW_COLLECTION_WINDOW = True
COLLECTION_WINDOW_NAME = "LIVE CHALLAN COLLECTION"
COLLECTION_WINDOW_WIDTH = 900
COLLECTION_WINDOW_HEIGHT = 540

# Save a local annotated video of every live session.
SAVE_SESSION_VIDEO = True

OUTPUT_DIR = Path("output")
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
FULL_FRAME_DIR = EVIDENCE_DIR / "full_frames"
RIDER_CROP_DIR = EVIDENCE_DIR / "rider_crops"
SESSION_DIR = OUTPUT_DIR / "sessions"

CSV_PATH = OUTPUT_DIR / "violations.csv"
STATE_PATH = OUTPUT_DIR / "state.json"
RESET_LOG_PATH = OUTPUT_DIR / "reset_history.txt"

for folder in [
    OUTPUT_DIR,
    EVIDENCE_DIR,
    FULL_FRAME_DIR,
    RIDER_CROP_DIR,
    SESSION_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def expanded_driver_crop(frame, box):
    """
    Keep a larger rider/bike crop for manual proof and future plate work.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    x1 = max(0, int(x1 - bw * 0.35))
    x2 = min(w, int(x2 + bw * 0.35))

    y1 = max(0, int(y1 - bh * 0.20))
    y2 = min(h, int(y2 + bh * 0.80))

    return frame[y1:y2, x1:x2]


def draw_rider_label(frame, status, track_id, box, color):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        4
    )

    line1 = f"RIDER #{track_id}"
    line2 = status

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.92
    thickness = 3

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)

    label_width = max(w1, w2) + 24
    label_height = h1 + h2 + 40
    top = max(0, y1 - label_height)

    cv2.rectangle(
        frame,
        (x1, top),
        (min(frame.shape[1] - 1, x1 + label_width), y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        line1,
        (x1 + 10, top + h1 + 7),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

    cv2.putText(
        frame,
        line2,
        (x1 + 10, top + h1 + h2 + 26),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )



def create_collection_dashboard(
    displayed_fine,
    total_collection,
    total_violations,
    current_session_violations,
    session_id,
    live_fps,
    fine_popup_frames_remaining,
    fine_popup_amount
):
    """
    Build a separate presentation-style money counter window.
    This window is purely visual and does not change tracking/fine logic.
    """
    w = COLLECTION_WINDOW_WIDTH
    h = COLLECTION_WINDOW_HEIGHT

    # Dark blue/purple vertical gradient.
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    top_color = np.array([42, 20, 20], dtype=np.float32)      # BGR
    bottom_color = np.array([85, 28, 72], dtype=np.float32)

    for y in range(h):
        t = y / max(h - 1, 1)
        color = ((1.0 - t) * top_color + t * bottom_color).astype(np.uint8)
        canvas[y, :] = color

    # Decorative glow blocks.
    cv2.circle(canvas, (810, 70), 160, (120, 60, 110), -1)
    cv2.circle(canvas, (70, 500), 140, (85, 85, 30), -1)

    # Main title.
    cv2.putText(
        canvas,
        "TOTAL CHALLAN COLLECTION",
        (52, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.18,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "LIVE HELMET ENFORCEMENT DEMO",
        (55, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (205, 215, 245),
        1,
        cv2.LINE_AA
    )

    # Main amount card.
    cv2.rectangle(canvas, (45, 135), (855, 335), (25, 25, 35), -1)
    cv2.rectangle(canvas, (45, 135), (855, 335), (0, 210, 255), 3)

    amount_text = f"Rs {displayed_fine:,}"
    cv2.putText(
        canvas,
        amount_text,
        (85, 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.75,
        (0, 230, 255),
        7,
        cv2.LINE_AA
    )

    # Small sub-label under amount.
    cv2.putText(
        canvas,
        f"Rs {DEMO_FINE} per confirmed no-helmet rider",
        (90, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 220, 235),
        2,
        cv2.LINE_AA
    )

    # Three stat cards.
    card_y1, card_y2 = 370, 475
    cards = [
        (45, 300, "NO HELMET", str(total_violations), (60, 60, 230)),
        (322, 577, "THIS SESSION", str(current_session_violations), (50, 170, 255)),
        (599, 855, "LIVE FPS", f"{live_fps:.1f}", (70, 210, 100)),
    ]

    for x1, x2, label, value, color in cards:
        cv2.rectangle(canvas, (x1, card_y1), (x2, card_y2), (30, 32, 44), -1)
        cv2.rectangle(canvas, (x1, card_y1), (x2, card_y2), color, 3)

        cv2.putText(
            canvas,
            label,
            (x1 + 18, card_y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 215, 225),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            value,
            (x1 + 18, card_y1 + 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            color,
            4,
            cv2.LINE_AA
        )

    # New challan flash.
    if fine_popup_frames_remaining > 0:
        cv2.rectangle(canvas, (285, 487), (615, 530), (0, 160, 245), -1)
        cv2.putText(
            canvas,
            f"NEW CHALLAN  + Rs {fine_popup_amount}",
            (310, 518),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            canvas,
            f"SESSION {session_id}   |   X: RESET MENU   Q: STOP",
            (250, 515),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 195, 210),
            1,
            cv2.LINE_AA
        )

    return canvas


def load_state():
    default = {
        "total_violations": 0,
        "total_collection": 0,
        "next_violation_sequence": 1
    }

    if not STATE_PATH.exists():
        return default

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))

        total_violations = int(data.get("total_violations", 0))
        total_collection = int(data.get("total_collection", 0))

        return {
            "total_violations": total_violations,
            "total_collection": total_collection,
            "next_violation_sequence": int(
                data.get("next_violation_sequence", total_violations + 1)
            )
        }

    except Exception as exc:
        print(f"Warning: could not read state.json: {exc}")
        return default


def save_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )

    tmp.replace(STATE_PATH)


def ensure_csv_header():
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        return

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "sequence",
            "session_id",
            "track_id",
            "timestamp",
            "status",
            "confidence",
            "fine",
            "cumulative_collection",
            "full_frame_file",
            "rider_crop_file",
            "manual_review",
            "plate_text"
        ])


def find_best_model():
    candidates = [
        MODEL_PATH,
        Path("runs/detect/helmet_training/helmet_v1/weights/best.pt"),
        Path("../runs/detect/helmet_training/helmet_v1/weights/best.pt"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None



def reset_counters_keep_history(next_sequence):
    """
    Reset visible violation and money counters to zero while keeping:
    - previous evidence images
    - violations.csv history
    - permanent violation sequence numbers

    This lets you start a fresh testing/day counter without overwriting proof.
    """
    reset_time = datetime.now().isoformat(timespec="seconds")

    save_state({
        "total_violations": 0,
        "total_collection": 0,
        "next_violation_sequence": next_sequence
    })

    with open(RESET_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"{reset_time} | counters reset to zero | "
            f"next_violation_sequence={next_sequence}\n"
        )


# ============================================================
# STARTUP
# ============================================================

model_file = find_best_model()

if model_file is None:
    raise FileNotFoundError(
        "\nCould not find best.pt.\n"
        "Copy your trained best.pt into: models/best.pt\n"
    )

print(f"Loading local model: {model_file}")
model = YOLO(str(model_file))

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}. "
        "Make sure iPhone Continuity Camera is active."
    )

camera_fps = cap.get(cv2.CAP_PROP_FPS)
if camera_fps <= 0:
    camera_fps = CAMERA_FPS

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    width, height = CAMERA_WIDTH, CAMERA_HEIGHT

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = None
session_video_path = SESSION_DIR / f"live_session_{session_id}.mp4"

if SAVE_SESSION_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(session_video_path),
        fourcc,
        camera_fps,
        (width, height)
    )

persistent_state = load_state()

total_violations = persistent_state["total_violations"]
total_collection = persistent_state["total_collection"]
violation_sequence = persistent_state["next_violation_sequence"]

displayed_fine = total_collection
fine_popup_frames_remaining = 0
fine_popup_amount = 0

history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

# IMPORTANT:
# Tracker IDs are valid only for this live session.
# This prevents repeat fines while the same tracked rider stays in the session.
fined_track_ids = set()
confirmed_helmet_track_ids = set()

ensure_csv_header()
csv_file = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)

print()
print("==============================================")
print("LOCAL LIVE HELMET MONITOR V3 + COLLECTION DASHBOARD")
print("==============================================")
print(f"Camera index: {CAMERA_INDEX}")
print(f"Camera size: {width}x{height}")
print(f"Saved previous violations: {total_violations}")
print(f"Saved previous collection: Rs {total_collection:,}")
print("Press Q in the video window to stop.")
print("Press X to open RESET confirmation. Then type RESET and press ENTER.")
print("Old evidence images and CSV history are kept.")
print("A separate challan collection window will open automatically.")
print()

previous_clock = time.time()

# Safe reset confirmation state.
# Nothing resets from a single key press.
reset_confirmation_mode = False
reset_input_buffer = ""
reset_message = ""
reset_message_frames = 0


# ============================================================
# LIVE LOOP
# ============================================================

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera frame read failed.")
            break

        raw_frame = frame.copy()

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )[0]

        drivers = []
        helmets = []
        no_helmets = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                if cls_id == DRIVER_CLASS and track_id is not None:
                    drivers.append({
                        "track_id": track_id,
                        "box": xyxy,
                        "confidence": confidence
                    })

                elif cls_id == HELMET_CLASS:
                    helmets.append({
                        "box": xyxy,
                        "confidence": confidence
                    })

                elif cls_id == NO_HELMET_CLASS:
                    no_helmets.append({
                        "box": xyxy,
                        "confidence": confidence
                    })

        for driver in drivers:
            track_id = driver["track_id"]
            driver_box = driver["box"]

            helmet_hits = []
            no_helmet_hits = []

            for helmet in helmets:
                if point_inside_box(
                    box_center(helmet["box"]),
                    driver_box
                ):
                    helmet_hits.append(helmet)

            for nh in no_helmets:
                if point_inside_box(
                    box_center(nh["box"]),
                    driver_box
                ):
                    no_helmet_hits.append(nh)

            frame_status = "unknown"
            best_conf = 0.0

            if no_helmet_hits:
                best = max(
                    no_helmet_hits,
                    key=lambda x: x["confidence"]
                )
                frame_status = "no-helmet"
                best_conf = best["confidence"]

            elif helmet_hits:
                best = max(
                    helmet_hits,
                    key=lambda x: x["confidence"]
                )
                frame_status = "helmet"
                best_conf = best["confidence"]

            history[track_id].append({
                "status": frame_status,
                "confidence": best_conf
            })

            nh_votes = sum(
                1
                for item in history[track_id]
                if item["status"] == "no-helmet"
            )

            helmet_votes = sum(
                1
                for item in history[track_id]
                if item["status"] == "helmet"
            )

            if track_id in fined_track_ids:
                final_status = "NO HELMET - COUNTED"
                color = (0, 0, 255)

            elif nh_votes >= NO_HELMET_CONFIRM_FRAMES:
                final_status = "NO HELMET"
                color = (0, 0, 255)

                current_sequence = violation_sequence

                fined_track_ids.add(track_id)
                confirmed_helmet_track_ids.discard(track_id)

                total_violations += 1
                total_collection += DEMO_FINE
                violation_sequence += 1

                fine_popup_amount = DEMO_FINE
                fine_popup_frames_remaining = max(
                    1,
                    int(camera_fps * FINE_POPUP_SECONDS)
                )

                timestamp = datetime.now().isoformat(timespec="seconds")

                full_filename = (
                    f"violation_{current_sequence:05d}"
                    f"_session_{session_id}"
                    f"_track_{track_id}_full.jpg"
                )

                crop_filename = (
                    f"violation_{current_sequence:05d}"
                    f"_session_{session_id}"
                    f"_track_{track_id}_rider.jpg"
                )

                full_path = FULL_FRAME_DIR / full_filename
                crop_path = RIDER_CROP_DIR / crop_filename

                # Save UNANNOTATED original frame as evidence.
                cv2.imwrite(str(full_path), raw_frame)

                rider_crop = expanded_driver_crop(
                    raw_frame,
                    driver_box
                )

                if rider_crop.size > 0:
                    cv2.imwrite(
                        str(crop_path),
                        rider_crop
                    )

                csv_writer.writerow([
                    current_sequence,
                    session_id,
                    track_id,
                    timestamp,
                    "no-helmet",
                    round(best_conf, 3),
                    DEMO_FINE,
                    total_collection,
                    str(full_path),
                    str(crop_path),
                    "pending",
                    ""
                ])

                csv_file.flush()

                save_state({
                    "total_violations": total_violations,
                    "total_collection": total_collection,
                    "next_violation_sequence": violation_sequence
                })

                print(
                    f"VIOLATION #{current_sequence} | "
                    f"Track {track_id} | "
                    f"NO HELMET | "
                    f"+Rs {DEMO_FINE} | "
                    f"TOTAL Rs {total_collection:,}"
                )

            elif (
                helmet_votes >= HELMET_CONFIRM_FRAMES
                and helmet_votes > nh_votes
            ):
                final_status = "HELMET"
                color = (0, 255, 0)
                confirmed_helmet_track_ids.add(track_id)

            else:
                final_status = "CHECKING"
                color = (0, 165, 255)

            draw_rider_label(
                frame,
                final_status,
                track_id,
                driver_box,
                color
            )

        # ----------------------------------------------------
        # Fine animation
        # ----------------------------------------------------
        target_fine = total_collection

        if displayed_fine < target_fine:
            displayed_fine += FINE_ANIMATION_STEP
            displayed_fine = min(
                displayed_fine,
                target_fine
            )

        # ----------------------------------------------------
        # Live FPS
        # ----------------------------------------------------
        now = time.time()
        live_fps = 1.0 / max(now - previous_clock, 0.001)
        previous_clock = now

        # ----------------------------------------------------
        # Dashboard
        # ----------------------------------------------------
        header_height = 135

        cv2.rectangle(
            frame,
            (0, 0),
            (width, header_height),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            f"HELMET RIDERS: {len(confirmed_helmet_track_ids)}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"NO HELMET: {total_violations}",
            (380, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (0, 0, 255),
            3
        )

        # Larger, bolder challan collection text.
        # FONT_HERSHEY_DUPLEX gives a fuller/cleaner display than SIMPLEX.
        cv2.putText(
            frame,
            f"DEMO FINE COLLECTED: Rs {displayed_fine:,}",
            (20, 98),
            cv2.FONT_HERSHEY_DUPLEX,
            1.38,
            (0, 215, 255),
            5,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"LIVE FPS: {live_fps:.1f}",
            (width - 245, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"SESSION {session_id}",
            (width - 330, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            "Q: STOP   X: RESET MENU (type RESET + ENTER)",
            (20, 128),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # +Rs500 pop-up
        # ----------------------------------------------------
        if fine_popup_frames_remaining > 0:
            popup = f"+ Rs {fine_popup_amount}"

            cv2.putText(
                frame,
                popup,
                (width // 2 - 130, header_height + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                (0, 215, 255),
                5
            )

            fine_popup_frames_remaining -= 1

        # ----------------------------------------------------
        # SAFE RESET CONFIRMATION OVERLAY
        # ----------------------------------------------------
        if reset_confirmation_mode:
            overlay = frame.copy()

            box_x1 = max(30, width // 2 - 380)
            box_x2 = min(width - 30, width // 2 + 380)
            box_y1 = max(header_height + 30, height // 2 - 145)
            box_y2 = min(height - 30, height // 2 + 145)

            # Dark modal.
            cv2.rectangle(
                overlay,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (20, 20, 25),
                -1
            )

            # Slight transparency.
            frame = cv2.addWeighted(overlay, 0.94, frame, 0.06, 0)

            cv2.rectangle(
                frame,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (0, 165, 255),
                4
            )

            cv2.putText(
                frame,
                "RESET COLLECTION COUNTERS?",
                (box_x1 + 35, box_y1 + 58),
                cv2.FONT_HERSHEY_DUPLEX,
                0.95,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                "Type RESET and press ENTER",
                (box_x1 + 35, box_y1 + 108),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (0, 215, 255),
                2,
                cv2.LINE_AA
            )

            typed_text = reset_input_buffer if reset_input_buffer else "_"

            cv2.rectangle(
                frame,
                (box_x1 + 35, box_y1 + 132),
                (box_x2 - 35, box_y1 + 196),
                (45, 45, 55),
                -1
            )

            cv2.putText(
                frame,
                typed_text,
                (box_x1 + 55, box_y1 + 178),
                cv2.FONT_HERSHEY_DUPLEX,
                1.12,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                "ESC = CANCEL   |   Old evidence/CSV will be kept",
                (box_x1 + 35, box_y2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        elif reset_message_frames > 0:
            cv2.rectangle(
                frame,
                (width // 2 - 230, header_height + 15),
                (width // 2 + 230, header_height + 70),
                (25, 25, 30),
                -1
            )

            cv2.putText(
                frame,
                reset_message,
                (width // 2 - 195, header_height + 53),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (0, 215, 255),
                2,
                cv2.LINE_AA
            )

            reset_message_frames -= 1

        if writer is not None:
            writer.write(frame)

        # ----------------------------------------------------
        # Separate challan collection presentation window
        # ----------------------------------------------------
        if SHOW_COLLECTION_WINDOW:
            collection_dashboard = create_collection_dashboard(
                displayed_fine=displayed_fine,
                total_collection=total_collection,
                total_violations=total_violations,
                current_session_violations=len(fined_track_ids),
                session_id=session_id,
                live_fps=live_fps,
                fine_popup_frames_remaining=fine_popup_frames_remaining,
                fine_popup_amount=fine_popup_amount
            )

            cv2.imshow(
                COLLECTION_WINDOW_NAME,
                collection_dashboard
            )

        cv2.imshow(
            "LOCAL LIVE HELMET MONITOR V3",
            frame
        )

        # ----------------------------------------------------
        # SAFE KEYBOARD CONTROLS
        # ----------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if not reset_confirmation_mode:
            if key == ord("q"):
                break

            # X only opens the reset confirmation screen.
            # It NEVER resets anything by itself.
            if key == ord("x"):
                reset_confirmation_mode = True
                reset_input_buffer = ""
                reset_message = ""
                reset_message_frames = 0

        else:
            # ESC cancels without changing any counter/history.
            if key == 27:
                reset_confirmation_mode = False
                reset_input_buffer = ""
                reset_message = "RESET CANCELLED"
                reset_message_frames = 45

            # ENTER confirms only when the exact word RESET was typed.
            elif key in (10, 13):
                if reset_input_buffer.upper() == "RESET":
                    # Fresh counting period. Old evidence and CSV are preserved.
                    total_violations = 0
                    total_collection = 0
                    displayed_fine = 0

                    fined_track_ids.clear()
                    confirmed_helmet_track_ids.clear()
                    history.clear()

                    fine_popup_amount = 0
                    fine_popup_frames_remaining = 0

                    # Keep permanent violation numbering so evidence is never overwritten.
                    reset_counters_keep_history(violation_sequence)

                    reset_confirmation_mode = False
                    reset_input_buffer = ""
                    reset_message = "COUNTERS RESET TO ZERO"
                    reset_message_frames = 90

                    print()
                    print("==============================================")
                    print("FRESH SESSION COUNTERS STARTED")
                    print("NO HELMET: 0")
                    print("DEMO FINE COLLECTED: Rs 0")
                    print("Previous evidence and CSV history preserved.")
                    print(f"Next permanent violation #: {violation_sequence}")
                    print("==============================================")
                    print()

                else:
                    reset_message = "TYPE EXACTLY: RESET"
                    reset_message_frames = 60
                    reset_input_buffer = ""

            # BACKSPACE
            elif key in (8, 127):
                reset_input_buffer = reset_input_buffer[:-1]

            # Accept letters only while reset confirmation is open.
            elif key != 255 and 32 <= key <= 126:
                char = chr(key).upper()

                if char.isalpha() and len(reset_input_buffer) < 10:
                    reset_input_buffer += char

finally:
    cap.release()

    if writer is not None:
        writer.release()

    csv_file.close()
    cv2.destroyAllWindows()

    print()
    print("Live session stopped.")
    print(f"Violations this session: {len(fined_track_ids)}")
    print(f"Cumulative violations: {total_violations}")
    print(f"Cumulative collection: Rs {total_collection:,}")
    print(f"Evidence: {EVIDENCE_DIR}")
    print(f"CSV: {CSV_PATH}")
    print(f"State: {STATE_PATH}")

    if SAVE_SESSION_VIDEO:
        print(f"Session video: {session_video_path}")
