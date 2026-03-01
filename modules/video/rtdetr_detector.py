import cv2
import torch
import logging
from ultralytics import RTDETR
from pathlib import Path

from config import RTDETR_CONF_THRESHOLD
from .blur import blur_region

# ===============================
# LOGGING
# ===============================
logger = logging.getLogger("RTDETR")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s][RT-DETR] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "models" / "rtdetr_train.pt"

_device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading RT-DETR model on {_device}")
_detector = RTDETR(WEIGHTS_PATH)
logger.info("RT-DETR model loaded successfully")

# ===============================
# DETECTOR
# ===============================
def run_rtdetr_detector(video_path: str, violent_clips: list):
    """
    Runs RT-DETR only inside CLIP-flagged windows.

    Returns:
    {
        "blurred_frames": { frame_idx: frame },
        "detections": [
            {
                "frame": int,
                "time": float,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    blurred_frames = {}
    detections_out = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps

        # Gate using CLIP segments
        if not any(
            clip["start"] <= time_sec <= clip["end"]
            for clip in violent_clips
        ):
            frame_idx += 1
            continue

        results = _detector.predict(
            source=frame,
            conf=RTDETR_CONF_THRESHOLD,
            verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                frame = blur_region(frame, (x1, y1, x2, y2))

                detections_out.append({
                    "frame": frame_idx,
                    "time": round(time_sec, 3),
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

            blurred_frames[frame_idx] = frame.copy()

        frame_idx += 1

    cap.release()

    logger.info(
        f"RT-DETR finished | "
        f"Blurred frames: {len(blurred_frames)} | "
        f"Detections: {len(detections_out)}"
    )

    return {
        "blurred_frames": blurred_frames,
        "detections": detections_out
    }