import cv2
import torch
import logging
from ultralytics import RTDETR
from pathlib import Path

from .blur import blur_region


# ===============================
# LOGGING CONFIG
# ===============================
logger = logging.getLogger("RTDETR")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s][RTDETR] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).parent
RTDETR_WEIGHTS = BASE_DIR / "rtdetr_train.pt"
DETECTION_CONF = 0.40


# ===============================
# LOAD MODEL (ONCE)
# ===============================
_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading RT-DETR model on {_device}")
_detector = RTDETR(RTDETR_WEIGHTS)
logger.info("RT-DETR model loaded successfully")


# ===============================
# DETECTOR
# ===============================
def run_rtdetr_detector(video_path: str, violent_clips: list):
    """
    Runs RT-DETR on frames selected by CLIP.

    Returns:
    {
        "blurred_frames": {
            frame_idx: frame (np.ndarray)
        },
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

    logger.info(f"Starting RT-DETR detection on video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        logger.warning("FPS not detected. Falling back to 30 FPS.")

    blurred_frames = {}
    detections_out = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps

        # ---- Gate using CLIP windows ----
        if not any(
            clip["start"] <= time_sec <= clip["end"]
            for clip in violent_clips
        ):
            frame_idx += 1
            continue

        # ---- RT-DETR inference ----
        results = _detector.predict(
            source=frame,
            conf=DETECTION_CONF,
            verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            logger.info(
                f"Frame {frame_idx} | "
                f"{len(results.boxes)} violent regions detected"
            )

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Blur region
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
