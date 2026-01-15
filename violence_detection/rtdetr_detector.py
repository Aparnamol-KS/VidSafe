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
    Returns:
    {
        frame_index: blurred_frame
    }
    """

    logger.info(f"Starting RT-DETR detection on video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        logger.warning("FPS could not be read. Falling back to 30 FPS.")

    blurred_frames = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps

        # Check if frame lies inside violent clip window
        if not any(
            clip["start"] <= time_sec <= clip["end"]
            for clip in violent_clips
        ):
            frame_idx += 1
            continue

        logger.debug(
            f"Frame {frame_idx} (t={time_sec:.2f}s) inside violent window"
        )

        detections = _detector.predict(
            source=frame,
            conf=DETECTION_CONF,
            verbose=False
        )[0]

        num_boxes = len(detections.boxes)
        if num_boxes > 0:
            logger.info(
                f"Frame {frame_idx} | {num_boxes} violent regions detected"
            )

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame = blur_region(frame, (x1, y1, x2, y2))
            logger.debug(
                f"Blur applied at frame {frame_idx} "
                f"bbox=({x1},{y1},{x2},{y2})"
            )

        if num_boxes > 0:
            blurred_frames[frame_idx] = frame.copy()

        frame_idx += 1

    cap.release()

    logger.info(
        f"RT-DETR finished. Total blurred frames: {len(blurred_frames)}"
    )

    return blurred_frames
