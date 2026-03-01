import cv2
from .clip_filter import run_clip_filter
from .rtdetr_detector import run_rtdetr_detector
from .segment_builder import build_segments_from_detections
from .temporal_filter import apply_temporal_consistency
from ...config import RTDETR_MIN_CONSECUTIVE_FRAMES


def run_video_pipeline(video_path: str, output_path: str):

    # -------------------------------
    # Stage 1: CLIP filtering
    # -------------------------------
    clip_segments = run_clip_filter(video_path)

    if not clip_segments:
        return {
            "status": "safe",
            "violent_segments": [],
            "rtdetr_detections": [],
            "video_max_confidence": 0.0
        }

    # -------------------------------
    # Stage 2: RT-DETR
    # -------------------------------
    rtdetr_output = run_rtdetr_detector(
        video_path=video_path,
        violent_clips=clip_segments
    )

    blurred_frames = rtdetr_output["blurred_frames"]
    detections = rtdetr_output["detections"]

    # -------------------------------
    # Stage 3: Temporal consistency
    # -------------------------------
    stable_detections = apply_temporal_consistency(
        detections,
        min_frames=RTDETR_MIN_CONSECUTIVE_FRAMES
    )

    # -------------------------------
    # Stage 4: Video metadata
    # -------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------------------------------
    # Stage 5: Build segments
    # -------------------------------
    segments = build_segments_from_detections(
        stable_detections,
        fps
    )

    video_max_confidence = (
        max(d["confidence"] for d in stable_detections)
        if stable_detections else 0.0
    )

    # -------------------------------
    # Stage 6: Write output
    # -------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in blurred_frames:
            frame = blurred_frames[frame_idx]

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "status": "violent",
        "violent_segments": segments,
        "rtdetr_detections": stable_detections,
        "video_max_confidence": round(video_max_confidence, 3)
    }