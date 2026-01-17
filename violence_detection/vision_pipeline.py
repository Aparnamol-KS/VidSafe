import cv2

from .clip_filter import run_clip_filter
from .rtdetr_detector import run_rtdetr_detector


# ==================================================
# HELPER: BUILD SEGMENTS FROM RT-DETR DETECTIONS
# ==================================================
def build_segments_from_detections(
    detections: list,
    fps: float,
    gap_frames: int = 10
):
    """
    Merge frame-level RT-DETR detections into time segments.

    Args:
        detections: list of {
            "frame": int,
            "confidence": float,
            ...
        }
        fps: video FPS
        gap_frames: max allowed gap between frames to be same segment

    Returns:
        [
            {
                "start": float,
                "end": float,
                "confidence": float
            }
        ]
    """

    if not detections:
        return []

    # Sort detections by frame index
    detections = sorted(detections, key=lambda d: d["frame"])

    segments = []

    start_frame = detections[0]["frame"]
    end_frame = start_frame
    max_conf = detections[0]["confidence"]

    for det in detections[1:]:
        frame = det["frame"]
        conf = det["confidence"]

        if frame <= end_frame + gap_frames:
            end_frame = frame
            max_conf = max(max_conf, conf)
        else:
            segments.append({
                "start": round(start_frame / fps, 3),
                "end": round(end_frame / fps, 3),
                "confidence": round(max_conf, 3)
            })

            start_frame = frame
            end_frame = frame
            max_conf = conf

    # Add final segment
    segments.append({
        "start": round(start_frame / fps, 3),
        "end": round(end_frame / fps, 3),
        "confidence": round(max_conf, 3)
    })

    return segments


# ==================================================
# MAIN VISION PIPELINE
# ==================================================
def run_vision_pipeline(video_path: str, output_path="output_blurred.mp4"):
    """
    Vision moderation pipeline.

    Returns:
    {
        "status": "safe" | "violent",
        "output_video": str | None,
        "violent_segments": [...],        # RT-DETR based segments
        "rtdetr_detections": [...],        # frame-level detections
        "video_max_confidence": float
    }
    """

    # -------------------------------
    # Stage 1: CLIP filtering (GATE)
    # -------------------------------
    clip_segments = run_clip_filter(video_path)

    if not clip_segments:
        return {
            "status": "safe",
            "output_video": None,
            "violent_segments": [],
            "rtdetr_detections": [],
            "video_max_confidence": 0.0
        }

    # -------------------------------
    # Stage 2: RT-DETR + Blur (TRUTH)
    # -------------------------------
    rtdetr_output = run_rtdetr_detector(
        video_path=video_path,
        violent_clips=clip_segments
    )

    blurred_frames = rtdetr_output["blurred_frames"]
    detections = rtdetr_output["detections"]

    # -------------------------------
    # Stage 3: Read video metadata
    # -------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------------------------------
    # Stage 4: Build RT-DETR segments
    # -------------------------------
    video_segments = build_segments_from_detections(
        detections=detections,
        fps=fps
    )

    if detections:
        video_max_confidence = max(d["confidence"] for d in detections)
    else:
        video_max_confidence = 0.0

    # -------------------------------
    # Stage 5: Write output video
    # -------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

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

    # -------------------------------
    # Stage 6: Final structured output
    # -------------------------------
    return {
        "status": "violent",
        "output_video": output_path,

        # ✅ RT-DETR BASED SEGMENTS (FIXES PDF)
        "violent_segments": video_segments,

        # Frame-level detections (audit/debug)
        "rtdetr_detections": detections,

        # Used for fusion severity
        "video_max_confidence": round(video_max_confidence, 3)
    }
