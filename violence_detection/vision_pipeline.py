import cv2

from .clip_filter import run_clip_filter
from .rtdetr_detector import run_rtdetr_detector


def run_vision_pipeline(video_path: str, output_path="output_blurred.mp4"):
    # -------------------------------
    # Stage 1: CLIP filtering (GATE)
    # -------------------------------
    clip_segments = run_clip_filter(video_path)

    if not clip_segments:
        return {
            "status": "safe",
            "output_video": None,
            "video_max_confidence": 0.0,
            "violent_segments": [],
            "rtdetr_detections": []
        }

    # -------------------------------
    # Stage 2: RT-DETR + Blur (TRUTH)
    # -------------------------------
    # Expected to return:
    # {
    #   "blurred_frames": {frame_idx: frame},
    #   "detections": [
    #       {
    #           "frame": int,
    #           "confidence": float,
    #           "bbox": [x1,y1,x2,y2]
    #       }
    #   ]
    # }
    rtdetr_output = run_rtdetr_detector(
        video_path=video_path,
        violent_clips=clip_segments
    )

    blurred_frames = rtdetr_output["blurred_frames"]
    detections = rtdetr_output["detections"]

    # -------------------------------
    # Stage 3: Compute VIDEO confidence
    # -------------------------------
    if detections:
        video_max_confidence = max(d["confidence"] for d in detections)
    else:
        video_max_confidence = 0.0

    # -------------------------------
    # Stage 4: Write Output Video
    # -------------------------------
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    # Stage 5: Return structured output
    # -------------------------------
    return {
        "status": "violent",
        "output_video": output_path,

        # CLIP segments (for timestamps)
        "violent_segments": clip_segments,

        # RT-DETR truth signal
        "rtdetr_detections": detections,

        # THIS is what fusion should use
        "video_max_confidence": round(video_max_confidence, 3)
    }
