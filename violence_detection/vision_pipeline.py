import cv2

from .clip_filter import run_clip_filter
from .rtdetr_detector import run_rtdetr_detector


def run_vision_pipeline(video_path: str, output_path="output_blurred.mp4"):
    # -------------------------------
    # Stage 1: CLIP filtering
    # -------------------------------
    violent_clips = run_clip_filter(video_path)

    if not violent_clips:
        return {
            "status": "safe",
            "output_video": None
        }

    # -------------------------------
    # Stage 2: RT-DETR + Blur
    # -------------------------------
    blurred_frames = run_rtdetr_detector(
        video_path=video_path,
        violent_clips=violent_clips
    )

    # -------------------------------
    # Stage 3: Write Output Video
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

    return {
        "status": "violent",
        "output_video": output_path,
        "violent_segments": violent_clips
    }
