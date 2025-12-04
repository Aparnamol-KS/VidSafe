#!/usr/bin/env python3
"""
blur_violent_clip_semantic.py

Detect violent segments in a video using CLIP (image-text similarity),
then blur those segments with a Gaussian blur.
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import CLIPModel, CLIPProcessor

# ---------------- USER PATHS ----------------
INPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1.mp4"
OUTPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1_blurred_clip_semantic.mp4"
# --------------------------------------------

# -------- CLIP & detection settings ----------
DEVICE = "cpu"  # change to "cuda" if GPU available
WINDOW = 16      # number of frames per segment
THRESHOLD = 0.8  # probability threshold for violence
TEXT_PROMPTS = ["violent fight with blood and gore", "safe normal scene"]
BLUR_KERNEL = (55, 55)
RESIZE_DIM = 224  # CLIP image size
# --------------------------------------------

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


def detect_violent_segments(video_path, window=WINDOW, threshold=THRESHOLD):
    """
    Read video frames, process in windows with CLIP, return list of (start_sec, end_sec) violent segments
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    violent_flags = []

    frame_buffer = []
    orig_frames = []

    frame_idx = 0
    pbar = tqdm(desc="Processing frames (CLIP detection)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM)), cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)
        orig_frames.append(frame)
        frame_idx += 1

        if len(frame_buffer) == window:
            # CLIP inputs
            inputs = processor(text=TEXT_PROMPTS, images=frame_buffer, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image
                probs = logits.softmax(dim=1)
            # violent if average probability > threshold
            is_violent = probs[:, 0].mean().item() > threshold
            violent_flags.extend([is_violent]*window)
            frame_buffer.clear()
        pbar.update(1)

    cap.release()
    pbar.close()

    # Handle leftover frames
    if frame_buffer:
        is_violent = probs[:, 0].mean().item() > threshold
        violent_flags.extend([is_violent]*len(frame_buffer))

    # Convert frame flags -> segments in seconds
    segments = []
    start = None
    for i, flag in enumerate(violent_flags):
        t = i / fps
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            segments.append((start, t))
            start = None
    if start is not None:
        segments.append((start, len(violent_flags)/fps))

    return segments


def apply_blur_with_moviepy(input_path, output_path, segments, blur_kernel=BLUR_KERNEL):
    clip = VideoFileClip(input_path)
    duration = clip.duration

    segments_clamped = [(max(0, s), min(duration, e)) for s, e in segments if e > 0 and s < duration]
    if not segments_clamped:
        print("No violent segments detected. Copying original.")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()
        return

    def process_frame(gf, t):
        frame = gf(t)
        inside = any(s <= t < e for s, e in segments_clamped)
        if not inside:
            return frame
        # convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame_u8 = (frame*255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        else:
            frame_u8 = frame
        blurred = cv2.GaussianBlur(frame_u8, blur_kernel, 0)
        return blurred if frame.dtype == np.uint8 else blurred.astype(np.float32)/255.0

    processed = clip.fl(lambda gf, t: process_frame(gf, t), apply_to=["video"])
    print(f"Exporting blurred video to: {output_path}")
    processed.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise SystemExit(f"Input file not found: {INPUT_VIDEO}")

    print("Step 1 — Detecting violent segments with CLIP...")
    segments = detect_violent_segments(INPUT_VIDEO)
    if segments:
        print("Detected violent segments (seconds):")
        for s, e in segments:
            print(f"  {s:.2f} -> {e:.2f} (duration {(e-s):.2f}s)")
    else:
        print("No violent segments detected.")

    print("Step 2 — Applying Gaussian blur to detected segments...")
    apply_blur_with_moviepy(INPUT_VIDEO, OUTPUT_VIDEO, segments)
    print("✅ Done! Output saved to:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()