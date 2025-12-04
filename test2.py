#!/usr/bin/env python3
"""
blur_violent_xclip.py

Detect violent segments in a video using X-CLIP (video-text similarity),
then blur those segments with a Gaussian blur.
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from transformers import AutoTokenizer, XCLIPModel


# ---------------- USER PATHS ----------------
INPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1.mp4"
OUTPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1_blurred_xclip.mp4"
# --------------------------------------------

# -------- X-CLIP & detection settings ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text tokenizer for X-CLIP
tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

# X-CLIP video-text model
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(DEVICE)
model.eval()

WINDOW = 8        # frames per clip
THRESHOLD = 0.8
TEXT_PROMPTS = ["violent fight with blood and gore", "safe normal scene"]
BLUR_KERNEL = (55, 55)
RESIZE_DIM = 224

# --------------------------------------------

print(f"Using device: {DEVICE}")


def detect_violent_segments_xclip(video_path, window=WINDOW, threshold=THRESHOLD):
    """
    Read video frames, group them into windows, run X-CLIP zero-shot
    video-text matching, and return [(start_sec, end_sec), ...] for violent segments.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    violent_flags = []

    frame_buffer_rgb = []
    frame_idx = 0

    pbar = tqdm(desc="Processing frames (X-CLIP detection)")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize & convert BGR -> RGB
        frame_resized = cv2.resize(frame_bgr, (RESIZE_DIM, RESIZE_DIM))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        frame_buffer_rgb.append(frame_rgb)
        frame_idx += 1
        pbar.update(1)

        # If we have a full window of frames, treat them as a short video clip
        if len(frame_buffer_rgb) == window:
            is_violent = classify_window_with_xclip(frame_buffer_rgb, threshold)
            violent_flags.extend([is_violent] * window)
            frame_buffer_rgb.clear()

    cap.release()
    pbar.close()

    # Handle leftover frames (if total frames not divisible by window size)
    if frame_buffer_rgb:
        is_violent = classify_window_with_xclip(frame_buffer_rgb, threshold)
        violent_flags.extend([is_violent] * len(frame_buffer_rgb))

    # Convert frame-wise flags -> continuous segments in seconds
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
        segments.append((start, len(violent_flags) / fps))

    return segments

def classify_window_with_xclip(frames_rgb, threshold):
    """
    Given a list of RGB frames (np.array HxWx3, already resized to 224x224),
    normalize them like CLIP, ensure we have exactly num_frames frames,
    then run X-CLIP with TEXT_PROMPTS.

    Returns True if P(violent) > threshold, else False.
    """
    # Make a copy so we don't modify the original list
    frames = list(frames_rgb)

    # X-CLIP expects a fixed number of frames (defaults to 8 for this checkpoint)
    try:
        num_frames = model.config.vision_config.num_frames
    except AttributeError:
        num_frames = 8  # fallback

    T = len(frames)

    # If we have fewer than num_frames, pad by repeating the last frame
    if T < num_frames:
        if T == 0:
            raise ValueError("No frames provided to classify_window_with_xclip")
        frames += [frames[-1]] * (num_frames - T)
    # If we have more than num_frames, subsample uniformly down to num_frames
    elif T > num_frames:
        indices = np.linspace(0, T - 1, num_frames).astype(int)
        frames = [frames[i] for i in indices]

    # Now we have exactly num_frames frames
    video = np.stack(frames).astype(np.float32) / 255.0  # (num_frames, 224, 224, 3)

    # CLIP normalization (same as CLIPImageProcessor)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 1, 3)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 1, 3)

    video = (video - mean) / std

    # (T, H, W, C) -> (T, C, H, W)
    video = np.transpose(video, (0, 3, 1, 2))  # (num_frames, 3, 224, 224)

    # Add batch dimension: (1, T, 3, 224, 224)
    pixel_values = torch.from_numpy(video).unsqueeze(0).to(DEVICE)

    # --- Text encoding ---
    text_inputs = tokenizer(
        TEXT_PROMPTS,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # --- Forward pass ---
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            **text_inputs
        )
        logits_per_video = outputs.logits_per_video  # shape: (1, 2)

    probs = logits_per_video.softmax(dim=1)  # softmax over the 2 prompts
    violent_prob = probs[0, 0].item()        # index 0 = "violent fight with blood and gore"

    return violent_prob > threshold




def apply_blur_with_moviepy(input_path, output_path, segments, blur_kernel=BLUR_KERNEL):
    clip = VideoFileClip(input_path)
    duration = clip.duration

    # Clamp segments to video duration and discard invalid ones
    segments_clamped = [
        (max(0, s), min(duration, e)) for s, e in segments if e > 0 and s < duration
    ]

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

        # Convert to uint8 if needed (MoviePy may give float [0,1])
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame_u8 = (frame * 255).astype(np.uint8)
            else:
                frame_u8 = frame.astype(np.uint8)
        else:
            frame_u8 = frame

        blurred = cv2.GaussianBlur(frame_u8, blur_kernel, 0)
        return blurred if frame.dtype == np.uint8 else blurred.astype(np.float32) / 255.0

    processed = clip.fl(lambda gf, t: process_frame(gf, t), apply_to=["video"])
    print(f"Exporting blurred video to: {output_path}")
    processed.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise SystemExit(f"Input file not found: {INPUT_VIDEO}")

    print("Step 1 — Detecting violent segments with X-CLIP...")
    segments = detect_violent_segments_xclip(INPUT_VIDEO)
    if segments:
        print("Detected violent segments (seconds):")
        for s, e in segments:
            print(f"  {s:.2f} -> {e:.2f} (duration {(e - s):.2f}s)")
    else:
        print("No violent segments detected.")

    print("Step 2 — Applying Gaussian blur to detected segments...")
    apply_blur_with_moviepy(INPUT_VIDEO, OUTPUT_VIDEO, segments)
    print("✅ Done! Output saved to:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
