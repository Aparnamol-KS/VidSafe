#!/usr/bin/env python3
"""
blur_clip_rtdetr_video.py - IMPROVED VERSION

Key improvements:
1. Lower confidence threshold for better detection coverage
2. Expanded blur regions with padding for better coverage
3. Progressive blur strength based on box size
4. Feathered edges for smoother transitions
5. Optional tracking to maintain consistent blurring across frames
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import CLIPModel, CLIPProcessor
from ultralytics import RTDETR

# ---------------- USER PATHS ----------------
INPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1.mp4"
OUTPUT_VIDEO = r"C:\Users\alici\OneDrive\Desktop\Main_project\Mainproject\Video1_blurred_clip_rtdetr.mp4"
# --------------------------------------------

# -------- CLIP settings ----------
DEVICE = "cpu"  # "cuda" if you have GPU
WINDOW = 16      # number of frames per segment
THRESHOLD = 0.8  # probability threshold for violence
TEXT_PROMPTS = ["violent fight with blood and gore", "safe normal scene"]
RESIZE_DIM = 224  # CLIP image size
# ---------------------------------

# -------- RT-DETR settings (IMPROVED) ----------
RTDETR_MODEL_PATH = r"rtdetr-l.pt"
CONF_THRESH = 0.25                  # LOWERED for better detection coverage
IOU_THRESH = 0.45                   # NMS threshold
BLUR_MODE = "gaussian"              # "gaussian" or "pixelate"
CLASSES_TO_BLUR = None              # e.g. ["weapon","blood_region"], or None to blur all

# NEW: Blur enhancement parameters
BLUR_PADDING = 0.15                 # Expand detected boxes by 15% on each side
MIN_BLUR_KERNEL = 51                # Larger minimum kernel for stronger blur
MAX_BLUR_KERNEL = 99                # Maximum kernel size
FEATHER_PIXELS = 15                 # Pixels for edge feathering (smooth transitions)
PIXELATE_FACTOR = 8                 # Stronger pixelation (lower = more blur)

# Optional: Enable tracking for temporal consistency
USE_TRACKING = True                 # Track objects across frames
TRACK_BUFFER = 30                   # Frames to keep lost tracks
# -----------------------------------------------

print(f"Using device: {DEVICE}")

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load RT-DETR model
rtdetr_model = RTDETR(RTDETR_MODEL_PATH)
rtdetr_names = rtdetr_model.names
name_to_id = {v: k for k, v in rtdetr_names.items()}

if CLASSES_TO_BLUR:
    class_ids_to_blur = {name_to_id[c] for c in CLASSES_TO_BLUR if c in name_to_id}
else:
    class_ids_to_blur = None


# ---------------- Helper funcs ----------------

def odd(n: int) -> int:
    """Ensure n is odd (required for Gaussian kernel)."""
    return n if n % 2 == 1 else n + 1


def expand_box(x1, y1, x2, y2, W, H, padding=BLUR_PADDING):
    """Expand bounding box by padding percentage."""
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1_new = max(0, x1 - pad_w)
    y1_new = max(0, y1 - pad_h)
    x2_new = min(W, x2 + pad_w)
    y2_new = min(H, y2 + pad_h)
    
    return x1_new, y1_new, x2_new, y2_new


def create_feather_mask(h, w, feather_pixels=FEATHER_PIXELS):
    """Create a mask with feathered edges for smooth blending."""
    if feather_pixels <= 0:
        return np.ones((h, w), dtype=np.float32)
    
    mask = np.ones((h, w), dtype=np.float32)
    
    # Create distance transform from edges
    for i in range(feather_pixels):
        alpha = (i + 1) / feather_pixels
        # Top edge
        if i < h:
            mask[i, :] = min(mask[i, :].min(), alpha)
        # Bottom edge
        if i < h:
            mask[h - 1 - i, :] = min(mask[h - 1 - i, :].min(), alpha)
        # Left edge
        if i < w:
            mask[:, i] = np.minimum(mask[:, i], alpha)
        # Right edge
        if i < w:
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)
    
    return mask


def blur_roi(roi, mode="gaussian", box_area=0, img_area=1):
    """
    Blur a region with adaptive kernel size based on detection size.
    Larger detections get stronger blur.
    """
    h, w = roi.shape[:2]
    
    if mode == "pixelate":
        small_w = max(1, w // PIXELATE_FACTOR)
        small_h = max(1, h // PIXELATE_FACTOR)
        tmp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # Adaptive kernel size based on box size
        area_ratio = box_area / img_area
        kernel_size = int(MIN_BLUR_KERNEL + (MAX_BLUR_KERNEL - MIN_BLUR_KERNEL) * area_ratio)
        kernel_size = odd(min(MAX_BLUR_KERNEL, max(MIN_BLUR_KERNEL, kernel_size)))
        
        # Apply stronger blur with multiple passes for large regions
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        if area_ratio > 0.1:  # Large detection, apply second pass
            blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
        
        return blurred


def detect_violent_segments(video_path, window=WINDOW, threshold=THRESHOLD):
    """
    Read video frames, process in windows with CLIP,
    return list of (start_sec, end_sec) violent segments.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    violent_flags = []
    frame_buffer = []
    frame_idx = 0
    
    pbar = tqdm(desc="Processing frames (CLIP detection)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM)), cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)
        frame_idx += 1

        if len(frame_buffer) == window:
            inputs = clip_processor(
                text=TEXT_PROMPTS,
                images=frame_buffer,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            with torch.no_grad():
                logits = clip_model(**inputs).logits_per_image
                probs = logits.softmax(dim=1)
            is_violent = probs[:, 0].mean().item() > threshold
            violent_flags.extend([is_violent] * window)
            frame_buffer.clear()
        pbar.update(1)

    cap.release()
    pbar.close()

    # Handle leftover frames
    if frame_buffer:
        inputs = clip_processor(
            text=TEXT_PROMPTS,
            images=frame_buffer,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image
            probs = logits.softmax(dim=1)
        is_violent = probs[:, 0].mean().item() > threshold
        violent_flags.extend([is_violent] * len(frame_buffer))

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
        segments.append((start, len(violent_flags) / fps))

    return segments


def blur_frame_with_rtdetr_rgb(frame_rgb, use_tracking=USE_TRACKING):
    """
    Given an RGB frame, run RT-DETR with improved detection and blur settings.
    Returns an RGB frame with properly blurred regions.
    """
    original_dtype = frame_rgb.dtype
    if original_dtype != np.uint8:
        if frame_rgb.max() <= 1.0:
            frame_u8 = (frame_rgb * 255).astype(np.uint8)
        else:
            frame_u8 = frame_rgb.astype(np.uint8)
    else:
        frame_u8 = frame_rgb.copy()

    bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    img_area = H * W

    # Run RT-DETR with tracking if enabled
    if use_tracking:
        results = rtdetr_model.track(
            source=bgr, 
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
    else:
        results = rtdetr_model.predict(
            source=bgr, 
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            verbose=False
        )
    
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        out_bgr = bgr
    else:
        r = results[0]
        out_bgr = bgr.copy()
        
        # Create a mask for all blurred regions
        blur_mask = np.zeros((H, W), dtype=np.float32)
        blurred_full = np.zeros_like(bgr, dtype=np.float32)
        
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Filter classes if specified
            if class_ids_to_blur is not None and cls_id not in class_ids_to_blur:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Expand the box for better coverage
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H)
            
            if x2 <= x1 or y2 <= y1:
                continue

            roi = out_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Calculate box area for adaptive blur
            box_area = (x2 - x1) * (y2 - y1)
            
            # Blur the ROI with adaptive strength
            roi_blurred = blur_roi(roi, BLUR_MODE, box_area, img_area)
            
            # Create feathered mask for this region
            feather_mask = create_feather_mask(y2 - y1, x2 - x1)
            feather_mask_3ch = feather_mask[:, :, np.newaxis]
            
            # Blend blurred and original using feathered mask
            blurred_full[y1:y2, x1:x2] = np.maximum(
                blurred_full[y1:y2, x1:x2],
                roi_blurred.astype(np.float32) * feather_mask_3ch
            )
            blur_mask[y1:y2, x1:x2] = np.maximum(blur_mask[y1:y2, x1:x2], feather_mask)
        
        # Apply the accumulated blur
        blur_mask_3ch = blur_mask[:, :, np.newaxis]
        out_bgr = (
            out_bgr.astype(np.float32) * (1 - blur_mask_3ch) +
            blurred_full * blur_mask_3ch
        ).astype(np.uint8)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    if original_dtype != np.uint8:
        out_rgb = out_rgb.astype(np.float32) / 255.0

    return out_rgb


def apply_clip_rtdetr_blur(input_path, output_path, segments):
    """
    Use MoviePy to write a new video with improved RT-DETR blurring.
    """
    clip = VideoFileClip(input_path)
    duration = clip.duration

    segments_clamped = [
        (max(0, s), min(duration, e))
        for s, e in segments
        if e > 0 and s < duration
    ]
    
    if not segments_clamped:
        print("No violent segments detected. Copying original.")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()
        return

    print("Violent segments (seconds):")
    for s, e in segments_clamped:
        print(f"  {s:.2f} -> {e:.2f} (duration {(e - s):.2f}s)")

    def process_frame(gf, t):
        frame = gf(t)
        inside = any(s <= t < e for s, e in segments_clamped)
        if not inside:
            return frame
        return blur_frame_with_rtdetr_rgb(frame)

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
            print(f"  {s:.2f} -> {e:.2f} (duration {(e - s):.2f}s)")
    else:
        print("No violent segments detected.")

    print("\nStep 2 — Applying improved RT-DETR boundary blur...")
    print(f"Settings:")
    print(f"  - Confidence threshold: {CONF_THRESH}")
    print(f"  - Blur padding: {BLUR_PADDING * 100}%")
    print(f"  - Edge feathering: {FEATHER_PIXELS}px")
    print(f"  - Tracking enabled: {USE_TRACKING}")
    
    apply_clip_rtdetr_blur(INPUT_VIDEO, OUTPUT_VIDEO, segments)
    print("✅ Done! Output saved to:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()