import cv2
import torch
from transformers import CLIPProcessor, CLIPModel

# ===============================
# CONFIG
# ===============================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD = 0.18          # was 0.30 (too high)
SAMPLE_EVERY_N_FRAMES = 5      # was 15 (too sparse)


VIOLENCE_PROMPTS = [
    "people fighting violently",
    "a physical fight between people",
    "a person punching another person",
    "a person kicking another person",
    "blood on a person",
    "a violent assault",
    "weapon attack on a person",
    "a man attacking another man",
    "a woman being attacked"
]


# ===============================
# LOAD MODEL (SINGLETON)
# ===============================
_device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# ---- Cache text embeddings (OPTIMIZED) ----
_text_inputs = _clip_processor(
    text=VIOLENCE_PROMPTS,
    return_tensors="pt",
    padding=True
).to(_device)

with torch.no_grad():
    _text_features = _clip_model.get_text_features(**_text_inputs)
    _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)

# ===============================
# CLIP FILTER
# ===============================
def run_clip_filter(video_path: str):
    """
    Returns list of violent clips:
    [
        {
            "start": float,
            "end": float,
            "confidence": float
        }
    ]
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # safe fallback

    frame_idx = 0
    violent_segments = []
    current_segment = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Image only (text already cached) ----
        image_inputs = _clip_processor(
            images=rgb,
            return_tensors="pt"
        ).to(_device)

        with torch.no_grad():
            image_features = _clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ _text_features.T
            score = similarity.max().item()

        time_sec = frame_idx / fps
        print(f"[CLIP] t={time_sec:.2f}s score={score:.3f}")


        if score >= CLIP_THRESHOLD:
            if current_segment is None:
                current_segment = {
                    "start": time_sec,
                    "end": time_sec,
                    "confidence": score
                }
            else:
                current_segment["end"] = time_sec
                current_segment["confidence"] = max(
                    current_segment["confidence"], score
                )
        else:
            if current_segment:
                violent_segments.append(current_segment)
                current_segment = None

        frame_idx += 1
        

    if current_segment:
        violent_segments.append(current_segment)

    cap.release()
    return violent_segments
