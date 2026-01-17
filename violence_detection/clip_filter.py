import cv2
import torch
from collections import deque
from transformers import CLIPProcessor, CLIPModel


# ===============================
# CONFIG
# ===============================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- Sampling & thresholds ----
SAMPLE_EVERY_N_FRAMES = 5        # dense sampling
CLIP_LOW_THRESHOLD = 0.22        # weak violence
CLIP_HIGH_THRESHOLD = 0.45       # strong violence
TEMPORAL_WINDOW = 5              # smoothing window
TOPK_PROMPTS = 3                 # top-k similarity averaging


# ===============================
# ENRICHED VIOLENCE PROMPTS
# ===============================
VIOLENCE_PROMPTS = [

    # ---- Direct physical violence ----
    "people fighting violently",
    "a violent physical fight",
    "a person punching another person",
    "a person kicking another person",
    "a person beating another person",
    "a man attacking another man",
    "a woman being attacked",

    # ---- Weapon-based violence ----
    "a person attacking with a knife",
    "a person attacking with a gun",
    "a weapon being used to attack someone",
    "a stabbing incident",
    "a shooting incident",

    # ---- Injury & blood ----
    "blood on a person",
    "a badly injured person",
    "a person bleeding heavily",
    "a person lying injured after an attack",

    # ---- Aggressive / implied violence ----
    "aggressive violent behavior",
    "a person threatening violence",
    "a violent confrontation",
    "a person being physically harmed",

    # ---- Crowd / riot ----
    "mob violence",
    "a violent crowd fight",
    "riot with people fighting",

    # ---- Contextual cues ----
    "police arrest involving force",
    "people restraining someone violently"
]


# ===============================
# LOAD MODEL (SINGLETON)
# ===============================
_device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# ---- Cache text embeddings ----
_text_inputs = _clip_processor(
    text=VIOLENCE_PROMPTS,
    return_tensors="pt",
    padding=True
).to(_device)

with torch.no_grad():
    _text_features = _clip_model.get_text_features(**_text_inputs)
    _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)


# ===============================
# CLIP VIOLENCE FILTER
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
        fps = 30.0

    frame_idx = 0
    violent_segments = []
    current_segment = None

    # ---- Temporal smoothing ----
    score_window = deque(maxlen=TEMPORAL_WINDOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_inputs = _clip_processor(
            images=rgb,
            return_tensors="pt"
        ).to(_device)

        with torch.no_grad():
            image_features = _clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ _text_features.T

            # ---- Top-K aggregation (robust) ----
            topk_scores = torch.topk(similarity.squeeze(), k=TOPK_PROMPTS).values
            raw_score = topk_scores.mean().item()

        # ---- Temporal smoothing ----
        score_window.append(raw_score)
        smoothed_score = sum(score_window) / len(score_window)

        time_sec = frame_idx / fps

        print(
            f"[CLIP] t={time_sec:7.2f}s "
            f"raw={raw_score:.3f} "
            f"smooth={smoothed_score:.3f}"
        )

        # ---- Segment logic ----
        if smoothed_score >= CLIP_LOW_THRESHOLD:
            if current_segment is None:
                current_segment = {
                    "start": time_sec,
                    "end": time_sec,
                    "confidence": smoothed_score
                }
            else:
                current_segment["end"] = time_sec
                current_segment["confidence"] = max(
                    current_segment["confidence"], smoothed_score
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
