import cv2
import torch
import torch.nn.functional as F
from collections import deque
from transformers import CLIPProcessor, CLIPModel


# ===============================
# CONFIG
# ===============================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

SAMPLE_EVERY_N_FRAMES = 5
CLIP_LOW_THRESHOLD = 0.22
CLIP_HIGH_THRESHOLD = 0.45
TEMPORAL_WINDOW = 5
TOPK_PROMPTS = 3


# ===============================
# ENRICHED VIOLENCE PROMPTS
# ===============================
VIOLENCE_PROMPTS = [
    "people fighting violently",
    "a violent physical fight",
    "a person punching another person",
    "a person kicking another person",
    "a person beating another person",
    "a man attacking another man",
    "a woman being attacked",
    "a person attacking with a knife",
    "a person attacking with a gun",
    "a weapon being used to attack someone",
    "a stabbing incident",
    "a shooting incident",
    "blood on a person",
    "a badly injured person",
    "a person bleeding heavily",
    "a person lying injured after an attack",
    "aggressive violent behavior",
    "a person threatening violence",
    "a violent confrontation",
    "a person being physically harmed",
    "mob violence",
    "a violent crowd fight",
    "riot with people fighting",
    "police arrest involving force",
    "people restraining someone violently"
]


# ===============================
# DEVICE
# ===============================
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# LOAD MODEL (SINGLETON)
# ===============================
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
_clip_model.eval()

_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)


# ===============================
# CACHE TEXT EMBEDDINGS (SAFE)
# ===============================
_text_inputs = _clip_processor(
    text=VIOLENCE_PROMPTS,
    return_tensors="pt",
    padding=True,
    truncation=True
)

_text_inputs = {k: v.to(_device) for k, v in _text_inputs.items()}

with torch.no_grad():
    text_outputs = _clip_model.get_text_features(**_text_inputs)

    if not isinstance(text_outputs, torch.Tensor):
        if hasattr(text_outputs, "pooler_output"):
            _text_features = text_outputs.pooler_output
        else:
            raise RuntimeError("Unexpected text output structure from CLIP")
    else:
        _text_features = text_outputs

    _text_features = F.normalize(_text_features, dim=-1)


# ===============================
# CLIP VIOLENCE FILTER
# ===============================
def run_clip_filter(video_path: str):
    """
    Returns list of violent segments:
    [
        {
            "start": float,
            "end": float,
            "confidence": float
        }
    ]
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    frame_idx = 0
    violent_segments = []
    current_segment = None

    score_window = deque(maxlen=TEMPORAL_WINDOW)

    while True:
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
        )

        image_inputs = {k: v.to(_device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_outputs = _clip_model.get_image_features(**image_inputs)

            if not isinstance(image_outputs, torch.Tensor):
                if hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output
                else:
                    raise RuntimeError("Unexpected image output structure from CLIP")
            else:
                image_features = image_outputs

            image_features = F.normalize(image_features, dim=-1)

            similarity = image_features @ _text_features.T

            topk_scores = torch.topk(
                similarity.squeeze(),
                k=min(TOPK_PROMPTS, similarity.shape[-1])
            ).values

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

        # ---- Segment detection ----
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
                    current_segment["confidence"],
                    smoothed_score
                )
        else:
            if current_segment is not None:
                violent_segments.append(current_segment)
                current_segment = None

        frame_idx += 1

    if current_segment is not None:
        violent_segments.append(current_segment)

    cap.release()

    return violent_segments