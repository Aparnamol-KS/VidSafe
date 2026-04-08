import cv2
import torch
import torch.nn.functional as F
from collections import deque, Counter
from transformers import CLIPProcessor, CLIPModel

from config import (
    CLIP_SAMPLE_RATE,
    CLIP_THRESHOLD,
    CLIP_TEMPORAL_WINDOW
)

# ===============================
# MODEL CONFIG
# ===============================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TOPK_PROMPTS = 3

# ===============================
# VIOLENCE PROMPTS
# ===============================
VIOLENCE_PROMPTS = [
    "people fighting violently",
    "a violent physical fight",
    "a person punching another person",
    "a person kicking another person",
    "a person beating another person",
    "a physical assault between people",
    "a person violently attacking someone",

    "a man attacking another man",
    "a woman being attacked",
    "a person attacking another person",

    "a person attacking with a knife",
    "a person attacking with a gun",
    "a weapon being used to attack someone",
    "a violent attack with a weapon",
    "a stabbing incident",
    "a shooting incident",

    "blood on a person",
    "a badly injured person",
    "a person bleeding heavily",
    "a person lying injured after an attack",

    "aggressive violent behavior",
    "a person threatening violence",
    "a violent confrontation",

    "mob violence",
    "a violent crowd fight",
    "riot with people fighting"
]

# ===============================
# DEVICE
# ===============================
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD MODEL
# ===============================
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
_clip_model.eval()

_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# ===============================
# SAFE FEATURE EXTRACTION
# ===============================
def _extract_tensor(output):

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if isinstance(output, (list, tuple)):
        return output[0]

    raise RuntimeError("Unexpected CLIP output type")


# ===============================
# CACHE TEXT EMBEDDINGS
# ===============================
with torch.no_grad():

    text_inputs = _clip_processor(
        text=VIOLENCE_PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_inputs = {k: v.to(_device) for k, v in text_inputs.items()}

    text_outputs = _clip_model.get_text_features(**text_inputs)

    text_features = _extract_tensor(text_outputs)

    text_features = F.normalize(text_features, dim=-1)


# ===============================
# CLIP FILTER
# ===============================
def run_clip_filter(video_path: str):

    """
    Returns:
    [
        {
            "start": float,
            "end": float,
            "confidence": float,
            "queries": [top matched prompts]
        }
    ]
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    frame_idx = 0
    segments = []
    current_segment = None

    score_window = deque(maxlen=CLIP_TEMPORAL_WINDOW)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # sample frames
        if frame_idx % CLIP_SAMPLE_RATE != 0:
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

            image_features = _extract_tensor(image_outputs)

            image_features = F.normalize(image_features, dim=-1)

            similarity = image_features @ text_features.T

            # ---------------------------
            # Top-K prompt selection
            # ---------------------------
            topk = torch.topk(
                similarity.squeeze(0),
                k=min(TOPK_PROMPTS, similarity.shape[-1])
            )

            topk_scores = topk.values
            topk_indices = topk.indices

            raw_score = float(topk_scores.mean().cpu())

            topk_prompts = [
                VIOLENCE_PROMPTS[i] for i in topk_indices.cpu().tolist()
            ]

        score_window.append(raw_score)

        smoothed_score = sum(score_window) / len(score_window)

        time_sec = frame_idx / fps

        # ---------------------------
        # Segment logic
        # ---------------------------
        if smoothed_score >= CLIP_THRESHOLD:

            if current_segment is None:

                current_segment = {
                    "start": time_sec,
                    "end": time_sec,
                    "confidence": smoothed_score,
                    "query_counter": Counter(topk_prompts)
                }

            else:

                current_segment["end"] = time_sec

                current_segment["confidence"] = max(
                    current_segment["confidence"],
                    smoothed_score
                )

                current_segment["query_counter"].update(topk_prompts)

        else:

            if current_segment is not None:

                # select most frequent prompts
                top_queries = [
                    q for q, _ in
                    current_segment["query_counter"].most_common(TOPK_PROMPTS)
                ]

                current_segment["queries"] = top_queries
                del current_segment["query_counter"]

                segments.append(current_segment)

                current_segment = None

        frame_idx += 1

    # finalize last segment
    if current_segment is not None:

        top_queries = [
            q for q, _ in
            current_segment["query_counter"].most_common(TOPK_PROMPTS)
        ]

        current_segment["queries"] = top_queries
        del current_segment["query_counter"]

        segments.append(current_segment)

    cap.release()

    return segments