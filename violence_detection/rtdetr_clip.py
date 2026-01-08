import cv2
import torch
import json
import numpy as np
from ultralytics import RTDETR
from transformers import CLIPProcessor, CLIPModel

# =========================================================
# MODEL CONFIG
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).parent
RTDETR_WEIGHTS = BASE_DIR / "rtdetr_train.pt"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CLIP_THRESHOLD = 0.30
DETECTION_CONF = 0.40
TEMPORAL_WINDOW = 8

VIOLENCE_PROMPTS = [
    "a violent fight",
    "people fighting",
    "physical violence",
    "blood and injury",
    "aggressive behavior",
    "weapon attack"
]


# =========================================================
# MODEL LOADING
# =========================================================

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = RTDETR(RTDETR_WEIGHTS)
    detector.model.to(device)

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    clip_model.eval()
    return detector, clip_model, clip_processor, device


def is_violent(crop, clip_model, clip_processor, device):
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    inputs = clip_processor(
        text=VIOLENCE_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    score = probs.max().item()
    return score > CLIP_THRESHOLD, score


# =========================================================
# WRAPPED PIPELINE FUNCTION
# =========================================================

def run_violence_detection_and_blur(
    input_video,
    output_video
):
    """
    Detects violent visual content using RT-DETR + CLIP,
    blurs violent regions, and returns structured evidence
    for policy RAG.
    """

    detector, clip_model, clip_processor, device = load_models()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    violent_segments = []
    detections = []

    frame_index = 0
    current_event = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        time_sec = frame_index / fps

        results = detector(frame, conf=DETECTION_CONF, device=0 if device == "cuda" else "cpu")


        violent_frame = False
        max_clip_score = 0.0
        detected_objects = set()

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                violent, score = is_violent(
                    crop, clip_model, clip_processor, device
                )

                max_clip_score = max(max_clip_score, score)

                if violent:
                    violent_frame = True
                    detected_objects.add("person")

                    blurred = cv2.GaussianBlur(crop, (51, 51), 0)
                    frame[y1:y2, x1:x2] = blurred

                    detections.append({
                        "time": round(time_sec, 2),
                        "object": "person",
                        "confidence": round(score, 3)
                    })

        # -------- TEMPORAL EVENT AGGREGATION --------
        if violent_frame:
            if current_event is None:
                current_event = {
                    "start_time": round(time_sec, 2),
                    "objects": set(),
                    "confidence": 0.0
                }

            current_event["objects"].update(detected_objects)
            current_event["confidence"] = max(
                current_event["confidence"], max_clip_score
            )
        else:
            if current_event is not None:
                current_event["end_time"] = round(time_sec, 2)
                current_event["objects"] = list(current_event["objects"])
                violent_segments.append(current_event)
                current_event = None

        writer.write(frame)

    if current_event is not None:
        current_event["end_time"] = round(frame_index / fps, 2)
        current_event["objects"] = list(current_event["objects"])
        violent_segments.append(current_event)

    cap.release()
    writer.release()

    print("✅ Blurred video saved:", output_video)

    return {
        "segments": violent_segments,
        "detections": detections
    }


# =========================================================
# OPTIONAL STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    run_violence_detection_and_blur(
        input_video="test1.mp4",
        output_video="output_blurred.mp4"
    )
