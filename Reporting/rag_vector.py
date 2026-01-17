#!/usr/bin/env python3
"""
rag_vector.py

Policy-aware RAG reasoning for VidSafe

- RT-DETR = visual truth
- Audio toxicity = audio truth
- Policies are content-based and multimodal
- Modality-aware thresholds
- Fusion severity support
"""

import json
from pathlib import Path
from datetime import timedelta

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==================================================
# BASE PATHS
# ==================================================

BASE_DIR = Path(__file__).parent
POLICIES_DIR = BASE_DIR / "policies"

DEFAULT_POLICY_FILE = POLICIES_DIR / "youtube_multimodal_policies.json"
DEFAULT_FAISS_INDEX = BASE_DIR / "policy_index.faiss"
DEFAULT_METADATA_FILE = BASE_DIR / "policy_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==================================================
# THRESHOLDS (CALIBRATED)
# ==================================================

VIDEO_POLICY_THRESHOLD = 0.40   # RT-DETR calibrated
AUDIO_POLICY_THRESHOLD = 0.60   # Audio toxicity sharper

# ==================================================
# UTILITIES
# ==================================================

def sec_to_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=float(seconds)))


def determine_severity(confidence: float) -> str:
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"


def determine_fusion_severity(video_conf: float, audio_conf: float) -> str:
    if video_conf >= 0.7 and audio_conf >= 0.7:
        return "Critical"
    if video_conf >= 0.7 or audio_conf >= 0.7:
        return "High"
    if video_conf >= 0.6 or audio_conf >= 0.6:
        return "Medium"
    return "Low"


# ==================================================
# VECTOR DB BUILD
# ==================================================

def build_policy_vector_db(policy_file, faiss_index_file, metadata_file):
    print("\n🔧 Building Policy Vector DB")

    with open(policy_file, "r", encoding="utf-8") as f:
        policies = json.load(f)

    texts = []
    metadata = []

    for p in policies:
        text = (
            f"{p['policy_name']}. "
            f"{p['description']}. "
            f"Category: {p['category']}. "
            f"Applies to: {', '.join(p.get('applies_to', []))}."
        )
        texts.append(text)

        metadata.append({
            "policy_id": p["policy_id"],
            "policy_name": p["policy_name"],
            "category": p["category"],
            "applies_to": p.get("applies_to", ["video", "audio"])
        })

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(
        texts, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(faiss_index_file))
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print("✅ Policy Vector DB ready")


# ==================================================
# MAIN RAG LOGIC
# ==================================================

def run_policy_rag(evidence_file, output_file):
    print("\n🧠 Running Policy-aware RAG (Video + Audio)")

    evidence = json.loads(Path(evidence_file).read_text())
    video_id = evidence.get("video_id", "unknown")

    rtdetr_detections = evidence.get("rtdetr_detections", [])
    audio_violations = evidence.get("audio_violations", [])

    print(f"🎥 RT-DETR detections: {len(rtdetr_detections)}")
    print(f"🔊 Audio violations: {len(audio_violations)}")

    # Build DB if missing
    if not DEFAULT_FAISS_INDEX.exists():
        build_policy_vector_db(
            DEFAULT_POLICY_FILE,
            DEFAULT_FAISS_INDEX,
            DEFAULT_METADATA_FILE
        )

    index = faiss.read_index(str(DEFAULT_FAISS_INDEX))
    policy_meta = json.loads(DEFAULT_METADATA_FILE.read_text())
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    violations = []

    # ==================================================
    # VIDEO VIOLENCE (RT-DETR)
    # ==================================================

    video_confidences = [d["confidence"] for d in rtdetr_detections]
    video_max_conf = max(video_confidences) if video_confidences else 0.0

    for det in rtdetr_detections:
        conf = det["confidence"]
        time_sec = det.get("time", 0.0)

        if conf < VIDEO_POLICY_THRESHOLD:
            continue

        query = "violent physical attack detected visually"
        q_emb = embedder.encode([query]).astype("float32")
        _, idxs = index.search(q_emb, k=3)

        for i in idxs[0]:
            policy = policy_meta[i]

            # ✅ Modality filter
            if "video" not in policy["applies_to"]:
                continue

            violations.append({
                "modality": "video",
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": determine_severity(conf),
                "confidence": conf,
                "timestamp": sec_to_timestamp(time_sec),
                "reason": "Detected violent visual activity (RT-DETR)"
            })

    # ==================================================
    # AUDIO TOXICITY / THREATS
    # ==================================================

    audio_confidences = [a["confidence"] for a in audio_violations]
    audio_max_conf = max(audio_confidences) if audio_confidences else 0.0

    for a in audio_violations:
        conf = a["confidence"]

        if conf < AUDIO_POLICY_THRESHOLD:
            continue

        query = "violent or threatening speech"
        q_emb = embedder.encode([query]).astype("float32")
        _, idxs = index.search(q_emb, k=3)

        for i in idxs[0]:
            policy = policy_meta[i]

            # ✅ Modality filter
            if "audio" not in policy["applies_to"]:
                continue

            violations.append({
                "modality": "audio",
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": determine_severity(conf),
                "confidence": conf,
                "timestamp": (
                    f"{sec_to_timestamp(a['start'])} - "
                    f"{sec_to_timestamp(a['end'])}"
                ),
                "reason": f"Toxic or threatening speech: '{a.get('text','')}'"
            })

    # ==================================================
    # FINALIZE OUTPUT
    # ==================================================

    # Deduplicate
    violations = list({
        json.dumps(v, sort_keys=True): v for v in violations
    }.values())

    fusion_severity = determine_fusion_severity(
        video_max_conf, audio_max_conf
    )

    output = {
        "video_id": video_id,
        "fusion_severity": fusion_severity,
        "video_max_confidence": round(video_max_conf, 3),
        "audio_max_confidence": round(audio_max_conf, 3),
        "total_violations": len(violations),
        "policy_violations": violations
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(output, indent=2))

    print(f"🔥 Fusion severity: {fusion_severity}")
    print(f"📄 Output saved → {output_file}")


# ==================================================
# STANDALONE TEST
# ==================================================

def main():
    evidence_path = BASE_DIR / "output" / "moderation_evidence.json"
    output_path = BASE_DIR / "output" / "policy_violations_output.json"

    print("🚀 Running standalone RAG test")
    run_policy_rag(evidence_path, output_path)


if __name__ == "__main__":
    main()
