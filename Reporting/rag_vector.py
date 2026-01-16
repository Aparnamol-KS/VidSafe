#!/usr/bin/env python3
"""
rag_vector.py

- Builds Policy Vector DB (if missing)
- Performs Policy-aware RAG reasoning
- Confidence-based violation decision
- Aligned with Violence Detection output
"""

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==================================================
# BASE PATHS
# ==================================================

BASE_DIR = Path(__file__).parent
POLICIES_DIR = BASE_DIR / "policies"

DEFAULT_POLICY_FILE = POLICIES_DIR / "youtube_violence.json"
DEFAULT_FAISS_INDEX = BASE_DIR / "policy_index.faiss"
DEFAULT_METADATA_FILE = BASE_DIR / "policy_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==================================================
# CONFIG
# ==================================================

POLICY_VIOLATION_THRESHOLD = 0.60  # confidence gate

# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def sec_to_timestamp(seconds):
    return str(timedelta(seconds=float(seconds)))


def determine_severity(confidence: float) -> str:
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"



def determine_fusion_severity(
    video_confidences: list,
    audio_confidences: list
) -> str:
    """
    Determines overall severity based on multimodal agreement.
    """

    if not video_confidences and not audio_confidences:
        return "None"

    max_video = max(video_confidences) if video_confidences else 0.0
    max_audio = max(audio_confidences) if audio_confidences else 0.0

    if max_video >= 0.7 and max_audio >= 0.7:
        return "Critical"
    elif max_video >= 0.7 or max_audio >= 0.7:
        return "High"
    elif max_video >= 0.6 or max_audio >= 0.6:
        return "Medium"
    else:
        return "Low"

# ==================================================
# PHASE 1: BUILD POLICY VECTOR DATABASE
# ==================================================

def build_policy_vector_db(
    policy_file: Path,
    faiss_index_file: Path,
    metadata_file: Path
):
    print("\n🔧 Building Policy Vector Database...")

    with open(policy_file, "r", encoding="utf-8") as f:
        policies = json.load(f)

    policy_texts = []
    policy_metadata = []

    for p in policies:
        policy_texts.append(
            f"Policy Name: {p['policy_name']}. "
            f"Description: {p['description']}. "
            f"Category: {p['category']}."
        )

        policy_metadata.append({
            "policy_id": p["policy_id"],
            "policy_name": p["policy_name"],
            "category": p["category"]
        })

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(
        policy_texts, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(faiss_index_file))

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(policy_metadata, f, indent=2)

    print("✅ Policy Vector DB built successfully")
    print(f"📁 FAISS Index: {faiss_index_file}")
    print(f"📁 Metadata  : {metadata_file}")


# ==================================================
# PHASE 2: POLICY-AWARE RAG
# ==================================================

def run_policy_rag(
    evidence_file: Path,
    output_file: Path,
    policy_file: Path = DEFAULT_POLICY_FILE,
    faiss_index_file: Path = DEFAULT_FAISS_INDEX,
    metadata_file: Path = DEFAULT_METADATA_FILE
):
    print("\n🧠 Running Policy-aware RAG Reasoning (Video + Audio)")

    evidence_file = Path(evidence_file)
    output_file = Path(output_file)

    # ---- Build DB if missing ----
    if not faiss_index_file.exists() or not metadata_file.exists():
        build_policy_vector_db(
            policy_file,
            faiss_index_file,
            metadata_file
        )

    # ---- Load Evidence ----
    print(f"📄 Loading evidence: {evidence_file}")
    with open(evidence_file, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    video_id = evidence.get("video_id", "unknown")
    violent_segments = evidence.get("violent_segments", [])
    audio_violations = evidence.get("audio_violations", [])

    print(f"🎥 Video ID: {video_id}")
    print(f"⚠️ Visual segments: {len(violent_segments)}")
    print(f"🔊 Audio violations: {len(audio_violations)}")

    # ---- Load Vector DB ----
    index = faiss.read_index(str(faiss_index_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        policy_metadata = json.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    violations = []
    video_confidences = []
    audio_confidences = []


    # ==================================================
    # VIDEO VIOLENCE REASONING
    # ==================================================
    for idx, segment in enumerate(violent_segments, 1):
        start = segment["start"]
        end = segment["end"]
        confidence = segment.get("confidence", 0.0)
        video_confidences.append(confidence)

        print(
            f"\n🎥 [VIDEO] Segment {idx}: "
            f"{sec_to_timestamp(start)} → {sec_to_timestamp(end)} | "
            f"confidence={confidence:.2f}"
        )

        if confidence < POLICY_VIOLATION_THRESHOLD:
            print("   ⛔ Below policy threshold → ignored")
            continue

        query = f"violent physical activity in a video with confidence {confidence}"

        query_embedding = embedder.encode([query]).astype("float32")
        _, results = index.search(query_embedding, k=3)

        for i in results[0]:
            policy = policy_metadata[i]
            severity = determine_severity(confidence)

            print(
                f"   ✅ Violates (VIDEO): {policy['policy_name']} "
                f"(Severity: {severity})"
            )

            violations.append({
                "modality": "video",
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": severity,
                "confidence": confidence,
                "timestamp": (
                    f"{sec_to_timestamp(start)} - "
                    f"{sec_to_timestamp(end)}"
                ),
                "reason": "Detected violent visual content"
            })

    # ==================================================
    # AUDIO TOXICITY / THREAT REASONING
    # ==================================================
    for idx, audio in enumerate(audio_violations, 1):
        start = audio["start"]
        end = audio["end"]
        confidence = audio.get("confidence", 0.0)
        text = audio.get("text", "")
        audio_confidences.append(confidence)


        print(
            f"\n🔊 [AUDIO] Segment {idx}: "
            f"{sec_to_timestamp(start)} → {sec_to_timestamp(end)} | "
            f"confidence={confidence:.2f}"
        )

        if confidence < POLICY_VIOLATION_THRESHOLD:
            print("   ⛔ Below policy threshold → ignored")
            continue

        query = (
            f"spoken violent or threatening speech such as '{text}' "
            f"with confidence {confidence}"
        )

        query_embedding = embedder.encode([query]).astype("float32")
        _, results = index.search(query_embedding, k=3)

        for i in results[0]:
            policy = policy_metadata[i]
            severity = determine_severity(confidence)

            print(
                f"   ✅ Violates (AUDIO): {policy['policy_name']} "
                f"(Severity: {severity})"
            )

            violations.append({
                "modality": "audio",
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": severity,
                "confidence": confidence,
                "timestamp": (
                    f"{sec_to_timestamp(start)} - "
                    f"{sec_to_timestamp(end)}"
                ),
                "reason": f"Toxic or threatening speech: '{text}'"
            })

    # ---- Deduplicate ----
    violations = list({
        json.dumps(v, sort_keys=True): v
        for v in violations
    }.values())

    fusion_severity = determine_fusion_severity(
        video_confidences,
        audio_confidences
    )

    print(f"\n🔥 Fusion Severity Level: {fusion_severity}")


    # ---- Save Output ----
    output = {
        "video_id": video_id,
        "fusion_severity": fusion_severity,
        "video_max_confidence": max(video_confidences) if video_confidences else None,
        "audio_max_confidence": max(audio_confidences) if audio_confidences else None,
        "total_violations": len(violations),
        "policy_violations": violations
    }


    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Policy reasoning completed (Multimodal)")
    print(f"📄 Output saved → {output_file}")
    print(f"🚨 Total Violations: {len(violations)}")




# ==================================================
# MAIN: STANDALONE TEST ENTRY POINT
# ==================================================

def main():
    """
    Standalone entry point for testing Policy-aware RAG.
    """

    print("\n🚀 Starting standalone Policy RAG test")

    evidence_path = BASE_DIR / "output" / "moderation_evidence.json"
    output_path = BASE_DIR / "output" / "policy_violations_output.json"

    print(f"📥 Evidence file : {evidence_path}")
    print(f"📤 Output file   : {output_path}")

    run_policy_rag(
        evidence_file=evidence_path,
        output_file=output_path
    )


if __name__ == "__main__":
    main()
