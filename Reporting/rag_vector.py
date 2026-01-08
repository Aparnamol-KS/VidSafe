#!/usr/bin/env python3
"""
policy_aware_rag.py

- Builds Policy Vector DB (if needed)
- Performs Policy-aware RAG reasoning
"""

import os
import json
import numpy as np
import faiss
from datetime import timedelta
from sentence_transformers import SentenceTransformer

# ==================================================
# CONFIG
# ==================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def sec_to_timestamp(seconds):
    return str(timedelta(seconds=float(seconds)))


def determine_severity(detected_objects, severity_rules):
    for obj in detected_objects:
        if obj in severity_rules:
            return severity_rules[obj].capitalize()
    return "Medium"


# ==================================================
# PHASE 1: BUILD POLICY VECTOR DATABASE
# ==================================================

def build_policy_vector_db(
    policy_file,
    faiss_index_file,
    metadata_file
):
    print("🔧 Building Policy Vector Database...")

    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"{policy_file} not found")

    with open(policy_file, "r") as f:
        policies = json.load(f)

    policy_texts = []
    policy_metadata = []

    for p in policies:
        policy_texts.append(
            f"Policy: {p['policy_name']}. "
            f"Description: {p['description']}. "
            f"Category: {p['category']}. "
            f"Relevant to child safety and kids content."
        )

        policy_metadata.append({
            "policy_id": p["policy_id"],
            "policy_name": p["policy_name"],
            "category": p["category"],
            "severity_rules": p.get("severity_rules", {})
        })

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedder.encode(
        policy_texts,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, faiss_index_file)

    with open(metadata_file, "w") as f:
        json.dump(policy_metadata, f, indent=2)

    print("✅ Policy Vector DB Created")


# ==================================================
# PHASE 2: POLICY-AWARE RAG (WRAPPED)
# ==================================================

def run_policy_rag(
    evidence_file,
    output_file,
    policy_file="youtube_violence.json",
    faiss_index_file="policy_index.faiss",
    metadata_file="policy_metadata.json"
):
    """
    Called by global main.py
    """

    print("🧠 Running Policy-aware RAG Reasoning...")

    # Build vector DB only if missing
    if not os.path.exists(faiss_index_file) or not os.path.exists(metadata_file):
        build_policy_vector_db(
            policy_file,
            faiss_index_file,
            metadata_file
        )

    for file in [evidence_file, faiss_index_file, metadata_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found")

    with open(evidence_file, "r") as f:
        evidence = json.load(f)

    video_id = evidence["video_id"]
    violent_segments = evidence.get("violent_segments", [])
    detections = evidence.get("rtdetr_detections", [])

    detected_objects = {d["object"] for d in detections}

    index = faiss.read_index(faiss_index_file)

    with open(metadata_file, "r") as f:
        policy_metadata = json.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    violations = []

    for segment in violent_segments:
        start = segment["start_time"]
        end = segment["end_time"]

        query = (
            f"violent content involving {', '.join(detected_objects)} "
            f"in animated videos for children"
        )

        query_embedding = embedder.encode([query]).astype("float32")
        _, idx = index.search(query_embedding, k=3)

        for i in idx[0]:
            policy = policy_metadata[i]

            severity = determine_severity(
                detected_objects,
                policy.get("severity_rules", {})
            )

            violations.append({
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": severity,
                "timestamp": f"{sec_to_timestamp(start)} - {sec_to_timestamp(end)}",
                "reason": (
                    f"Detected {', '.join(detected_objects)} violates "
                    f"{policy['policy_name']} guidelines"
                )
            })

    # Remove duplicates
    unique = {json.dumps(v, sort_keys=True): v for v in violations}
    violations = list(unique.values())

    output = {
        "video_id": video_id,
        "total_violations": len(violations),
        "policy_violations": violations
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Policy-aware RAG reasoning completed")
    print(f"📄 Output saved → {output_file}")


# ==================================================
# OPTIONAL: STANDALONE TESTING
# ==================================================

if __name__ == "__main__":
    run_policy_rag(
        evidence_file="moderation_evidence.json",
        output_file="policy_violations_output.json"
    )
