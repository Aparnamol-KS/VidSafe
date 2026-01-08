#!/usr/bin/env python3
"""
rag_vector.py

- Builds Policy Vector DB (if missing)
- Performs Policy-aware RAG reasoning
- Path-safe & project-structure aware
"""

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==================================================
# BASE PATHS (CRITICAL FIX)
# ==================================================

BASE_DIR = Path(__file__).parent              # VidSafe/Reporting
POLICIES_DIR = BASE_DIR / "policies"

DEFAULT_POLICY_FILE = POLICIES_DIR / "youtube_violence.json"
DEFAULT_FAISS_INDEX = BASE_DIR / "policy_index.faiss"
DEFAULT_METADATA_FILE = BASE_DIR / "policy_metadata.json"

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
    policy_file: Path,
    faiss_index_file: Path,
    metadata_file: Path
):
    print("🔧 Building Policy Vector Database...")

    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")

    with open(policy_file, "r", encoding="utf-8") as f:
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

    faiss.write_index(index, str(faiss_index_file))

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(policy_metadata, f, indent=2)

    print("✅ Policy Vector DB Created")
    print(f"📁 FAISS index → {faiss_index_file}")
    print(f"📁 Metadata   → {metadata_file}")


# ==================================================
# PHASE 2: POLICY-AWARE RAG (CALLED BY main.py)
# ==================================================

def run_policy_rag(
    evidence_file,
    output_file,
    policy_file=DEFAULT_POLICY_FILE,
    faiss_index_file=DEFAULT_FAISS_INDEX,
    metadata_file=DEFAULT_METADATA_FILE
):
    print("🧠 Running Policy-aware RAG Reasoning...")

    # -------- Normalize paths --------
    evidence_file = Path(evidence_file)
    output_file = Path(output_file)
    policy_file = Path(policy_file)
    faiss_index_file = Path(faiss_index_file)
    metadata_file = Path(metadata_file)

    # -------- Build vector DB if missing --------
    if not faiss_index_file.exists() or not metadata_file.exists():
        build_policy_vector_db(
            policy_file,
            faiss_index_file,
            metadata_file
        )

    # -------- Sanity checks --------
    for f in [evidence_file, faiss_index_file, metadata_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file missing: {f}")

    # -------- Load moderation evidence --------
    with open(evidence_file, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    video_id = evidence.get("video_id", "unknown")
    violent_segments = evidence.get("violent_segments", [])
    detections = evidence.get("rtdetr_detections", [])

    detected_objects = {d.get("object", "unknown") for d in detections}

    # -------- Load vector DB --------
    index = faiss.read_index(str(faiss_index_file))

    with open(metadata_file, "r", encoding="utf-8") as f:
        policy_metadata = json.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    violations = []

    # -------- RAG reasoning --------
    for segment in violent_segments:
        start = segment["start_time"]
        end = segment["end_time"]

        query = (
            f"violent content involving {', '.join(detected_objects)} "
            f"in videos for children"
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

    # -------- Deduplicate --------
    unique = {json.dumps(v, sort_keys=True): v for v in violations}
    violations = list(unique.values())

    # -------- Save output --------
    output = {
        "video_id": video_id,
        "total_violations": len(violations),
        "policy_violations": violations
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("✅ Policy-aware RAG reasoning completed")
    print(f"📄 Output saved → {output_file}")


# ==================================================
# OPTIONAL: STANDALONE TEST
# ==================================================

if __name__ == "__main__":
    run_policy_rag(
        evidence_file=Path("output") / "moderation_evidence.json",
        output_file=Path("output") / "policy_violations_output.json"
    )
