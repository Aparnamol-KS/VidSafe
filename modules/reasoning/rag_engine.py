import json
from pathlib import Path
from datetime import timedelta

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).parent
POLICIES_DIR = BASE_DIR / "policies"

DEFAULT_POLICY_FILE = POLICIES_DIR / "youtube_multimodal_policies.json"
DEFAULT_FAISS_INDEX = BASE_DIR / "policy_index.faiss"
DEFAULT_METADATA_FILE = BASE_DIR / "policy_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==============================
# UTILITIES
# ==============================

def sec_to_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=float(seconds)))


def determine_severity(score: float) -> str:
    if score >= 0.75:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


# ==============================
# VECTOR DB BUILD
# ==============================

def build_policy_vector_db(policy_file, faiss_index_file, metadata_file):

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
    embeddings = embedder.encode(texts).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(faiss_index_file))
    metadata_file.write_text(json.dumps(metadata, indent=2))


# ==============================
# MAIN RAG LOGIC
# ==============================

def run_policy_rag(evidence_file, output_file):

    evidence = json.loads(Path(evidence_file).read_text())
    video_id = evidence.get("video_id", "unknown")

    fused_events = evidence.get("fused_events", [])

    # Ensure index exists
    if not DEFAULT_FAISS_INDEX.exists():
        build_policy_vector_db(
            DEFAULT_POLICY_FILE,
            DEFAULT_FAISS_INDEX,
            DEFAULT_METADATA_FILE
        )

    index = faiss.read_index(str(DEFAULT_FAISS_INDEX))
    policy_meta = json.loads(DEFAULT_METADATA_FILE.read_text())
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Precompute query embeddings
    video_query = "violent physical attack detected visually"
    audio_query = "violent or threatening speech"

    video_q_emb = embedder.encode([video_query]).astype("float32")
    audio_q_emb = embedder.encode([audio_query]).astype("float32")

    _, video_policy_idxs = index.search(video_q_emb, k=3)
    _, audio_policy_idxs = index.search(audio_q_emb, k=3)

    violations = []

    for event in fused_events:

        severity = determine_severity(event["severity_score"])

        # Normalize modality naming
        normalized_modalities = [
            "video" if m == "vision" else m
            for m in event["modalities"]
        ]

        if "video" in normalized_modalities and "audio" in normalized_modalities:
            policy_idxs = list(
                set(video_policy_idxs[0]) | set(audio_policy_idxs[0])
            )
            modality_label = "video+audio"
        elif "video" in normalized_modalities:
            policy_idxs = video_policy_idxs[0]
            modality_label = "video"
        else:
            policy_idxs = audio_policy_idxs[0]
            modality_label = "audio"

        for idx in policy_idxs:
            policy = policy_meta[idx]

            if not any(m in policy["applies_to"] for m in normalized_modalities):
                continue

            violations.append({
                "modality": modality_label,
                "policy_id": policy["policy_id"],
                "policy_name": policy["policy_name"],
                "category": policy["category"],
                "severity": severity,
                "confidence": event["severity_score"],
                "timestamp": (
                    f"{sec_to_timestamp(event['start'])} - "
                    f"{sec_to_timestamp(event['end'])}"
                ),
                "reason": "Multimodal evidence indicates policy violation"
            })

    # Deduplicate
    violations = list({
        json.dumps(v, sort_keys=True): v for v in violations
    }.values())

    fusion_severity = determine_severity(
        max([e["severity_score"] for e in fused_events], default=0.0)
    )

    output = {
        "video_id": video_id,
        "fusion_severity": fusion_severity,
        "video_max_confidence": round(
            max([e["vision_confidence"] for e in fused_events], default=0.0), 3
        ),
        "audio_max_confidence": round(
            max([e["audio_confidence"] for e in fused_events], default=0.0), 3
        ),
        "total_violations": len(violations),
        "policy_violations": violations
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(output, indent=2))