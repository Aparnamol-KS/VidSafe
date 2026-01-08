#!/usr/bin/env python3
"""
MAIN MULTI-MODAL MODERATION PIPELINE
Final Output:
1) Blurred + Audio-masked video
2) Policy-aware RAG report
"""

import json
from pathlib import Path

# ==================================================
# PATH CONFIG
# ==================================================

INPUT_VIDEO = Path("data") / "vid1.mp4"


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

AUDIO_WORK_DIR = OUTPUT_DIR / "audio"
AUDIO_WORK_DIR.mkdir(exist_ok=True)

BLURRED_VIDEO = OUTPUT_DIR / "blurred_video.mp4"
FINAL_VIDEO = OUTPUT_DIR / "final_blurred_masked_video.mp4"

EVIDENCE_FILE = OUTPUT_DIR / "moderation_evidence.json"
POLICY_OUTPUT = OUTPUT_DIR / "policy_violations_output.json"

# ==================================================
# IMPORT MODULES
# ==================================================

from audio_processing.main import run_audio_moderation
from violence_detection.rtdetr_clip import run_violence_detection_and_blur
from audio_processing.merger import merge_audio_to_video
from Reporting.rag_vector import run_policy_rag

# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    print("\n🚀 Starting Multi-Modal Moderation Pipeline\n")

    # --------------------------------------------------
    # STEP 1: AUDIO MODERATION (MASKING)
    # --------------------------------------------------
    print("🔊 Step 1: Audio moderation (masking profanity)")

    audio_results = run_audio_moderation(
        input_video=str(INPUT_VIDEO),
        work_dir=str(AUDIO_WORK_DIR)
    )

    # --------------------------------------------------
    # STEP 2: VIOLENCE DETECTION + BLURRING
    # --------------------------------------------------
    print("\n🎯 Step 2: Violence detection & blurring")

    violence_results = run_violence_detection_and_blur(
        input_video=str(INPUT_VIDEO),
        output_video=str(BLURRED_VIDEO)
    )

    # --------------------------------------------------
    # STEP 3: MERGE MASKED AUDIO + BLURRED VIDEO
    # --------------------------------------------------
    print("\n🔀 Step 3: Merging censored audio with blurred video")

    merge_audio_to_video(
        video_path=str(BLURRED_VIDEO),
        audio_path=str(audio_results["censored_audio"]),
        output_path=str(FINAL_VIDEO)
    )

    # --------------------------------------------------
    # STEP 4: BUILD MODERATION EVIDENCE
    # --------------------------------------------------
    print("\n🧾 Step 4: Writing moderation evidence")

    evidence = {
        "video_id": INPUT_VIDEO.stem,
        "audio_moderation": audio_results,
        "violent_segments": violence_results.get("segments", []),
        "rtdetr_detections": violence_results.get("detections", [])
    }

    with open(EVIDENCE_FILE, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"📄 Evidence saved → {EVIDENCE_FILE}")

    # --------------------------------------------------
    # STEP 5: POLICY-AWARE RAG
    # --------------------------------------------------
    print("\n🧠 Step 5: Policy-aware RAG reasoning")

    run_policy_rag(
        evidence_file=str(EVIDENCE_FILE),
        output_file=str(POLICY_OUTPUT)
    )

    # --------------------------------------------------
    # DONE
    # --------------------------------------------------
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"🎬 Final moderated video → {FINAL_VIDEO}")
    print(f"📄 Policy report → {POLICY_OUTPUT}")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
