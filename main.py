#!/usr/bin/env python3
"""
MAIN MULTI-MODAL MODERATION PIPELINE
Final Output:
1) Blurred + Audio-masked video
2) Policy-aware RAG report
"""

# ==================================================
# WINDOWS MULTIPROCESSING FIX (MUST BE FIRST)
# ==================================================
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ==================================================
# IMPORTS (SAFE AFTER SPAWN)
# ==================================================
import json
from pathlib import Path
import torch

from audio_processing.main import run_audio_moderation
from violence_detection.rtdetr_clip import run_violence_detection_and_blur
from audio_processing.merger import merge_audio_to_video
from Reporting.rag_vector import run_policy_rag

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
# MAIN PIPELINE
# ==================================================

def main():
    print("🔥 ROOT MAIN.PY EXECUTING 🔥", flush=True)

    print("🖥️ GPU available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("🚀 Using GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("⚠️ Using CPU", flush=True)

    print("\n🚀 Starting Multi-Modal Moderation Pipeline\n", flush=True)

    # --------------------------------------------------
    # STEP 1: AUDIO MODERATION
    # --------------------------------------------------
    print("🔊 Step 1: Audio moderation (masking profanity)", flush=True)

    audio_results = run_audio_moderation(
        input_video=str(INPUT_VIDEO),
        work_dir=str(AUDIO_WORK_DIR)
    )

    # --------------------------------------------------
    # STEP 2: VIOLENCE DETECTION
    # --------------------------------------------------
    print("\n🎯 Step 2: Violence detection & blurring", flush=True)

    violence_results = run_violence_detection_and_blur(
        input_video=str(INPUT_VIDEO),
        output_video=str(BLURRED_VIDEO)
    )

    # --------------------------------------------------
    # STEP 3: MERGE AUDIO + VIDEO
    # --------------------------------------------------
    print("\n🔀 Step 3: Merging censored audio with blurred video", flush=True)

    merge_audio_to_video(
        original_video=str(BLURRED_VIDEO),
        new_audio=str(audio_results["censored_audio"]),
        out_video=str(FINAL_VIDEO)
    )

    # --------------------------------------------------
    # STEP 4: BUILD MODERATION EVIDENCE
    # --------------------------------------------------
    print("\n🧾 Step 4: Writing moderation evidence", flush=True)

    evidence = {
        "video_id": INPUT_VIDEO.stem,
        "audio_moderation": audio_results,
        "violent_segments": violence_results.get("segments", []),
        "rtdetr_detections": violence_results.get("detections", [])
    }

    with open(EVIDENCE_FILE, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"📄 Evidence saved → {EVIDENCE_FILE}", flush=True)

    # --------------------------------------------------
    # STEP 5: POLICY-AWARE RAG
    # --------------------------------------------------
    print("\n🧠 Step 5: Policy-aware RAG reasoning", flush=True)

    run_policy_rag(
        evidence_file=str(EVIDENCE_FILE),
        output_file=str(POLICY_OUTPUT)
    )

    # --------------------------------------------------
    # DONE
    # --------------------------------------------------
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print(f"🎬 Final moderated video → {FINAL_VIDEO}", flush=True)
    print(f"📄 Policy report → {POLICY_OUTPUT}", flush=True)


# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()
