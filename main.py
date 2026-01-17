#!/usr/bin/env python3
"""
ROOT ENTRY POINT
"""

# ==================================================
# WINDOWS MULTIPROCESSING FIX
# ==================================================
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ==================================================
# IMPORTS
# ==================================================
from pathlib import Path
import torch

from core.pipeline import VidSafePipeline

# ==================================================
# PATH CONFIG (TEMPORARY — can move to config.yaml)
# ==================================================
INPUT_VIDEO = Path("data") / "video2.mp4"
OUTPUT_DIR = Path("output")

# ==================================================
# MAIN
# ==================================================
def main():
    print("🔥 VidSafe Root Entry Executing 🔥", flush=True)

    print("🖥️ GPU available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("🚀 Using GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("⚠️ Using CPU", flush=True)

    pipeline = VidSafePipeline(output_dir=OUTPUT_DIR)

    results = pipeline.run(INPUT_VIDEO)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print(f"🎬 Final moderated video → {results['final_video']}", flush=True)
    print(f"📄 Policy report → {results['policy_report']}", flush=True)


# ==================================================
# ENTRY
# ==================================================
if __name__ == "__main__":
    main()
