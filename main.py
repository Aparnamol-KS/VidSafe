#!/usr/bin/env python3

"""
VidSafe backend entry.
Used only for manual testing.
"""

from modules.pipeline import VidSafePipeline

if __name__ == "__main__":
    pipeline = VidSafePipeline("outputs")
    pipeline.run("test_video.mp4")