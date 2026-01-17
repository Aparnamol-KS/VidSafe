from violence_detection.vision_pipeline import run_vision_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)

result = run_vision_pipeline(
    video_path="h1.mp4",
    output_path="output_blurred.mp4"
)

print(result)
