import subprocess
from pathlib import Path

def merge_audio_to_video(original_video, new_audio, out_video):

    original_video = Path(original_video).resolve()
    new_audio = Path(new_audio).resolve()
    out_video = Path(out_video).resolve()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-i", str(new_audio),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_video)
    ]

    subprocess.check_call(cmd)