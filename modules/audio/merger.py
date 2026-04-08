import subprocess
from pathlib import Path


"""
Input 0 → original video
Input 1 → new audio
"-c:v", "copy",- copy video stream exactly as it is
"-map", "0:v:0", - This selects video stream from input 0
"-map", "1:a:0", - Select Audio from Input 1
"-shortest",-
example--
video length = 60 seconds
audio length = 59 seconds

Without -shortest:

last second might have missing audio

With -shortest:

video stops at 59 seconds
"""
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