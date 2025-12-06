import os
import subprocess
from pathlib import Path
from pydub import AudioSegment

VERBOSE = True

def run_cmd(cmd):
    if VERBOSE:
        print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

def extract_audio_ffmpeg(video_path: str, out_audio: str):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", out_audio
    ]
    run_cmd(cmd)

def get_audio_duration_seconds(path: str) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0

def split_audio_ffmpeg(in_audio: str, chunk_seconds: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "chunk_%04d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", in_audio,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        out_pattern
    ]
    run_cmd(cmd)
    return sorted([str(p) for p in Path(out_dir).glob("chunk_*.mp3")])
