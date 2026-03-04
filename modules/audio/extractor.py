import subprocess
from pathlib import Path
from pydub import AudioSegment


def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed: {e.stderr.decode()}")
    

"""
-acodec, libmp3lame → Specifies the audio encoder. 
libmp3lame → converts audio to MP3 format.

-ar → audio sampling rate (16000 Hz for whisper)
-ac → number of audio channels = 1

-vn → video none  - So only audio is kept.
"""



def extract_audio_ffmpeg(video_path: str, out_audio: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        str(out_audio)
    ]
    run_cmd(cmd)


def get_audio_duration_seconds(path: str) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0

""" 
-y → overwrite output file if it exists.
-i → input file.
-segment_time - Each chunk = how many seconds
"-f", "segment", Tells FFmpeg to split audio into segments.
"-c", "copy" - Do not re-encode audio. Just copy the audio stream.
"""

def split_audio_ffmpeg(in_audio: str, chunk_seconds: int, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = str(out_dir / "chunk_%04d.mp3")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_audio),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        out_pattern
    ]

    run_cmd(cmd)

    return sorted(str(p) for p in out_dir.glob("chunk_*.mp3"))