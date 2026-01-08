import os
from pathlib import Path
import torch
from faster_whisper import WhisperModel
from .extractor import split_audio_ffmpeg, get_audio_duration_seconds

# ==================================================
# CONFIG
# ==================================================

CHUNK_SECONDS = 120
MODEL_NAME = "small"

BASE_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = BASE_DIR / "output"
CHUNK_DIR = WORK_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE = "float16" if torch.cuda.is_available() else "float32"
VAD_FILTER = False

# ==================================================
# 🔥 LOAD MODEL ONCE (CRITICAL FIX)
# ==================================================

print("🎙️ Initializing Whisper model (persistent)...", flush=True)

WHISPER_MODEL = WhisperModel(
    MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE
)

print("✅ Whisper model loaded", flush=True)

# ==================================================
# TRANSCRIPTION
# ==================================================

def transcribe_audio_in_chunks(audio_path):
    print(f"\n✂️ Splitting audio into chunks...", flush=True)

    chunks = split_audio_ffmpeg(
        audio_path,
        CHUNK_SECONDS,
        str(CHUNK_DIR)
    )

    print(f"🧩 Total chunks: {len(chunks)}", flush=True)

    all_segments = []
    audio_offset = 0.0

    for i, chunk in enumerate(chunks):
        print(f"\n📝 Transcribing chunk {i+1}/{len(chunks)}", flush=True)

        segments, info = WHISPER_MODEL.transcribe(
            chunk,
            word_timestamps=True,
            vad_filter=VAD_FILTER,
            beam_size=1
        )

        segments = list(segments)

        print(f"🔎 Segments found: {len(segments)}", flush=True)

        for seg in segments:
            words = [
                {
                    "word": w.word,
                    "start": float(w.start) + audio_offset,
                    "end": float(w.end) + audio_offset
                }
                for w in (seg.words or [])
            ]

            all_segments.append({
                "text": seg.text.strip(),
                "start": float(seg.start) + audio_offset,
                "end": float(seg.end) + audio_offset,
                "words": words
            })

        chunk_len = get_audio_duration_seconds(chunk)
        audio_offset += chunk_len

        print(f"✅ Chunk {i+1} done ({chunk_len:.2f}s)", flush=True)

    print(f"\n✅ Whisper transcription COMPLETE", flush=True)
    print(f"📊 Total segments collected: {len(all_segments)}", flush=True)

    return all_segments
