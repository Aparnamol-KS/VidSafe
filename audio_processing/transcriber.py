import os
from pathlib import Path
from faster_whisper import WhisperModel
from extractor import split_audio_ffmpeg, get_audio_duration_seconds

# CONFIG
CHUNK_SECONDS = 120
WORK_DIR = "../output"
MODEL_NAME = "small"

import torch
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE = "float16" if torch.cuda.is_available() else "float32"
VAD_FILTER = False

def transcribe_audio_in_chunks(audio_path):
    print("Loading Whisper model:", MODEL_NAME)
    model = WhisperModel(MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

    chunk_dir = os.path.join(WORK_DIR, "chunks")
    chunks = split_audio_ffmpeg(audio_path, CHUNK_SECONDS, chunk_dir)
    print("Chunks to transcribe:", len(chunks))

    all_segments = []
    audio_offset = 0.0

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)} (offset {audio_offset}s)")
        segments, info = model.transcribe(chunk, word_timestamps=True, vad_filter=VAD_FILTER, beam_size=1)
        for seg in segments:
            words = [{"word": w.word, "start": float(w.start)+audio_offset, "end": float(w.end)+audio_offset} for w in seg.words]
            all_segments.append({
                "text": seg.text.strip(),
                "start": float(seg.start)+audio_offset,
                "end": float(seg.end)+audio_offset,
                "words": words
            })
        chunk_len = get_audio_duration_seconds(chunk)
        audio_offset += chunk_len

    return all_segments
