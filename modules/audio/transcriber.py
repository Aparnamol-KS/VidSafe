from pathlib import Path
import torch
from faster_whisper import WhisperModel

from .extractor import split_audio_ffmpeg, get_audio_duration_seconds

CHUNK_SECONDS = 120
MODEL_NAME = "small"
VAD_FILTER = False


class WhisperTranscriber:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if torch.cuda.is_available() else "float32"

        print("🎙️ Loading Whisper model...", flush=True)
        self.model = WhisperModel(
            MODEL_NAME,
            device=device,
            compute_type=compute
        )
        print("✅ Whisper model ready", flush=True)

    def transcribe(self, audio_path: str, work_dir: str):

        chunk_dir = Path(work_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunks = split_audio_ffmpeg(
            audio_path,
            CHUNK_SECONDS,
            str(chunk_dir)
        )

        all_segments = []
        audio_offset = 0.0

        for chunk in chunks:

            segments, _ = self.model.transcribe(
                chunk,
                word_timestamps=True,
                vad_filter=VAD_FILTER,
                beam_size=1
            )

            segments = list(segments)

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

            audio_offset += get_audio_duration_seconds(chunk)

        return all_segments