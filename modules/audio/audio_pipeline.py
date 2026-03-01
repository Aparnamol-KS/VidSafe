from pathlib import Path

from vidsafe.config import (
    BEEP_FREQUENCY,
    BEEP_MIN_DURATION_MS,
    USE_BEEP
)

from .extractor import extract_audio_ffmpeg
from .transcriber import WhisperTranscriber
from .roberta_detector import detect_toxic_words
from .detoxify_detector import SentenceToxicityDetector
from .censor import censor_audio


class AudioPipeline:

    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.sentence_detector = SentenceToxicityDetector()

    def run(self, input_video: str, work_dir: str):

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_path = work_dir / "extracted_audio.mp3"
        censored_audio_path = work_dir / "censored_audio.mp3"

        # -------------------------
        # 1️⃣ Extract audio
        # -------------------------
        extract_audio_ffmpeg(input_video, str(audio_path))

        # -------------------------
        # 2️⃣ Transcription
        # -------------------------
        segments = self.transcriber.transcribe(
            str(audio_path),
            str(work_dir)
        )

        # -------------------------
        # 3️⃣ Sentence-level toxicity
        # -------------------------
        toxic_sentences = self.sentence_detector.detect(segments)

        # -------------------------
        # 4️⃣ Word-level toxicity
        # -------------------------
        toxic_words = detect_toxic_words(segments)

        # -------------------------
        # 5️⃣ Censoring
        # -------------------------
        if toxic_words:
            censor_audio(
                str(audio_path),
                toxic_words,
                str(censored_audio_path),
                beep_freq=BEEP_FREQUENCY,
                min_ms=BEEP_MIN_DURATION_MS,
                use_beep=USE_BEEP
            )
        else:
            censored_audio_path = audio_path

        return {
            "segments": segments,
            "toxic_sentences": toxic_sentences,
            "word_level_toxic": toxic_words,
            "censored_audio": str(censored_audio_path)
        }