import os
from pathlib import Path

from .extractor import extract_audio_ffmpeg, get_audio_duration_seconds
from .transcriber import transcribe_audio_in_chunks
from .toxicity import find_toxic_sentences_and_words
from .censor import censor_audio
from .merger import merge_audio_to_video


def run_audio_moderation(input_video, work_dir="output"):
    """
    Audio moderation pipeline wrapper.
    Safe for Windows + ffmpeg.
    """

    # ---- FORCE ABSOLUTE PATHS (CRITICAL FIX) ----
    input_video = Path(input_video).resolve()
    work_dir = Path(work_dir).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    work_dir.mkdir(parents=True, exist_ok=True)

    audio_mp3 = work_dir / "extracted_audio.mp3"
    censored_audio = work_dir / "censored_audio.mp3"
    censored_video = work_dir / "censored_video.mp4"

    print("1) Extract audio ...")
    extract_audio_ffmpeg(
        str(input_video),
        str(audio_mp3)
    )

    dur = get_audio_duration_seconds(str(audio_mp3))
    print(f"Extracted audio duration: {dur:.2f} sec")

    print("2) Transcribing audio in chunks ...")
    segments = transcribe_audio_in_chunks(str(audio_mp3))

    print("3) Detect toxic sentences and candidate words ...")
    toxic_sentences, word_level_toxic = find_toxic_sentences_and_words(segments)
    print(
        f"Found {len(toxic_sentences)} toxic sentences, "
        f"{len(word_level_toxic)} word segments."
    )

    print("4) Censoring audio ...")
    censor_audio(
        str(audio_mp3),
        word_level_toxic,
        str(censored_audio)
    )
    print("Censored audio saved to:", censored_audio)

    print("5) Merging censored audio back to video ...")
    merge_audio_to_video(
        str(input_video),
        str(censored_audio),
        str(censored_video)
    )
    print("Final censored video:", censored_video)

    return {
        "segments": segments,
        "toxic_sentences": toxic_sentences,
        "word_level_toxic": word_level_toxic,
        "censored_audio": str(censored_audio),
        "censored_video": str(censored_video)
    }


# --------------------------------------------------
# OPTIONAL: standalone testing (SAFE)
# --------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    test_video = BASE_DIR / "data" / "video2.mp4"
    output_dir = BASE_DIR / "output" / "audio"

    results = run_audio_moderation(
        input_video=test_video,
        work_dir=output_dir
    )
