# ==================================================
# IMPORTS (NO MULTIPROCESSING HERE)
# ==================================================
import time
import gc
from pathlib import Path
import torch

from .extractor import extract_audio_ffmpeg, get_audio_duration_seconds
from .transcriber import transcribe_audio_in_chunks
from .toxicity import find_toxic_sentences_and_words
from .censor import censor_audio
from .merger import merge_audio_to_video


def run_audio_moderation(input_video, work_dir="output"):
    """
    Audio moderation pipeline wrapper.
    Windows + CUDA + Whisper safe.
    """

    input_video = Path(input_video).resolve()
    work_dir = Path(work_dir).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    work_dir.mkdir(parents=True, exist_ok=True)

    audio_mp3 = work_dir / "extracted_audio.mp3"
    censored_audio = work_dir / "censored_audio.mp3"
    censored_video = work_dir / "censored_video.mp4"

    # --------------------------------------------------
    print("🔊 1) Extracting audio ...", flush=True)
    extract_audio_ffmpeg(str(input_video), str(audio_mp3))

    dur = get_audio_duration_seconds(str(audio_mp3))
    print(f"✅ Audio extracted ({dur:.2f} sec)", flush=True)

    # --------------------------------------------------
    print("🎙️ 2) Transcribing audio in chunks ...", flush=True)
    segments = transcribe_audio_in_chunks(str(audio_mp3))

    # ---- HARD CUDA + THREAD BARRIER (CRITICAL) ----
    print("🧱 Whisper finished — stabilizing process", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    time.sleep(0.5)

    print(f"✅ Transcription complete → {len(segments)} segments", flush=True)

    # --------------------------------------------------
    print("🧪 3) Running toxicity detection ...", flush=True)
    toxic_sentences, word_level_toxic = find_toxic_sentences_and_words(segments)

    print(
        f"✅ Toxicity complete → {len(word_level_toxic)} toxic words",
        flush=True
    )

    # --------------------------------------------------
    if not word_level_toxic:
        print("⚠️ No toxic words — skipping censoring", flush=True)
        censored_audio = audio_mp3
    else:
        print("✂️ 4) Censoring audio ...", flush=True)
        censor_audio(str(audio_mp3), word_level_toxic, str(censored_audio))
        print(f"✅ Censored audio saved → {censored_audio}", flush=True)

    # --------------------------------------------------
    print("🎬 5) Merging audio back to video ...", flush=True)
    merge_audio_to_video(
        str(input_video),
        str(censored_audio),
        str(censored_video)
    )

    print(f"🎉 FINAL VIDEO → {censored_video}", flush=True)

    return {
        "segments": segments,
        "toxic_sentences": toxic_sentences,
        "word_level_toxic": word_level_toxic,
        "censored_audio": str(censored_audio),
        "censored_video": str(censored_video)
    }


# # ==================================================
# # OPTIONAL STANDALONE TEST
# # ==================================================
# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).parent.parent
#     test_video = BASE_DIR / "data" / "video2.mp4"
#     output_dir = BASE_DIR / "output" / "audio"

#     run_audio_moderation(test_video, output_dir)
