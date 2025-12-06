import os
from extractor import extract_audio_ffmpeg, get_audio_duration_seconds
from transcriber import transcribe_audio_in_chunks
from toxicity import find_toxic_sentences_and_words
from censor import censor_audio
from merger import merge_audio_to_video

# CONFIG
VIDEO_IN = "../data/video2.mp4"
WORK_DIR = "../output"
os.makedirs(WORK_DIR, exist_ok=True)
AUDIO_MP3 = os.path.join(WORK_DIR, "extracted_audio.mp3")

def run_pipeline():
    print("1) Extract audio ...")
    extract_audio_ffmpeg(VIDEO_IN, AUDIO_MP3)
    dur = get_audio_duration_seconds(AUDIO_MP3)
    print(f"Extracted audio duration: {dur:.2f} sec")

    print("2) Transcribing audio in chunks ...")
    segments = transcribe_audio_in_chunks(AUDIO_MP3)

    print("3) Detect toxic sentences and candidate words ...")
    toxic_sentences, word_level_toxic = find_toxic_sentences_and_words(segments)
    print(f"Found {len(toxic_sentences)} toxic sentences, {len(word_level_toxic)} word segments.")

    print("4) Censoring audio ...")
    censored_audio = os.path.join(WORK_DIR, "censored_audio.mp3")
    censor_audio(AUDIO_MP3, word_level_toxic, censored_audio)
    print("Censored audio saved to:", censored_audio)

    print("5) Merging censored audio back ...")
    censored_video = os.path.join(WORK_DIR, "censored_video.mp4")
    merge_audio_to_video(VIDEO_IN, censored_audio, censored_video)
    print("Final censored video:", censored_video)

    return {
        "segments": segments,
        "toxic_sentences": toxic_sentences,
        "word_level_toxic": word_level_toxic,
        "censored_audio": censored_audio,
        "censored_video": censored_video
    }

if __name__ == "__main__":
    results = run_pipeline()
