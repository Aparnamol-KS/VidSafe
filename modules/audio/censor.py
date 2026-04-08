from pydub import AudioSegment
from pydub.generators import Sine


def merge_overlapping_segments(segments):
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0]]

    for current in segments[1:]:
        last = merged[-1]

        if current["start"] <= last["end"]:
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(current)

    return merged


def censor_audio(in_audio_path, toxic_word_segments, out_audio_path,
                 beep_freq=1000, min_ms=120, use_beep=True):

    audio = AudioSegment.from_file(in_audio_path)
    output = AudioSegment.empty()

    toxic_word_segments = merge_overlapping_segments(toxic_word_segments)

    cursor_ms = 0

    for seg in toxic_word_segments:
        start_ms = max(0, int(seg["start"] * 1000))
        end_ms = min(len(audio), int(seg["end"] * 1000))

        if end_ms <= start_ms:
            continue

        if start_ms > cursor_ms:
            output += audio[cursor_ms:start_ms]

        duration = max(end_ms - start_ms, min_ms)

        if use_beep:
            beep = Sine(beep_freq).to_audio_segment(duration=duration) - 6
            output += beep
        else:
            output += AudioSegment.silent(duration=duration)

        cursor_ms = end_ms

    if cursor_ms < len(audio):
        output += audio[cursor_ms:]

    output.export(out_audio_path, format="mp3")
    return out_audio_path