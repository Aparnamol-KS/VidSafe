from pydub import AudioSegment
from pydub.generators import Sine

BEEP_FREQ = 1000
BEEP_MIN_MS = 120
USE_BEEP = True

def censor_audio(in_audio_path, toxic_word_segments, out_audio_path,
                 use_beep=USE_BEEP, beep_freq=BEEP_FREQ, min_ms=BEEP_MIN_MS):
    audio = AudioSegment.from_file(in_audio_path)
    output = AudioSegment.empty()
    cursor_ms = 0
    toxic_word_segments = sorted(toxic_word_segments, key=lambda x: x['start'])
    for seg in toxic_word_segments:
        start_ms = max(0, int(seg['start']*1000))
        end_ms = min(len(audio), int(seg['end']*1000))
        if end_ms <= start_ms:
            continue
        if start_ms > cursor_ms:
            output += audio[cursor_ms:start_ms]
        dur = max(end_ms - start_ms, min_ms)
        if use_beep:
            beep = Sine(beep_freq).to_audio_segment(duration=dur) - 6
            output += beep
        else:
            output += AudioSegment.silent(duration=dur)
        cursor_ms = end_ms
    if cursor_ms < len(audio):
        output += audio[cursor_ms:]
    output.export(out_audio_path, format="mp3")
    return out_audio_path
