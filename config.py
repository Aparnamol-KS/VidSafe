# ==============================
# AUDIO - WHISPER
# ==============================
WHISPER_MODEL_NAME = "small"
WHISPER_CHUNK_SECONDS = 120
WHISPER_VAD_FILTER = False

# ==============================
# AUDIO - CENSORING
# ==============================
BEEP_FREQUENCY = 1000
BEEP_MIN_DURATION_MS = 120
USE_BEEP = True

# ==============================
# CLIP
# ==============================
CLIP_SAMPLE_RATE = 5
CLIP_THRESHOLD = 0.25
CLIP_TEMPORAL_WINDOW = 5

# ==============================
# RT-DETR
# ==============================
RTDETR_CONF_THRESHOLD = 0.35
RTDETR_MIN_CONSECUTIVE_FRAMES = 3

# ==============================
# TOXICITY
# ==============================
WORD_TOXICITY_THRESHOLD = 0.5
SENTENCE_TOXICITY_THRESHOLD = 0.6

# ==============================
# FUSION
# ==============================
FUSION_TIME_WINDOW = 2.0