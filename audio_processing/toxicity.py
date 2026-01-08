import string
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==================================================
# DEVICE SETUP
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Toxicity module running on: {DEVICE}", flush=True)

# ==================================================
# CONFIG
# ==================================================

TOXICITY_THRESHOLD_WORD = 0.5
MAX_WORDS_PER_SEGMENT = 25     # hard cap (critical)
BATCH_SIZE = 64                # safe for RTX 2050

PROFANITY_SET = {
    "fuck", "shit", "bitch", "bastard", "asshole", "damn", "dick",
    "piss", "crap", "motherfucker", "slut"
}

# ==================================================
# MODEL (LOAD ONCE)
# ==================================================

print("📦 Loading RoBERTa toxicity model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    "s-nlp/roberta_toxicity_classifier"
)

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    "s-nlp/roberta_toxicity_classifier",
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
).to(DEVICE)

roberta_model.eval()
print("✅ Toxicity model ready", flush=True)

# ==================================================
# UTILITIES
# ==================================================

def normalize_word(w: str) -> str:
    return w.strip().lower().translate(
        str.maketrans("", "", string.punctuation)
    )


def batch_roberta_probs(words):
    """
    Batched GPU inference with explicit flushing.
    """
    print(f"🧠 Classifying {len(words)} candidate words...", flush=True)

    probs = {}
    total_batches = (len(words) - 1) // BATCH_SIZE + 1

    for i in range(0, len(words), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        batch = words[i:i + BATCH_SIZE]

        print(f"   → Batch {batch_idx}/{total_batches}", flush=True)

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = roberta_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1]

        for w, p in zip(batch, scores):
            probs[w] = float(p.detach().cpu())

        # ---- HARD SYNC (prevents silent stalls) ----
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    print("✅ Word-level classification complete", flush=True)
    return probs


# ==================================================
# MAIN FUNCTION
# ==================================================

def find_toxic_sentences_and_words(segments):
    """
    Returns:
      - toxic_sentences (currently passthrough)
      - merged word-level toxic segments
    """

    print("🔍 Running toxicity analysis...", flush=True)

    toxic_sentences = []
    candidate_words = []
    candidate_map = {}

    for seg in segments:
        toxic_sentences.append(seg)

        # ---- LIMIT WORDS PER SEGMENT ----
        for w in seg["words"][:MAX_WORDS_PER_SEGMENT]:
            norm = normalize_word(w["word"])
            if not norm:
                continue

            if norm in PROFANITY_SET or len(norm) >= 4:
                candidate_words.append(norm)
                candidate_map.setdefault(norm, []).append(w)

    # ---- DEDUP ----
    candidate_words = list(dict.fromkeys(candidate_words))

    if not candidate_words:
        print("✅ No candidate toxic words found", flush=True)
        return toxic_sentences, []

    # ---- CLASSIFY ----
    word_probs = batch_roberta_probs(candidate_words)

    # ---- BUILD SEGMENTS ----
    word_level_segments = []

    for norm_word, prob in word_probs.items():
        if prob >= TOXICITY_THRESHOLD_WORD:
            for wobj in candidate_map[norm_word]:
                word_level_segments.append({
                    "start": float(wobj["start"]),
                    "end": float(wobj["end"]),
                    "word": wobj["word"],
                    "score": prob
                })

    # ---- MERGE OVERLAPS ----
    word_level_segments.sort(key=lambda x: x["start"])
    merged = []

    for seg in word_level_segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        if seg["start"] <= last["end"] + 0.05:
            last["end"] = max(last["end"], seg["end"])
            last["word"] += " / " + seg["word"]
            last["score"] = max(last["score"], seg["score"])
        else:
            merged.append(seg)

    print(f"✅ Toxic words detected: {len(merged)}", flush=True)
    return toxic_sentences, merged
