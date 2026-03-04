import torch
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import WORD_TOXICITY_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "s-nlp/roberta_toxicity_classifier"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "s-nlp/roberta_toxicity_classifier"
).to(DEVICE)

model.eval()

def get_toxic_label(word: str) -> str:

    threat_words = {"kill", "die", "murder", "destroy"}
    abuse_words = {"idiot", "stupid", "moron", "dumb"}

    if word in threat_words:
        return "threatening speech"

    if word in abuse_words:
        return "verbal abuse"

    return "aggressive speech"

def normalize_word(w: str) -> str:
    return w.strip().lower().translate(
        str.maketrans("", "", string.punctuation)
    )


def score_words(words):
    """
    Returns dict {word: probability}
    """
    if not words:
        return {}

    inputs = tokenizer(
        words,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    return {
        w: float(p)
        for w, p in zip(words, probs)
    }

def get_toxic_label(word: str) -> str:

    threat_words = {"kill", "die", "murder", "destroy"}
    abuse_words = {"idiot", "stupid", "moron", "dumb"}

    if word in threat_words:
        return "threatening speech"

    if word in abuse_words:
        return "verbal abuse"

    return "aggressive speech"


def detect_toxic_words(segments):
    candidate_words = []
    candidate_map = {}

    for seg in segments:
        for w in seg["words"]:
            norm = normalize_word(w["word"])
            if not norm:
                continue
            candidate_words.append(norm)
            candidate_map.setdefault(norm, []).append(w)

    candidate_words = list(set(candidate_words))

    word_probs = score_words(candidate_words)

    toxic_segments = []

    for word, prob in word_probs.items():
        if prob >= WORD_TOXICITY_THRESHOLD:
            for wobj in candidate_map[word]:
                label = get_toxic_label(word)
                toxic_segments.append({
                    "start": float(wobj["start"]),
                    "end": float(wobj["end"]),
                    "word": wobj["word"],
                    "confidence": prob,
                    "label": label
                })

    return toxic_segments