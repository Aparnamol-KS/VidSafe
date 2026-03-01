from detoxify import Detoxify
from ...config import SENTENCE_TOXICITY_THRESHOLD


class SentenceToxicityDetector:

    def __init__(self):
        self.model = Detoxify("original")

    def detect(self, segments):
        toxic_sentences = []

        for seg in segments:
            text = seg["text"]
            score = self.model.predict(text)["toxicity"]

            if score >= SENTENCE_TOXICITY_THRESHOLD:
                toxic_sentences.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "confidence": float(score)
                })

        return toxic_sentences