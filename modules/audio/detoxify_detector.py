from detoxify import Detoxify
from config import SENTENCE_TOXICITY_THRESHOLD


class SentenceToxicityDetector:

    def __init__(self):
        self.model = Detoxify("original")

    def detect(self, segments):

        toxic_sentences = []

        for seg in segments:

            text = seg["text"]

            scores = self.model.predict(text)

            toxicity_score = scores["toxicity"]

            if toxicity_score >= SENTENCE_TOXICITY_THRESHOLD:

                # derive semantic label
                label = "aggressive speech"

                if scores.get("threat", 0) > 0.5:
                    label = "threatening speech"

                elif scores.get("insult", 0) > 0.5:
                    label = "verbal abuse"

                elif scores.get("obscene", 0) > 0.5:
                    label = "abusive language"

                toxic_sentences.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "confidence": float(toxicity_score),
                    "label": label
                })

        return toxic_sentences