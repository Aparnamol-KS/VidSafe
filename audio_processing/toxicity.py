import string
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

TOXICITY_THRESHOLD_SENT = 0.5
TOXICITY_THRESHOLD_WORD = 0.5
PROFANITY_SET = {"fuck","shit","bitch","bastard","asshole","damn","dick","piss","crap","motherfucker","nigger","slut"}

detox = Detoxify("original")
tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
roberta_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier")
roberta_model.eval()
if torch.cuda.is_available():
    roberta_model.to("cuda")

def normalize_word(w):
    return w.strip().lower().translate(str.maketrans("", "", string.punctuation))

def sentence_toxicity_score(sentence):
    res = detox.predict(sentence)
    return float(res.get("toxicity", 0.0))

def batch_roberta_probs(words, batch_size=64):
    unique_words = list(dict.fromkeys(words))
    probs = {}
    for i in range(0, len(unique_words), batch_size):
        batch = unique_words[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k:v.to("cuda") for k,v in inputs.items()}
        with torch.no_grad():
            logits = roberta_model(**inputs).logits.cpu()
            p = torch.softmax(logits, dim=1)[:,1].numpy()
        for w, prob in zip(batch, p):
            probs[w] = float(prob)
    return probs

def find_toxic_sentences_and_words(segments):
    toxic_sentences = []
    candidate_words = []
    candidate_map = {}

    for seg in segments:
        s = seg['text']
        score = sentence_toxicity_score(s)
        seg['toxicity'] = score
        seg['is_toxic'] = (score >= TOXICITY_THRESHOLD_SENT)
        if seg['is_toxic']:
            toxic_sentences.append(seg)
            for w in seg['words']:
                norm = normalize_word(w['word'])
                if not norm:
                    continue
                if (norm in PROFANITY_SET) or (len(norm) >= 3 and any(ch.isalpha() for ch in norm)):
                    candidate_words.append(norm)
                    candidate_map.setdefault(norm, []).append(w)

    candidate_words = list(dict.fromkeys(candidate_words))
    if not candidate_words:
        return toxic_sentences, []

    print(f"Running word-level classifier on {len(candidate_words)} candidate words...")
    word_probs = batch_roberta_probs(candidate_words)

    word_level_toxic_segments = []
    for norm_word, prob in word_probs.items():
        if prob >= TOXICITY_THRESHOLD_WORD:
            for wobj in candidate_map[norm_word]:
                word_level_toxic_segments.append({
                    "start": float(wobj['start']),
                    "end": float(wobj['end']),
                    "word": wobj['word'],
                    "score": prob
                })

    # merge overlapping segments
    word_level_toxic_segments = sorted(word_level_toxic_segments, key=lambda x: x['start'])
    merged = []
    for seg in word_level_toxic_segments:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg['start'] <= last['end'] + 0.05:
            last['end'] = max(last['end'], seg['end'])
            last['word'] = last['word'] + " / " + seg['word']
            last['score'] = max(last['score'], seg['score'])
        else:
            merged.append(seg)
    return toxic_sentences, merged
