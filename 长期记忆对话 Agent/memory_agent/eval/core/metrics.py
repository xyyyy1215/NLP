import re
import string
from collections import Counter


def normalize(text: object) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def f1_score(prediction: object, reference: object) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: object, reference: object) -> float:
    return float(normalize(prediction) == normalize(reference))
