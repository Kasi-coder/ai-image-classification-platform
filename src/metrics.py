import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def topk_accuracy(y_true, probs, k=5):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))

def classification_metrics(y_true, probs):
    pred = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "top5_accuracy": topk_accuracy(y_true, probs, 5),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
