import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score


# Accuracy, balanced accuracy, macro precision and F1, per-class accuracy and the confusion matrix
def classification_report(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    per_class = np.divide(matrix.diagonal(), support, out=np.zeros(len(labels)), where=support > 0)
    present = support > 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(per_class[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_accuracy": {name: float(value) for name, value in zip(class_names, per_class)},
        "confusion_matrix": matrix.tolist(),
    }
