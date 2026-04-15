from __future__ import annotations

from typing import Dict, List, Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], label_names: List[str]) -> Dict[str, object]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        'metrics': {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        },
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist(),
    }
