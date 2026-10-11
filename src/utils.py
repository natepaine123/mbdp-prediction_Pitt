import numpy as np
from sklearn.metrics import roc_curve

def find_threshold_for_sensitivity(y_true, y_pred_proba, target_sensitivity=0.95):
    """
    Find the highest threshold that still achieves at least target_sensitivity.
    This minimizes false positives while meeting the sensitivity goal.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    idx = np.where(tpr >= target_sensitivity)[0]
    if len(idx) == 0:
        raise ValueError(f"Cannot achieve sensitivity {target_sensitivity}")
    best_idx = idx[0]
    threshold = thresholds[best_idx]
    return threshold