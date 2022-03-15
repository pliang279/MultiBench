"""Implements utils for getting AUPRC score given input and output pairs."""
import sklearn.metrics

def ptsort(tu):
    """Return first element of input."""
    return tu[0]


def AUPRC(pts):
    """Get average precision given a list of (true, predicted) pairs.

    Args:
        pts (List): List of true, predicted pairs

    Returns:
        float: Average Precision Score.
    """
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)
