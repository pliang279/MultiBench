import sklearn.metrics


def ptsort(tu):
  return tu[0]

def AUPRC(pts):
  true_labels = [int(x[1]) for x in pts]
  predicted_probs = [x[0] for x in pts]
  return sklearn.metrics.average_precision_score(true_labels, predicted_probs)

