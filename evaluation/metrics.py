import numpy as np

def binarise(x: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    return np.where(x >= threshold, 1, 0)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    comparison = y_true == binarise(y_pred, threshold=threshold)
    return comparison.sum() / comparison.size

def precision(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    y_pred = binarise(y_pred, threshold=threshold)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fp)

def recall(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    y_pred = binarise(y_pred, threshold=threshold)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    p = precision(y_true, y_pred, threshold=threshold)
    r = recall(y_true, y_pred, threshold=threshold)
    return 2 / (1/p + 1/r)
