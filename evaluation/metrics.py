from typing import List, Tuple
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

def roc(y_true: np.ndarray, y_prob: np.ndarray, *, intervals: int) -> List[Tuple[float, float]]:
    thresholds = np.arange(intervals + 1) / intervals
    points = []
    for threshold in thresholds:
        y_pred = binarise(y_prob, threshold=threshold)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        sensitivity = float(tp / (tp + fn))
        specificity = float(tn / (tn + fp))
        points.append((1 - specificity, sensitivity))
    return points

def curve_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    i = 1
    while i < len(points):
        if points[i][0] == points[i-1][0]:
            points = points[:i] + points[i+1:]
        else:
            i += 1
    if points[-1] != (0.0, 0.0):
        points.append((0.0, 0.0))
    return points

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray, *, intervals: int) -> float:
    points = roc(y_true, y_prob, intervals=intervals)
    unique_points = curve_points(points)[::-1]
    area = 0
    for previous_point, next_point in zip(unique_points[:-1], unique_points[1:]):
        area += 0.5 * (next_point[0] - previous_point[0]) * (next_point[1] + previous_point[1])
    return area
