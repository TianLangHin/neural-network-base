from collections import namedtuple
from model import NeuralNetwork
from typing import Callable, List, TypeVar
import numpy as np

from data import DataScaler

Score = TypeVar('Score')
Estimator = Callable[[np.ndarray, np.ndarray], NeuralNetwork]
Scorer = Callable[[NeuralNetwork, np.ndarray, np.ndarray], Score]

def k_fold_cross_val_scores(
        estimator: Estimator,
        scorer: Scorer,
        scaler: DataScaler,
        *,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        verbose: bool) -> List[Score]:

    assert x.shape[0] == y.shape[0]
    data_size = x.shape[0]
    fold_size = data_size // k

    fold_points = list(range(0, data_size, fold_size))
    if len(fold_points) == k:
        fold_points.append(data_size)
    else:
        fold_points[-1] = data_size
    folds = [np.arange(start, end) for start, end in zip(fold_points[:-1], fold_points[1:])]

    scores = []
    for fold_num, fold_mask in enumerate(folds):
        x_train = np.delete(x, fold_mask, axis=0)
        x_test = x[fold_mask,:]
        y_train = np.delete(y, fold_mask, axis=0)
        y_test = y[fold_mask,:]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        trained_model = estimator(x_train, y_train)
        score = scorer(trained_model, x_test, y_test)
        scores.append(score)
        if verbose:
            print('Fold number', fold_num + 1, 'Score:', score)
    return scores
