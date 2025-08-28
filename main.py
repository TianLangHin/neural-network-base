from collections import namedtuple
from typing import List
import functools
import matplotlib.pyplot as plt
import numpy as np
import sys

from activation import ReLU, Sigmoid
from data import BatchLoader, MinMaxScaler, ZScoreScaler, load_dataset, train_test_split
from evaluation.metrics import accuracy, f1_score, precision, recall
from evaluation.validation import k_fold_cross_val_scores
from loss import BinaryCrossEntropy
from model import Layer, NeuralNetwork
from optimiser import AdamOptimiser, NaiveOptimiser, Optimiser

HyperParamsNN = namedtuple('HyperParamsNN', ['layers', 'seed'])
HyperParamsTrain = namedtuple('HyperParamsTrain', ['lr', 'loss_fn', 'optim_type', 'batch_size', 'epochs'])

Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1'])

def make_estimator(
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        nn_hyperparams: HyperParamsNN,
        train_hyperparams: HyperParamsTrain) -> NeuralNetwork:

    layers, seed = nn_hyperparams
    lr, loss_fn, optim_type, batch_size, epochs = train_hyperparams

    nn = NeuralNetwork(in_features=x_train.shape[1], layers=layers, seed=seed)

    optim: Optimiser = optim_type(nn, learning_rate=lr)
    batch_loader = BatchLoader(seed=seed)

    for i in range(epochs):
        batches = batch_loader.load_batch(x_train, y_train, batch_size=batch_size, shuffle=True)
        for x_batch, y_batch in batches:
            _ = nn.forward(x_batch)
            nn.backward(y_batch, loss_fn)
            optim.step()

    return nn

def score(nn: NeuralNetwork, x: np.ndarray, y_true: np.ndarray) -> Metrics:
    y_pred = nn.forward(x)
    return Metrics(
        accuracy(y_true, y_pred),
        precision(y_true, y_pred),
        recall(y_true, y_pred),
        f1_score(y_true, y_pred))

def main(seed: int):
    x, y_true = load_dataset()

    layers = [
        Layer(50, ReLU()),
        Layer(20, ReLU()),
        Layer(10, ReLU()),
        Layer(1, Sigmoid())
    ]
    scaler = MinMaxScaler(x.shape[1], 0, 1)

    k_folds = 5
    lr = 0.001
    batch_size = 32
    epochs = 1500
    loss_fn = BinaryCrossEntropy()

    estimator = functools.partial(
        make_estimator,
        nn_hyperparams=HyperParamsNN(layers, seed),
        train_hyperparams=HyperParamsTrain(lr, loss_fn, AdamOptimiser, batch_size, epochs))

    scores = k_fold_cross_val_scores(estimator, score, scaler, x=x, y=y_true, k=k_folds, verbose=True)
    score_mean = np.array(scores).mean(axis=0)
    score_std = np.array(scores).std(axis=0)
    print('Mean:')
    print('  Accuracy: {0:.4f} Precision: {1:.4f} Recall: {2:.4f} F1-Score: {3:.4f}'.format(*score_mean))
    print('Standard Deviation:')
    print('  Accuracy: {0:.4f} Precision: {1:.4f} Recall: {2:.4f} F1-Score: {3:.4f}'.format(*score_std))

if __name__ == '__main__':
    main(int(sys.argv[1]))
