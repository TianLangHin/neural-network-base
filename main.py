import numpy as np
import sys

from activation import ReLU, Sigmoid
from loss import BinaryCrossEntropy
from model import Layer, NeuralNetwork
from preprocessing import MinMaxScaler, load_dataset, train_test_split

def main(seed: int):
    x, y_true = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y_true, train_ratio=0.7, seed=seed)
    scaler = MinMaxScaler(x_train.shape[1], 0, 1)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    layers = [
        Layer(50, ReLU()),
        Layer(20, ReLU()),
        Layer(10, ReLU()),
        Layer(1, Sigmoid())
    ]

    nn = NeuralNetwork(in_features=x.shape[1], layers=layers, seed=seed)

    y_pred = np.where(nn.forward(x_train) >= 0.5, 1, 0)
    comparison = y_train == y_pred
    accuracy = comparison.sum() / comparison.size

    print('Accuracy:', accuracy)

if __name__ == '__main__':
    main(int(sys.argv[1]))
