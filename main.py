import matplotlib.pyplot as plt
import numpy as np
import sys

from activation import ReLU, Sigmoid
from data import BatchLoader, MinMaxScaler, ZScoreScaler, load_dataset, train_test_split
from evaluation.metrics import accuracy, f1_score, precision, recall
from loss import BinaryCrossEntropy
from model import Layer, NeuralNetwork
from optimiser import AdamOptimiser, NaiveOptimiser

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

    lr = 0.001
    batch_size = 32
    epochs = 1000
    interval = 50

    loss_fn = BinaryCrossEntropy()
    nn = NeuralNetwork(in_features=x.shape[1], layers=layers, seed=seed)
    optim = AdamOptimiser(nn, learning_rate=lr)
    batch_loader = BatchLoader(seed=seed)

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for i in range(1, epochs + 1):
        data_batches = batch_loader.load_batch(x_train, y_train, batch_size=batch_size, shuffle=True)
        for x_batch, y_batch in data_batches:
            loss = loss_fn.compute(nn.forward(x_batch), y_batch)
            nn.backward(y_batch, loss_fn)
            optim.step()

        y_pred_train = nn.forward(x_train)
        train_loss_list.append(loss_fn.compute(y_pred_train, y_train).mean())
        train_accuracy_list.append(accuracy(y_train, y_pred_train))

        y_pred_test = nn.forward(x_test)
        test_loss_list.append(loss_fn.compute(y_pred_test, y_test).mean())
        test_accuracy_list.append(accuracy(y_test, y_pred_test))

        if i % interval == 0:
            print(
                'Epoch', '{: >4}'.format(i),
                'completed:',
                'Train:', '{:.2f}%'.format(100 * train_accuracy_list[-1]),
                'Loss({:.4f})'.format(train_loss_list[-1]),
                'Test:', '{:.2f}%'.format(100 * test_accuracy_list[-1]),
                'Loss({:.4f})'.format(test_loss_list[-1]),)

    y_pred_test = nn.forward(x_test)
    print('Final Accuracy:  {:.2f}%'.format(100 * accuracy(y_test, y_pred_test)))
    print('Final Precision: {:.2f}%'.format(100 * precision(y_test, y_pred_test)))
    print('Final Recall:    {:.2f}%'.format(100 * recall(y_test, y_pred_test)))
    print('Final F1-Score:  {:.2f}%'.format(100 * f1_score(y_test, y_pred_test)))

    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_list, test_loss_list)
    plt.savefig('fig_loss.png')

    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracy_list, test_accuracy_list)
    plt.savefig('fig_accuracy.png')

if __name__ == '__main__':
    main(int(sys.argv[1]))
