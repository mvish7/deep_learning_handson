import torch
import torchvision
import torch.nn as nn
import numpy as np


import sys
sys.path.append('/home/flying-dutchman/PycharmProjects')

from dl_functions import load_fashion_mnist, train_on_cpu, show_fashion_mnist, get_fashion_mnist_labels


class MLP(nn.Module):

    def __init__(self, num_inputs=784, num_hidden=512, num_outputs=10):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.layer1 = nn.Linear(num_inputs, num_hidden)
        self.layer2 = nn.Linear(num_hidden, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.reshape(-1, X.shape[2]*X.shape[3])

        H = self.relu(self.layer1(X))
        y_hat = self.layer2(H)
        return y_hat


def main():
    dataset_path = '../Linear_Neural_Networks/FashionMNIST/'
    batch_size = 128

    train_loader, test_loader = load_fashion_mnist(dataset_path, batch_size)

    # creating model object, loss_fun and optimizer

    net = MLP()

    loss_fun = nn.CrossEntropyLoss()

    num_epochs = 10
    lr = 0.4

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_on_cpu(net, train_loader, test_loader, loss_fun, optimizer, num_epochs)

    # visualizing predictions

    # for X, y in test_loader:
    #     break
    # true_labels = get_fashion_mnist_labels(y.numpy())
    # pred_labels = get_fashion_mnist_labels(np.argmax(net(X).data.numpy(), axis=1))
    # titles = [truelabel + '\n' + predlabel for truelabel, predlabel in zip(true_labels, pred_labels)]
    # show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == "__main__":
    main()
