import torch
import torchvision
import torch.nn as nn
import numpy as np


import sys
sys.path.append('/home/flying-dutchman/PycharmProjects')

from dl_functions import load_fashion_mnist, train_on_cpu, relu, show_fashion_mnist, get_fashion_mnist_labels


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # parameters for ip 2 hidden layer
        self.w1 = nn.Parameter(torch.randn(784, 256, requires_grad=True)*0.01)
        self.b1 = nn.Parameter(torch.zeros(256, requires_grad=True))

        # parameters from hidden 2 op layer
        self.w2 = nn.Parameter(torch.randn(256, 10, requires_grad=True)*0.01)
        self.b2 = nn.Parameter(torch.zeros(10, requires_grad=True))

    def forward(self, X):
        X = X.reshape(-1, 784)
        # dot product instead of matmul coz of full connectivity
        H = relu(X@self.w1 + self.b1)
        return H@self.w2 + self.b2




def main():

    dataset_path = '../Linear_Neural_Networks/FashionMNIST/'
    batch_size = 128

    train_loader, test_loader = load_fashion_mnist(dataset_path, batch_size)

    # ########

    # creating a model having only 1 hiddean layer with 256 hidden units, and converting image into a flattened vector
    # of (1, image_rows/image_cols)
    # so, dimensions for layers are as follows:
    #
    #     input layer = 1*784
    #     hidden layer = 784*256
    #     output layer = 256*10

    # #########

    # creating model object, loss_fun and optimizer

    net = MLP()

    loss_fun = nn.CrossEntropyLoss()

    num_epochs = 10
    lr = 0.4

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_on_cpu(net, train_loader, test_loader, loss_fun, optimizer, num_epochs)

    # visualizing predictions

    for X, y in test_loader:
        break
    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(np.argmax(net(X).data.numpy(), axis=1))
    titles = [truelabel + '\n' + predlabel for truelabel, predlabel in zip(true_labels, pred_labels)]
    show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == "__main__":
    main()
