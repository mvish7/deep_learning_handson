"""
implementing fashion mnist classification using dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('/home/flying-dutchman/PycharmProjects')

import d2l
from dl_functions import train_on_cpu, load_fashion_mnist, dropout


# ######### module functions

class Net(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=10, num_hidden1=512, num_hidden2=256, drop_prob=0.8,
                 is_training=True):
        super(Net, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.is_training = is_training
        self.drop_prob = drop_prob

        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.reshape(-1, 784)
        H1 = self.relu(self.linear1(X))

        if self.is_training:
            H1 = dropout(H1, self.drop_prob)

        H2 = self.relu(self.linear2(H1))
        if self.is_training:
            H2 = dropout(H2, self.drop_prob)

        y_hat = self.linear3(H2)

        return y_hat

def main():

    # testing if dropout works

    # test_tensor = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    # print("original tensor", test_tensor)
    # print("dropout 40%", dropout(test_tensor, 0.4))
    # print("dropout 50%", dropout(test_tensor, 0.5))
    # print("dropout 70%", dropout(test_tensor, 0.7))
    # print("dropout 90%", dropout(test_tensor, 0.9))

    dataset_path = './Linear_Neural_Networks/'
    batch_size = 256

    train_loader, test_loader = load_fashion_mnist(dataset_path, batch_size)

    num_epochs = 10
    lr = 0.6

    net = Net(drop_prob=0.6)

    loss_fun = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_on_cpu(net, train_loader, test_loader, loss_fun, optimizer, num_epochs)


if __name__ == "__main__":
    main()