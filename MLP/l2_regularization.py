""""
Studying the effects of l2 regularization
"""
import torch
import torch.nn as nn
import torch.utils.data

import sys
sys.path.append('/home/flying-dutchman/PycharmProjects')

import d2l
from dl_functions import l2_penalty, linear_regression, squared_loss, grad_descent

# ####### module functions


def init_params(num_inputs):
    w = torch.randn(size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros((1,), requires_grad=True)
    return [w, b]


def fit_n_plot(lambd, num_inputs, num_epochs, train_iter, net, loss, grad_descent, lr, batch_size, train_features, train_labels,
               test_features, test_labels):
    w, b = init_params(num_inputs)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                y_pred = net(X, w, b)
                loss_val = loss(y_pred, y).sum() + lambd * l2_penalty(w)
            loss_val.backward()
            grad_descent([w, b], lr, batch_size)

        with torch.no_grad():
            train_ls.append(torch.mean(loss(net(train_features, w, b), train_labels)).item())
            test_ls.append(torch.mean(loss(net(test_features, w, b), test_labels)).item())

    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('l2 norm of w:', torch.norm(w).item())


def fit_n_plot_with_pytorch(lambd, num_inputs, num_outputs, lr, num_epochs, train_iter, train_features, train_labels,
                            test_features, test_labels):
    net = torch.nn.Linear(num_inputs, num_outputs)
    loss = torch.nn.MSELoss()
    for param in net.parameters():
        param.data.uniform_()

    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=lambd)

    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                optimizer.zero_grad()
                y_pred = net(X)
                loss_val = loss(y_pred, y)
            loss_val.backward()
            optimizer.step()

        train_ls.append(torch.mean(loss(net(train_features),
                                            train_labels)).item())
        test_ls.append(torch.mean(loss(net(test_features),
                                           test_labels)).item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                     range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.norm().item())


# ###### test section

def main():
    n_train, n_test, num_inputs = 50, 100, 150

    # creating ground truth weight and bias
    true_w = torch.ones(size=(num_inputs, 1)) *0.01
    true_b = torch.tensor(0.05)

    features = torch.randn(size=(n_train+n_test, num_inputs))
    labels = torch.matmul(features, true_w) + true_b
    labels.add_(torch.empty(size=labels.shape).normal_(0, std = 0.01))

    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # defining training and testing
    num_epochs, batch_size, lr = 100, 1, 0.003
    net = linear_regression
    loss = squared_loss

    # creating data iterator
    train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size,
                                             shuffle=True)

    # fit_n_plot(4, num_inputs, num_epochs, train_iter, net, loss, grad_descent, lr, batch_size, train_features, train_labels,
    #         test_features, test_labels)

    fit_n_plot_with_pytorch(2.8, num_inputs, 1, lr, num_epochs, train_iter, train_features, train_labels,
                           test_features, test_labels)


if __name__ == "__main__":
    main()


