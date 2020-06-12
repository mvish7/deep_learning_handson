import torch
import torchvision
import numpy as np


def grad_descent(params, lr, batch_size):
    """
    function implements gradient descent from scratch
    :param params: params of the nw
    :param lr: learning rate
    :param batch_size:  num of examples in a training batch
    :return:
    """
    # w.data = lr/batch_size(w.data - w.grad)  same for bias
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
    return params


def softmax(X):
    """
    function to perform softmax operation from scratch
    :param X: batch of n input images
    :return: returns the softmax applied input
    """
    numerator = torch.exp(X)
    denominator = torch.sum(X, dim=1, keepdim=True)
    return numerator/denominator


def cross_entropy(y_hat, y):
    """
    function calculates the cross entropy loss or likelyhood loss
    :param y_hat: nw predicted output
    :param y: labels
    :return: returns the cross entropy loss value
    """
    # loss = - torch.sum(torch.matmul(y.float(), y_hat))

    # the d2l way of finding loss
    # loss = -torch.gather(y_hat, 1, y.unsqueeze(dim=1)).log()
    loss_fun = torch.nn.CrossEntropyLoss()
    loss = loss_fun(y_hat, y)
    return loss


def evaluate_accuracy(data_iter, net):
    """
    function to run over test data and find accuracy
    :param data_iter:
    :param net:
    :return:
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).sum().item()
        n += y.size()[0]  # y.size()[0] = batch_size
    return acc_sum / n


def train_on_cpu(net, train_loader, test_loader, loss_fun, optimizer, epochs):
    """
    train_ch3 equivalent of d2l

    :param net:
    :param train_loader:
    :param test_loader:
    :param loss_fun:
    :param optimizer:
    :param epochs:
    :return:
    """
    train_loss, train_acc, n = 0.0, 0.0, 0

    for i in range(epochs):
        for X, y in train_loader:
            optimizer.zero_grad()

            y_hat = net(X)

            loss_val = loss_fun(y_hat, y)
            loss_val.backward()
            optimizer.step()

            y = y.type(torch.float32)
            train_loss += loss_val.item()
            train_acc += torch.sum((torch.argmax(y_hat, dim=1).type(torch.FloatTensor) == y).detach()).float()
            n += list(y.size())[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' \
              % (i + 1, train_loss / n, train_acc / n, test_acc))


def linear_regression(X, weight, bias):
    return torch.matmul(X, weight) + bias


def squared_loss(y_pred, y_label):
    return (y_pred - y_label.reshape(y_pred.size())) ** 2 / 2


def grad_descent(params, lr, batch_size):
    # w.data = lr/batch_size(w.data - w.grad)  same for bias
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
