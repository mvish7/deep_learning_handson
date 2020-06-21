import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


def load_fashion_mnist(path, batch_size):
    """
    returns the torch dataloader for train and test dataset of FashionMNIST
    :param path:
    :param batch_size:
    :return:
    """
    # define the transform
    to_tensor = transforms.ToTensor()

    # downloading dataset
    mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, transform=to_tensor, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, transform=to_tensor, download=True)

    # creating dataloader

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def relu(X):
    return torch.max(X, torch.zeros_like(X))


def get_fashion_mnist_labels(labels):

    labels_list = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels_list[i] for i in labels]


def show_fashion_mnist(images, labels):
    """
    visualizing function for image dataset
    :param images:
    :param labels:
    :return:
    """
    # d2l.use_svg_display()
    nrows = len(images)
    fig, ax_arr = plt.subplots(1, nrows)
    for f, img, lbl in zip(ax_arr.flatten(), images, labels):
        f.imshow(img.reshape((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


def l2_penalty(w):
    return torch.sum(torch.mul(w, w))/2


def dropout(X, drop_prob):
    """
    function implements (inverted) dropout
    :param X:  tensor representing value of a layer
    :param drop_prob:  hyperparameter P
    :return:
    """
    assert 0 <= drop_prob <= 1

    if drop_prob == 1:
        return torch.zeros_like(X)

    # mask = torch.tensor(np.random.binomial(1, drop_prob, X.shape), dtype=torch.float32)

    mask = (torch.FloatTensor(X.shape).uniform_(0, 1) > drop_prob).float()
    return mask * X / (1-drop_prob)

