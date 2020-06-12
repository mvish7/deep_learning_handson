import numpy
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
# import d2l
from torch.utils.data import DataLoader
import sys
from torch.distributions import normal
import sys
sys.path.append('/home/flying-dutchman/PycharmProjects/')
from dl_functions import softmax, cross_entropy, grad_descent, evaluate_accuracy

to_tensor = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=to_tensor, target_transform=None,
                                                download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=to_tensor, target_transform=None,
                                               download=True)
print(len(mnist_train))
print(len(mnist_test))


def get_fashion_mnist_labels(labels):
    labels_list = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels_list[i] for i in labels]


def show_fashion_mnist(images, labels):
    # d2l.use_svg_display()
    nrows = len(images)
    fig, ax_arr = plt.subplots(1, nrows)
    for f, img, lbl in zip(ax_arr.flatten(), images, labels):
        f.imshow(img.reshape((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


def net(X, W, b):
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), W) + b)


def train(net, train_loader, test_loader, epochs, lr, batch_size, W, b):
    """
    equivalent of train_ch3 from d2l

    :param net: the network object
    :param train_loader: data loader for training data set
    :param test_loader:  data loader for test data
    :param epochs: num of epochs
    :param lr: learning rate
    :param batch_size: batch size
    :param W: weights of the network
    :param b: bias
    :return:
    """
    for epoch in range(epochs):
        sum_train_l, train_accuracy, n = 0.0, 0.0, 0
        for X, y in train_loader:
            y_pred = net(X, W, b)
            loss_val = cross_entropy(y_pred, y).sum()
            loss_val.backward()
            W, b = grad_descent([W, b], lr, batch_size)
            sum_train_l = sum_train_l + loss_val.item()
            train_accuracy += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.size()[0]
        test_acc = evaluate_accuracy(test_loader, net, W, b)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, sum_train_l / n, train_accuracy / n, test_acc))

imgs = []
labels = []

for i, data in enumerate(mnist_train):
    if i < 10:
        imgs.append(data[0])
        labels.append(data[1])

    else:
        break

show_fashion_mnist(imgs, get_fashion_mnist_labels(labels))

# ## creating data loader

batch_size = 10
# if sys.platform.startswith("win"):
#     num_workers = 0
# else:
#     num_workers = 4

train_loader = DataLoader(mnist_train, batch_size, True)
test_loader = DataLoader(mnist_test, batch_size, True)

num_inputs = 784
num_outputs = 10

W = normal.Normal(loc=0, scale=0.01).sample((num_inputs, num_outputs))
b = torch.zeros(num_outputs)

W.requires_grad_(True)
b.requires_grad_(True)

lr = 0.1
epochs = 5

train(net, train_loader, test_loader, epochs, lr, batch_size, W, b)


