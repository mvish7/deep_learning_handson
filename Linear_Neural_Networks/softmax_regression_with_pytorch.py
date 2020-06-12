import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/flying-dutchman/PycharmProjects/')
from dl_functions import train_on_cpu


# ############ set of functions necessary for script

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, X):
        X = X.reshape(-1, 784)
        return self.linear(X)


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



def main():

    # handling dataset and creating dataloader

    # define the transform
    to_tensor = transforms.ToTensor()

    # downloading dataset
    mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=to_tensor, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=to_tensor, download=True)

    # creating dataloader

    train_loader = DataLoader(mnist_train, batch_size=10, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=10, shuffle=True)

    # #### visualizing data

    # imgs = []
    # labels = []
    #
    # for i in range(10):
    #     rnd_int = np.random.randint(len(mnist_train))
    #     imgs.append(mnist_train[rnd_int][0])  # image
    #     labels.append(mnist_train[rnd_int][1])  # label
    #
    # show_fashion_mnist(imgs, get_fashion_mnist_labels(labels))

    num_inputs = 784  # fashionmnist has 28*28 size images
    num_outputs = 10  # output classes are 10
    lr = 0.1
    epochs = 5
    batch_size = 10

    model = LogisticRegression(num_inputs, num_outputs)

    loss_fun = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_on_cpu(model, train_loader, test_loader, loss_fun, optimizer, epochs)







if __name__ == "__main__":
    main()

