import torch
import torch.nn as nn
import sys
from AlexNet import AlexNet
from VGG_Net import VGGNet
sys.path.append('/home/flying-dutchman/PycharmProjects')
from dl_functions import load_fashion_mnist, train_on_cpu


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def main():
    path_to_dataset = '/home/flying-dutchman/PycharmProjects/Linear_Neural_Networks/FashionMNIST/'
    batch_size = 128
    train_loader, test_loader = load_fashion_mnist(path_to_dataset, batch_size, resize=True)

    # ########## Alexnet
    # model = AlexNet(10)
    # print(model)

    # ########## VGG
    # The original VGG network had 5 convolutional blocks, among which the first two have one convolutional layer each,
    # the latter three contain two convolutional layers each. The first block has 64 output channels each subsequent blk
    # doubles the number of output channels, until that number reaches 512

    model_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # (num_conv_layers, out_channel)
    model_instance = VGGNet(model_arch)
    model = model_instance.build_model()

    # X = torch.randn(size=(1, 1, 224, 224))
    #
    # for layer in model:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape', X.shape)

    model.apply(init_weights)
    optimizer = torch.optim.SGD(model.parameters(), 0.001)
    loss_fun = torch.nn.CrossEntropyLoss()

    train_on_cpu(model, train_loader, test_loader, loss_fun, optimizer, 50)




if __name__ == "__main__":
    main()
