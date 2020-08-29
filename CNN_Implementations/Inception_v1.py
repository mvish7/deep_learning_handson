import torch
import torch.nn as nn
from torch.nn import functional as F


class InceptionBlock(nn.Module):
    """
    Inception block has 4 parallel paths
    The first three paths use convolutional layers with window sizes of  1×1 ,  3×3 , and  5×5  to extract
    information from different spatial sizes. The middle two paths perform a  1×1  convolution on the input to
    reduce the number of channels, reducing the model’s complexity. The fourth path uses a  3×3  maximum pooling
    layer, followed by a  1×1  convolutional layer to change the number of channels.

    :param in_channel:
    :param op_channel1:
    :param op_channel2:
    :param op_channel3:
    :param op_channel4:
    :return:
    """
    def __init__(self, in_channel, op_channel1, op_channel2, op_channel3, op_channel4):
        super(InceptionBlock, self).__init__()

        # path 1
        self.p1 = nn.Conv2d(in_channel, op_channel1, kernel_size=1)

        # path 2
        self.p2_1 = nn.Conv2d(in_channel, op_channel2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(op_channel2[0], op_channel2[1], kernel_size=3, padding=1)

        # path 3
        self.p3_1 = nn.Conv2d(in_channel, op_channel3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(op_channel3[0], op_channel3[1], kernel_size=5, padding=2)

        # path 4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channel, op_channel4, kernel_size=1)

    def forward(self, X):
        path_1 = F.relu(self.p1(X), inplace=True)

        path_2_1 = F.relu(self.p2_1(X), inplace=True)
        path_2_2 = F.relu(self.p2_2(path_2_1), inplace=True)

        path_3_1 = F.relu(self.p3_1(X), inplace=True)
        path_3_2 = F.relu(self.p3_2(path_3_1), inplace=True)

        path_4_1 = F.relu(self.p4_1(X), inplace=True)
        path_4_2 = F.relu(self.p4_2(path_4_1), inplace=True)

        return torch.cat((path_1, path_2_2, path_3_2, path_4_2), dim=1)


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()

    def build_model(self):

        blk1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blk2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blk3 = nn.Sequential(
            InceptionBlock(192, 64,  (96, 128), (16, 32), 32),
            InceptionBlock(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blk4 = nn.Sequential(
            InceptionBlock(480, 192, (96, 208), (16, 48), 64),
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),
            InceptionBlock(512, 112, (144, 288), (32, 64), 64),
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blk5 = nn.Sequential(
            InceptionBlock(832, 256, (160, 320), (32, 128), 128),
            InceptionBlock(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )

        net = nn.Sequential(blk1, blk2, blk3, blk4, blk5)

        return net





