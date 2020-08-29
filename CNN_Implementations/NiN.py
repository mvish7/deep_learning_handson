import torch
import torch.nn as nn


class NiN:
    def __init__(self, num_classes):
        super(NiN, self).__init__()
        self.num_classes = num_classes

    def construct_nin_block(self, num_in_channel, num_out_channel, kernel_size, stride, padding):

        blk = nn.Sequential(
            nn.Conv2d(num_in_channel, num_out_channel, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(num_out_channel, num_out_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_out_channel, num_out_channel, kernel_size=1),
            nn.ReLU(),
        )

        return blk

    def build_model(self):
        model = nn.Sequential(
            self.construct_nin_block(1, 96, kernel_size=11, stride=4, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),

            self.construct_nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            self.construct_nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2),

            nn.Dropout2d(0.5),

            self.construct_nin_block(384, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()  # converting 4D tensor to 1D
        )

        return model
