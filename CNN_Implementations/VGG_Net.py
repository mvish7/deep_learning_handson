import torch
import torch.nn as nn


class VGGNet:
    def __init__(self, model_arch):
        super(VGGNet, self).__init__()
        self.model_arch = model_arch

    def build_model(self):
        feature_extractor = []
        in_channels = 1
        for blk in self.model_arch:
            feature_extractor.append(nn.Sequential(self.construct_vgg_block(blk[0], in_channels, blk[1])))
            in_channels = blk[1]  # in_channels = out_channels for next iterations

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10),
        )

        return nn.Sequential(*feature_extractor, *classifier)

    def construct_vgg_block(self, num_conv_layers, in_channels, out_channels):
        layers = []

        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        vgg_block = nn.Sequential(*layers)
        return vgg_block
