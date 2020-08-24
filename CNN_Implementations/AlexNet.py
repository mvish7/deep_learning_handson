import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    formulae to calculate output of layers
    conv layer: (W-F + 2*P) / S + 1
    pooling layer: (W-F)/S + 1

    using these formulae calculate the output dimension of each layers
    """
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),  # accomodated from torchvision based alexnet implementation
            nn.Linear(6400, 4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),  # accomodated from torchvision based alexnet implementation
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes, bias=True)
        )

    def forward(self, X):
        extracted_features = self.feature_extractor(X)
        classified_output = self.classifier(extracted_features)
        return classified_output
