import torch
import torch.nn as nn


CFG = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x

        # Keep the CIFAR variant simple: global pooling to 1x1 before FC.
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg(name, num_classes, device):
    if name not in CFG:
        raise ValueError('Invalid VGG model name {}!'.format(name))
    model = VGG(CFG[name], num_classes=num_classes)
    model = model.to(device)
    return model
