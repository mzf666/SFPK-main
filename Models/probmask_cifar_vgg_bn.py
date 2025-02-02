import torch.nn as nn
from Models.probmask_cifar_model_utils import get_builder


class VGG(nn.Module):

    def __init__(self, num_classes, builder, features):
        super(VGG, self).__init__()
        self.features = features
        self.linear = builder.conv1x1(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x.squeeze()


def make_layers(cfg, builder, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = builder.conv3x3(in_channels, v)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, eps=1e-5), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(num_classes, cfg, batch_norm, builder):
    model = VGG(
        num_classes, builder, make_layers(cfgs[cfg], builder, batch_norm=batch_norm)
    )
    return model


def vgg19_bn(input_shape, num_classes):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(num_classes, "E", True, get_builder())
