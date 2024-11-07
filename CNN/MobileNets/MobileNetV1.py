import torch.nn as nn


class DefaultConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1,
                 stride=2, kernel_size=3, padding=0):
        super(DefaultConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=round(in_channels * alpha),
                      out_channels=round(out_channels * alpha),
                      stride=stride, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=round(out_channels * alpha)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1,
                 stride=2, kernel_size=3, padding=0):
        super(DepthwiseSeparableConvLayer, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=round(in_channels * alpha),
                      out_channels=round(in_channels * alpha),
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=round(in_channels * alpha),
                      bias=False),
            nn.BatchNorm2d(num_features=round(in_channels * alpha)),
            nn.ReLU()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=round(in_channels * alpha),
                      out_channels=round(out_channels * alpha),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=round(out_channels * alpha)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
