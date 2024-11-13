import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BottleneckConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=3, padding=0, t=6,
                 residual=False, se=True, non_lin='RE'):
        super(BottleneckConvLayer, self).__init__()
        self.residual = residual
        self.expansion_chls_n = round(in_channels * t)
        
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.expansion_chls_n,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.expansion_chls_n),
            nn.ReLU6() if non_lin == 'RE' else h_sigmoid()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_chls_n,
                      out_channels=self.expansion_chls_n,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=self.expansion_chls_n,
                      bias=False),
            nn.BatchNorm2d(num_features=self.expansion_chls_n),
            nn.ReLU6() if non_lin == 'RE' else h_sigmoid()
        )

        self.downsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_chls_n,
                      out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        if se:
            self.se_layer = SELayer(self.expansion_chls_n)

        self.se = se

    def forward(self, input):
        residual = input.detach().clone()
        x = self.upsample_conv(input)
        x = self.depthwise(x)
        if self.se:
            x = self.se_layer(x)
        x = self.downsample_conv(x)
        if self.residual:
            return x + residual
        else:
            return x

