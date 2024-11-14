import torch
import torch.nn as nn
from torchsummary import summary 


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
    def __init__(self, in_channels, out_channels, hidden_dim,
                 stride=2, kernel_size=3, padding=0,
                 residual=False, se=True, non_lin='RE'):
        super(BottleneckConvLayer, self).__init__()
        self.residual = residual
        self.hidden_dim = hidden_dim
        
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.hidden_dim,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.hidden_dim),
            nn.ReLU6() if non_lin == 'RE' else h_sigmoid()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim,
                      out_channels=self.hidden_dim,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=self.hidden_dim,
                      bias=False),
            nn.BatchNorm2d(num_features=self.hidden_dim),
            nn.ReLU6() if non_lin == 'RE' else h_sigmoid()
        )

        self.downsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim,
                      out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        if se:
            self.se_layer = SELayer(self.hidden_dim)

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


class DefaultConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=3,
                 padding=0, bn=True, non_lin=None):
        super(DefaultConvLayer, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                bias=not bn
            )
        ]
        
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        
        if non_lin == 'HS':
            layers.append(h_swish())

        self.layer = nn.Sequential(*[layer for layer in layers
                                     if layer is not None])

    def forward(self, x):
        return self.layer(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_classes, net_config, alpha=1, is_small=False):
        super(MobileNetV3, self).__init__()
        self.alpha = alpha
        self.n_classes = n_classes
        self.is_small = is_small
        self.conv1 = DefaultConvLayer(
            in_channels=3, out_channels=_make_divisible(16 * self.alpha, 8), stride=2,
            padding=1, non_lin='HS')

        self.bottleneck_convs = nn.Sequential(
            *self.create_bottleneck_conv_list(net_config)
        )
        
        self.conv2 = self.create_second_def_conv(is_small)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = self.create_classifier(is_small)

    def create_classifier(self, is_small):
        output_options = {'large': 1280, 'small': 1024}
        input_options = {'large': 960, 'small': 576}
        mode = "small" if is_small else "large"
        
        out_channels = _make_divisible(
            output_options[mode] * self.alpha, 8)
        in_channels = _make_divisible(
            input_options[mode] * self.alpha, 8)
        
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, self.n_classes),
        )

    def create_second_def_conv(self, is_small):
        if not is_small:
            in_channels = _make_divisible(160 * self.alpha, 8)
            out_channels = _make_divisible(960 * self.alpha, 8)
            return DefaultConvLayer(
                in_channels=in_channels,
                out_channels=out_channels, kernel_size=1,
                stride=1, non_lin='HS')
        else:
            in_channels = _make_divisible(96 * self.alpha, 8)
            out_channels = _make_divisible(576 * self.alpha, 8)
            return DefaultConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels, kernel_size=1,
                    stride=1, non_lin='HS')

    def create_bottleneck_conv_list(self, backbone):
        bottleneck_conv_list = []
        for layer in backbone:
            in_channels = _make_divisible(layer[0] * self.alpha, 8)
            out_channels = _make_divisible(layer[1] * self.alpha, 8)
            hidden_dim = _make_divisible(layer[2] * self.alpha, 8)
            is_residual = True if layer[4] == 1 and in_channels == out_channels else False
            bottleneck_conv_list.append(
                BottleneckConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=layer[3],
                    hidden_dim=hidden_dim,
                    stride=layer[4],
                    padding=layer[5],
                    residual=is_residual, se=layer[6],
                    non_lin=layer[-1])
            )
        return bottleneck_conv_list
            
    def forward(self, input):
        x = self.conv1(input)
        x = self.bottleneck_convs(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


v3_large_config = [
    # in_channels, out_channels, exp_channels, kernel_size, stride, padding, se, nl
    [16, 16, 16, 3, 1, 1, False, 'RE'],
    [16, 24, 64, 3, 2, 1, False, 'RE'],
    [24, 24, 72, 3, 1, 1, False, 'RE'],
    [24, 40, 72, 5, 2, 2, True, 'RE'],
    [40, 40, 120, 5, 1, 2, True, 'RE'],
    [40, 40, 120, 5, 1, 2, True, 'RE'],
    [40, 80, 240, 3, 2, 1, False, 'HS'],
    [80, 80, 200, 3, 1, 1, False, 'HS'],
    [80, 80, 184, 3, 1, 1, False, 'HS'],
    [80, 80, 184, 3, 1, 1, False, 'HS'],
    [80, 112, 480, 3, 1, 1, True, 'HS'],
    [112, 112, 672, 3, 1, 1, True, 'HS'],
    [112, 160, 672, 5, 2, 2, True, 'HS'],
    [160, 160, 960, 5, 1, 2, True, 'HS'],
    [160, 160, 960, 5, 1, 2, True, 'HS'],

]

v3_small_config = [
    # in_channels, out_channels, exp_channels, kernel_size, stride, padding, se, nl
    [16, 16, 16, 3, 2, 1, True, 'RE'],
    [16, 24, 72, 3, 2, 1, False, 'RE'],
    [24, 24, 88, 3, 1, 1, False, 'RE'],
    [24, 40, 96, 5, 2, 2, True, 'HS'],
    [40, 40, 240, 5, 1, 2, True, 'HS'],
    [40, 40, 240, 5, 1, 2, True, 'HS'],
    [40, 48, 120, 5, 1, 2, True, 'HS'],
    [48, 48, 144, 5, 1, 2, True, 'HS'],
    [48, 96, 288, 5, 2, 2, True, 'HS'],
    [96, 96, 576, 5, 1, 2, True, 'HS'],
    [96, 96, 576, 5, 1, 2, True, 'HS']
]


if __name__ == '__main__':
    model1 = MobileNetV3(10, net_config=v3_small_config, is_small=True)
    summary(model=model1, input_size=(3, 224, 224))

    model2 = MobileNetV3(10, net_config=v3_large_config, is_small=False)
    summary(model=model2, input_size=(3, 224, 224))