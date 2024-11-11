from torchsummary import summary
import torch.nn as nn

bottleneck_conv_backbone = [
    # [in_channels, out_channels, stride, padding, t, repeat_n]
    [32, 16, 1, 1, 1, 1],
    [16, 24, 2, 1, 6, 2],
    [24, 32, 2, 1, 6, 3],
    [32, 64, 2, 1, 6, 4],
    [64, 96, 1, 1, 6, 3],
    [96, 160, 2, 1, 6, 3],
    [160, 320, 1, 1, 6, 1]
]


class DefaultConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=3, padding=0):
        super(DefaultConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      stride=stride, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.layer(x)


class BottleneckConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=3, padding=0, t=6,
                 residual=False):
        super(BottleneckConvLayer, self).__init__()
        self.residual = residual
        self.expansion_chls_n = round(in_channels * t)
        
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.expansion_chls_n,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.expansion_chls_n),
            nn.ReLU6()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_chls_n,
                      out_channels=self.expansion_chls_n,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=self.expansion_chls_n,
                      bias=False),
            nn.BatchNorm2d(num_features=self.expansion_chls_n),
            nn.ReLU6()
        )

        self.downsample_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_chls_n,
                      out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, input):
        residual = input.detach().clone()
        x = self.upsample_conv(input)
        x = self.depthwise(x)
        x = self.downsample_conv(x)
        if self.residual:
            return x + residual
        else:
            return x


class MobileNetV2(nn.Module):
    def __init__(self, alpha=1, backbone=bottleneck_conv_backbone,
                 n_classes=1000):
        super(MobileNetV2, self).__init__()
        self.alpha = alpha
        self.full_conv = DefaultConvLayer(
            in_channels=3, out_channels=round(32 * alpha), padding=1
        )
        self.bottleneck_conv_net = nn.Sequential(
            *self.create_bottleneck_conv_list(backbone=backbone)
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(round(alpha*320), round(alpha*1280),
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(round(alpha*1280)),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.output_layer = nn.Conv2d(round(alpha*1280), n_classes,
                                      kernel_size=(1, 1))
        
    def create_bottleneck_conv_list(self, backbone):
        bottleneck_conv_list = []
        for layer in backbone:
            for idx in range(layer[-1]):
                in_channels = round(layer[0] * self.alpha)
                out_channels = round(layer[1] * self.alpha)
                bottleneck_conv_list.append(
                    BottleneckConvLayer(
                        in_channels=in_channels if idx == 0
                            else out_channels,
                        out_channels=out_channels,
                        stride=layer[2] if idx == 0 else 1,
                        padding=layer[3], t=layer[4],
                        residual=False if idx == 0 else True)
                )

        return bottleneck_conv_list

    def forward(self, input):
        x = self.full_conv(input)
        x = self.bottleneck_conv_net(x)
        x = self.pointwise_conv(x)
        x = self.avgpool(x)
        return self.output_layer(x)
         
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


if __name__ == "__main__":
    model = MobileNetV2(n_classes=10)
    summary(model=model, input_size=(3, 224, 224))