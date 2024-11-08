import torch as torch
import torch.nn as nn
import torch.optim as optim

dws_convolutions_backbone = [
    # [in_channels, out_channels, kernel_size, stride, padding, repeat_n]
    [32, 64, 3, 1, 1, 1],
    [64, 128, 3, 2, 1, 1],
    [128, 128, 3, 1, 1, 1],
    [128, 256, 3, 2, 1, 1],
    [256, 256, 3, 1, 1, 1],
    [256, 512, 3, 2, 1, 1],
    [512, 512, 3, 1, 1, 5],
    [512, 1024, 3, 2, 1, 1],
    [1024, 1024, 3, 2, 4, 1],
]


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


class MobileNetV1(nn.Module):
    def __init__(self, alpha=1, backbone=dws_convolutions_backbone,
                 n_classes=1000):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha
        self.full_conv = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2,
            padding=1
        )
        self.dws_conv_net = nn.ModuleList(
            self.create_dws_conv_list(backbone=backbone)
        )
        self.pool = nn.AvgPool2d(
            (7, 7)
        )
        self.fc = nn.Linear(
            1024, n_classes
        )

    def create_dws_conv_list(self, backbone):
        dws_conv_list = []
        for layer in backbone:
            for repeat in range(layer[-1]):
                dws_conv_list.append(DefaultConvLayer(
                    in_channels=layer[0], out_channels=layer[1],
                    alpha=self.alpha, stride=layer[3],
                    kernel_size=layer[2], padding=layer[4])
                )

        return dws_conv_list

    def forward(self, input):
        x = self.full_conv(input)
        x = self.dws_conv_net(x)
        x = self.pool(x).view(-1, 1024)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)

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


def get_optimizer_with_dw_decay(model: torch.nn.Module, base_lr,
                                base_weight_decay,
                                dw_weight_decay):
    dw_conv_params = []
    other_params = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d) \
            and module.groups == module.in_channels \
            and module.groups == module.out_channels:
            dw_conv_params.extend(module.parameters())
        else:
            other_params.extend(module.parameters())

    return optim.Adam([
        {"params": dw_conv_params, "weight_decay": dw_weight_decay},
        {"params": other_params, "weight_decay": base_weight_decay}
    ], lr=base_lr)


if __name__ == "__main__":
    model = MobileNetV1(n_classes=10)
    optimizer = get_optimizer_with_dw_decay(model, base_lr=0.001,
                                            base_weight_decay=1e-4,
                                            dw_weight_decay=0.0)