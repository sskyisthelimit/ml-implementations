import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_repeats, use_residual=True):
        super(ResidualBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.n_repeats = n_repeats

        for _ in range(n_repeats):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(in_channels, in_channels // 2,
                              kernel_size=1, stride=1),
                    ConvBlock(in_channels // 2, in_channels,
                              kernel_size=3, stride=1, padding=1)
                )
            ) 
        self.use_residual = use_residual

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x
    

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors):
        super(ScalePrediction, self).__init__()

        self.layer = nn.Sequential(
            ConvBlock(in_channels, in_channels*2, kernel_size=3,
                      stride=1, padding=1),
            
            nn.Conv2d(in_channels*2, n_anchors * (n_classes+5),
                      kernel_size=1, stride=1)
        )
        self.n_classes = n_classes
        self.n_anchors = n_anchors

    def forward(self, x):
        pred = self.layer(x)
        return pred.reshape(x.shape[0], self.n_anchors, self.n_classes + 5,
                            x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, n_classes=80):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchors = 3
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.n_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                n_repeats = module[1]
                layers.append(ResidualBlock(in_channels, n_repeats=n_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels,
                                      use_residual=False, n_repeats=1),
                        ConvBlock(in_channels, in_channels // 2,
                                  kernel_size=1),
                        ScalePrediction(in_channels // 2,
                                        n_classes=self.n_classes,
                                        n_anchors=self.n_anchors),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

if __name__ == "__main__":
    n_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(n_classes=n_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32,
                                 IMAGE_SIZE//32, n_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16,
                                 IMAGE_SIZE//16, n_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8,
                                 IMAGE_SIZE//8, n_classes + 5)