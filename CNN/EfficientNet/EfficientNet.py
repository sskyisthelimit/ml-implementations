import torch 
import torch.nn as nn
from math import ceil

base_arch = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

factors_val = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, groups=1):
        # groups - for depth-wise convolution, if groups=in_channels
        super(BasicBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class SEblock(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(SEblock, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        return  x * self.se(x)
    

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, 
                 expand_ratio, shrink_ratio=4, survival_prob=0.8):
        super(MBConvBlock, self).__init__()

        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        
        interm_channels = in_channels * expand_ratio
        reduced_channels = int(in_channels / shrink_ratio)
        self.expand = in_channels != interm_channels
        if self.expand:
            self.expand_conv = BasicBlock(in_channels, interm_channels,
                                          kernel_size=3, stride=1, padding=1)
        
        self.interm_conv = nn.Sequential(
            BasicBlock(interm_channels, interm_channels,
                       kernel_size, stride, padding, groups=interm_channels),
            SEblock(interm_channels, reduced_channels),
            nn.Conv2d(interm_channels, out_channels, 1, bias=False),  
            nn.BatchNorm2d(out_channels)
        )
            
    def stochastic_depth(self, x):
        if self.training:
            tnsr = (
                torch.rand(x.shape[0], 1, 1, 1,
                           device=x.device) < self.survival_prob
            )
            return torch.div(x, self.survival_prob) * tnsr
        else:
            return x
        
    def forward(self, input):
        x = self.expand_conv(input) if self.expand else input
        if self.use_residual:
            return self.stochastic_depth(self.interm_conv(x)) + input
        else:
            return self.interm_conv(x)


class EfficientNet(nn.Module):
    def __init__(self, net_version, n_classes):
        super(EfficientNet, self).__init__()
        
        (depth_factor,
         width_factor, drop_factor) = self.calc_factors(net_version)
        self.last_channels = ceil(1280 * width_factor)
        
        self.features = self.create_features(width_factor, depth_factor)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop_factor),
            nn.Linear(self.last_channels, n_classes)
        )
        
    def calc_factors(self, net_version, alpha=1.2, beta=1.1):
        phi, resol, dropout = factors_val[net_version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return depth_factor, width_factor, dropout
        
    def create_features(self, width_factor, depth_factor):
        channels = int(32 * width_factor)
        features = [BasicBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_arch:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for repeat in range(layers_repeats):
                features.append(
                    MBConvBlock(in_channels, out_channels, kernel_size,
                                stride=stride if repeat == 0 else 1, 
                                padding=kernel_size // 2,
                                expand_ratio=expand_ratio)
                )
                in_channels = out_channels
        
        features.append(
            BasicBlock(in_channels, self.last_channels, 1, 1, 0)
        )

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def test():
    version = 'b0'
    n_classes = 10
    phi_value, resolution, drop_rate = factors_val[version]
    x = torch.randn(2, 3, resolution, resolution)
    model = EfficientNet(version, n_classes)
    prediction = model(x)
    print(prediction.shape)

test()
