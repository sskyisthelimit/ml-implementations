import torch
import torch.nn as nn

# Figure 10 in paper - staring from left , from bottom to top
part_1_architecture = [
    # in_channels, out_channels, kernel_size, stride, padding
    [3, 32, 3, 2, 0],
    [32, 32, 3, 1, 0],
    [32, 64, 3, 1, 1],
]

part_2_architecture = [
    # in_channels, out_channels, kernel_size, stride, padding
    [64, 80, 3, 1, 0],
    [80, 192, 3, 1, 0]
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)


# Inception block that described in figure 5 of InceptionV2 paper
class InceptionFig5(nn.Module):
    def __init__(self, in_channels):
        
        super(InceptionFig5, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 1),
            ConvBlock(96, 96, 3, 1, 1)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 48, 1, 1, 0),
            ConvBlock(48, 64, 3, 1, 1)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(in_channels, 64, 1, 1, 0)
        )

        self.branch4 = ConvBlock(in_channels, 64, 1, 1, 0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionFig6(nn.Module):
    def __init__(self, in_channels, f_7x7):
        
        super(InceptionFig6, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, 1, 1, 0),
            ConvBlock(f_7x7, f_7x7, (1, 7), 1, (0, 3)),
            ConvBlock(f_7x7, f_7x7, (7, 1), 1, (3, 0)),
            ConvBlock(f_7x7, f_7x7, (1, 7), 1, (0, 3)),
            ConvBlock(f_7x7, 192, (7, 1), 1, (3, 0)))

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, 1, 1, 0),
            ConvBlock(f_7x7, f_7x7, (1, 7), 1, (0, 3)),
            ConvBlock(f_7x7, 192, (7, 1), 1, (3, 0)))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1), 
            ConvBlock(in_channels, 192, 1, 1, 0)
        )

        self.branch4 = ConvBlock(in_channels, 192, 1, 1, 0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionFig7(nn.Module):
    def __init__(self, in_channels):
        
        super(InceptionFig7, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size=1, stride=1, padding=0),
            ConvBlock(448, 384, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.branch1_top = ConvBlock(384, 384, kernel_size=(1, 3),
                                     stride=1, padding=(0, 1))
        self.branch1_bot = ConvBlock(384, 384, kernel_size=(3, 1),
                                     stride=1, padding=(1, 0))
        
        self.branch2 = ConvBlock(in_channels, 384, kernel_size=1,
                                 stride=1, padding=0)
        self.branch2_top = ConvBlock(384, 384, kernel_size=(1, 3),
                                     stride=1, padding=(0, 1))
        self.branch2_bot = ConvBlock(384, 384, kernel_size=(3, 1),
                                     stride=1, padding=(1, 0))
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 320, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b1 = torch.cat([self.branch1_top(b1), self.branch1_bot(b1)], 1)
        
        b2 = self.branch2(x)
        b2 = torch.cat([self.branch2_top(b2), self.branch2_bot(b2)], 1)
        
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        return torch.cat([b1, b2, b3, b4], 1)
 

# class AuxiliaryClsf(nn.Module):
#     def __init__(self, in_channels, n_classes):
#         super(AuxiliaryClsf, self).__init__()

#         self.clsf = nn.Sequential(
#             nn.AdaptiveAvgPool2d((4, 4)),
#             ConvBlock(in_channels, 128, 1, 1, 0), 
#             nn.Linear(128*16, 1024),
#             nn.Dropout(0.7),
#             nn.Linear(1024, n_classes))
        
#     def forward(self, x):
#         return self.clsf(x)


class ReductionBlock(nn.Module):
    def __init__(self, in_channels, f_3x3_r, add_ch=0):
        super(ReductionBlock, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 178 + add_ch,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(178 + add_ch, 178 + add_ch,
                      kernel_size=3, stride=2, padding=0)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 302 + add_ch,
                      kernel_size=3, stride=2, padding=0)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=0)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1)


class InceptionV2(nn.Module):
    def __init__(self, in_channels, n_classes):
        
        super(InceptionV2, self).__init__()
        
        self.conv_part1 = self._build_conv_part(part_1_architecture)
        self.pool1 = nn.MaxPool2d(3, 2, 0)
        self.conv_part2 = self._build_conv_part(part_2_architecture)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.inception_5a = InceptionFig5(192)
        self.inception_5b = InceptionFig5(288)
        self.inception_5c = InceptionFig5(288)

        self.reduction1 = ReductionBlock(288, f_3x3_r=64, add_ch=0)

        self.inception_6a = InceptionFig6(768, f_7x7=128)
        self.inception_6b = InceptionFig6(768, f_7x7=160)
        self.inception_6c = InceptionFig6(768, f_7x7=160)
        self.inception_6d = InceptionFig6(768, f_7x7=160)
        self.inception_6e = InceptionFig6(768, f_7x7=192)

        self.reduction2 = ReductionBlock(768, f_3x3_r=192, add_ch=16)

        self.inception_7a = InceptionFig7(1280)
        self.inception_7b = InceptionFig7(2048)

        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def _build_conv_part(self, architecture):
        layers = []
        for in_channels, out_channels, kernel_size, \
                stride, padding in architecture:
            
            layers.append(
                ConvBlock(in_channels, out_channels,
                          kernel_size, stride, padding)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_part1(x)
        x = self.pool1(x)
        x = self.conv_part2(x)
        x = self.pool2(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.inception_5c(x)

        x = self.reduction1(x)

        x = self.inception_6a(x)
        x = self.inception_6b(x)
        x = self.inception_6c(x)
        x = self.inception_6d(x)
        x = self.inception_6e(x)

        x = self.reduction2(x)

        x = self.inception_7a(x)
        x = self.inception_7b(x)

        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.softmax(x)
        return x


model = InceptionV2(in_channels=3, n_classes=10)
x = torch.randn(1, 3, 299, 299)  
output = model(x)
print(output.shape) 