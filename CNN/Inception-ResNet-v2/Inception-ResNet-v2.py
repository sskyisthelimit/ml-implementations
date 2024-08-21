import torch
import torch.nn as nn

from CNN.InceptionV2.InceptionV2 import ConvBlock


class InceptionFig3(nn.Module):
    def __init__(self, in_channels=3):
        super(InceptionFig3, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, 32, 3, 2, 0),
            ConvBlock(32, 32, 3, 1, 0),
            ConvBlock(32, 64, 3, 1, 'same'),
        )
        self.conv2 = ConvBlock(64, 96, 3, 2, 'valid')
        self.pool1 = nn.MaxPool2d(3, 2, 0)

        self.conv3_b1 = nn.Sequential(
            ConvBlock(160, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 'valid')
        )

        self.conv3_b2 = nn.Sequential(
            ConvBlock(160, 64, 1, 1, 0),
            ConvBlock(64, 64, (7, 1), 1, (3, 0)),
            ConvBlock(64, 64, (1, 7), 1, (0, 3)),
            ConvBlock(64, 96, 3, 1, 'valid'))
        
        self.conv4 = ConvBlock(192, 192, 3, 2, 'valid')
        self.pool2 = nn.MaxPool2d(3, 2, 0)

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.conv2(x)
        p1 = self.pool1(x)
        x = torch.cat([c2, p1], dim=1)

        c3_b1 = self.conv3_b1(x)
        c3_b2 = self.conv3_b2(x)
        x = torch.cat([c3_b1, c3_b2], dim=1)

        c4 = self.conv4(x)
        p2 = self.pool2(x)
        return torch.cat([c4, p2], dim=1)


class InceptionFig16(nn.Module):
    def __init__(self, in_channels, upsample_x):
        super(InceptionFig16, self).__init__()

        self.branch1 = ConvBlock(in_channels, 32, 1, 1, 0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 32, 1, 1, 0),
            ConvBlock(32, 32, 3, 1, 'same')
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, 32, 1, 1, 0),
            ConvBlock(32, 48, 3, 1, 'same'),
            ConvBlock(48, 64, 3, 1, 'same')
        )

        self.final_conv = ConvBlock(128, 384, 1, 1, 0)

        self.upsample_x = upsample_x

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        conct = torch.cat([b1, b2, b3], dim=1)
        final = self.final_conv(conct)
        
        return torch.relu(self.upsample_x(x) + final)
    

class ReductionFig7(nn.Module):
    def __init__(self, in_channels, K, I, M, N):
        super(ReductionFig7, self).__init__()

        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = ConvBlock(in_channels, N, 3, 2, 'valid')
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, K, 1, 1, 0),
            ConvBlock(K, I, 3, 1, 'same'),
            ConvBlock(I, M, 3, 2, 'valid'),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1)


class InceptionFig17(nn.Module):
    def __init__(self, in_channels, upsample_x):
        super(InceptionFig17, self).__init__()

        self.branch1 = ConvBlock(in_channels, 192, 1, 1, 0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, 1, 0),
            ConvBlock(128, 160, (1, 7), 1, (0, 3)),
            ConvBlock(160, 192, (7, 1), 1, (3, 0)),
        )
        self.final_conv = ConvBlock(384, 1154, 1, 1, 0)
        self.upsample_x = upsample_x

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        conct = torch.cat([b1, b2], dim=1)
        final = self.final_conv(conct)
        
        return torch.relu(self.upsample_x(x) + final)
    

class ReductionFig18(nn.Module):
    def __init__(self, in_channels):
        super(ReductionFig18, self).__init__()

        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 256, 1, 1, 0),
            ConvBlock(256, 384, 3, 2, 'valid')
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, 256, 1, 1, 0),
            ConvBlock(256, 288, 3, 2, 'valid')
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 256, 1, 1, 0),
            ConvBlock(256, 288, 3, 1, 'same'),
            ConvBlock(288, 320, 3, 2, 'valid')
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)
    

class InceptionFig19(nn.Module):
    def __init__(self, in_channels, upsample_x):
        super(InceptionFig19, self).__init__()

        self.branch1 = ConvBlock(in_channels, 192, 1, 1, 0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 192, 1, 1, 0),
            ConvBlock(192, 224, (1, 3), 1, (0, 1)),
            ConvBlock(224, 256, (3, 1), 1, (1, 0)),
        )
        self.final_conv = ConvBlock(448, 2146, 1, 1, 0)
        self.upsample_x = upsample_x

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        conct = torch.cat([b1, b2], dim=1)
        final = self.final_conv(conct)
        
        return torch.relu(self.upsample_x(x) + final)
    

class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionResNetV2, self).__init__()

        self.stem = InceptionFig3()

        self.inception_a = nn.Sequential(
            InceptionFig16(in_channels=384, upsample_x=lambda x: x),
            InceptionFig16(in_channels=384, upsample_x=lambda x: x),
            InceptionFig16(in_channels=384, upsample_x=lambda x: x),
            InceptionFig16(in_channels=384, upsample_x=lambda x: x),
            InceptionFig16(in_channels=384, upsample_x=lambda x: x)
        )

        self.reduction_a = ReductionFig7(in_channels=384, K=256, I=256, M=386, N=384)

        self.inception_b = nn.Sequential(
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x),
            InceptionFig17(in_channels=1154, upsample_x=lambda x: x)
        )

        self.reduction_b = ReductionFig18(in_channels=1154)

        self.inception_c = nn.Sequential(
            InceptionFig19(in_channels=2146, upsample_x=lambda x: x),
            InceptionFig19(in_channels=2146, upsample_x=lambda x: x),
            InceptionFig19(in_channels=2146, upsample_x=lambda x: x),
            InceptionFig19(in_channels=2146, upsample_x=lambda x: x),
            InceptionFig19(in_channels=2146, upsample_x=lambda x: x)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(2146, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_inception_resnet_v2():
    model = InceptionResNetV2(num_classes=1000)
    x = torch.randn(1, 3, 299, 299)
    out = model(x)
    print(out.shape)  # Should print torch.Size([1, 1000])


test_inception_resnet_v2()
