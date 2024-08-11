import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, one_by_one, reduce_3, three_by_three,
                 reduce_5, five_by_five, one_by_one_pool):
        super(InceptionBlock, self).__init__()  

        self.branch_1 = ConvBlock(in_channels, one_by_one, kernel_size=1,
                                  stride=1, padding=0)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, reduce_3, kernel_size=1,
                      stride=1, padding=0),
            ConvBlock(reduce_3, three_by_three, kernel_size=3,
                      stride=1, padding=1)
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, reduce_5, kernel_size=1,
                      stride=1, padding=0),
            ConvBlock(reduce_5, five_by_five, kernel_size=5,
                      stride=1, padding=2)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, one_by_one_pool, kernel_size=1,
                      stride=1, padding=0)
        )

    def forward(self, x):
        res_1 = self.branch_1(x)
        res_2 = self.branch_2(x)
        res_3 = self.branch_3(x)
        res_4 = self.branch_4(x)

        return torch.cat([res_1, res_2, res_3, res_4], dim=1)
    

class InceptionNetwork(nn.Module):
    def __init__(self, in_channels=3, n_classes=1000):
        super(InceptionNetwork, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = ConvBlock(64, 192, 3, 1, 1)
        self.max_pool2 = nn.MaxPool2d(3, 2, 1)

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(3, 2, 1)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(3, 2, 1)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(7, 1)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool3(x)
        
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        return self.linear(x)
        
        
model = InceptionNetwork()
x = torch.randn(3, 3, 224, 224)
print(model(x).shape)