import torch
import torch.nn as nn


class ResNetBelow50Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand_factor=1,
                 identity_downsample=None, stride=1):
        super(ResNetBelow50Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * expand_factor,
                      1, 1, bias=False),
            nn.BatchNorm2d(out_channels * expand_factor),
        )
        self.relu = nn.ReLU()
    
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        
        return self.relu(x)


class ResNetOver50Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand_factor,
                 identity_downsample=None, stride=1):
        super(ResNetOver50Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * expand_factor,
                      1, 1, bias=False),
            nn.BatchNorm2d(out_channels * expand_factor),
        )
        self.relu = nn.ReLU()
    
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, n_classes, expand_factor=4):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.expand_factor = expand_factor
        self.initial_block = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)            
        )

        self.layer1 = self._create_layer(block, layers[0], 64, 1)
        self.layer2 = self._create_layer(block, layers[1], 128, 2)
        self.layer3 = self._create_layer(block, layers[2], 256, 2)
        self.layer4 = self._create_layer(block, layers[3], 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expand_factor, n_classes)
    
    def forward(self, x):
        x = self.initial_block(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def _create_layer(self, block, n_res_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if (stride != 1 or
            self.in_channels != out_channels * self.expand_factor):
            
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * self.expand_factor, 1,
                          stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expand_factor))

        layers.append(
            block(self.in_channels, out_channels,
                  expand_factor=self.expand_factor,
                  identity_downsample=identity_downsample,
                  stride=stride))
        
        self.in_channels = out_channels * self.expand_factor

        for i in range(n_res_blocks - 1):
            layers.append(
                block(self.in_channels, out_channels,
                      expand_factor=self.expand_factor))
            
        return nn.Sequential(*layers)
    

model50 = ResNet(ResNetOver50Block, [3, 4, 6, 3], 3, 10)  
model34 = ResNet(ResNetBelow50Block, [3, 4, 6, 3], 3, 10, 1)  
output1 = model50(torch.randn(2, 3, 256, 256))
output2 = model34(torch.randn(2, 3, 256, 256))
print(output1.shape, output2.shape)
