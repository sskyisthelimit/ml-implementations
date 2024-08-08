import torch
import torch.nn as nn

architecture = [64, 64, 'POOL', 128, 128, 'POOL', 256, 256, 256,
                'POOL', 512, 512, 512, 'POOL',  512, 512, 512, 'POOL']


class VGG(nn.Module):
    def __init__(self, in_channels, n_classes, architecture):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv = self.architecture_setup(architecture)

        self.fully = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        fully_res = self.fully(x)
        return nn.functional.softmax(fully_res, dim=1)

    def architecture_setup(self, architecture):
        cur_in_channels = self.in_channels
        conv = []

        for x in architecture:
            if x == 'POOL':
                conv.append(nn.MaxPool2d(2, 2))
            else:
                conv.append(nn.Conv2d(cur_in_channels, x, 3, 1, 1))
                conv.append(nn.BatchNorm2d(x))
                conv.append(nn.ReLU(inplace=True))
            
                cur_in_channels = x

        return nn.Sequential(*conv)


model = VGG(3, 1000, architecture)
print(model(torch.randn((1, 3, 224, 224))).shape)
