import torch.nn as nn
import torch 


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        distances = torch.norm(x1 - x2, p=2, dim=1) 

        loss_positive = ((1 - label) / 2) * torch.pow(distances, 2) 
        loss_negative = (label / 2) * torch.pow(
            torch.clamp(self.margin - distances, min=0),
            2)  # For dissimilar pairs

        loss = torch.mean(loss_positive + loss_negative) 
        return loss
    

class SiameseNetwork(nn.Module):
    def __init__(self, ):
        super(SiameseNetwork, self).__init__()
        self.loss = ContrastiveLoss()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


