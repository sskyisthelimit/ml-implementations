import torch
import torch.nn as nn

# Define architecture
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# Define YOLOv1 class
class YOLOV1(nn.Module):
    def __init__(self, in_channels, architecture, S=7, B=2, C=20):
        super(YOLOV1, self).__init__()

        self.prms_and_clss = C + 5 * B
        self.S = S
        
        self.conv = self._create_conv_layers(in_channels, architecture)
        self.fc = self._create_fc_layers(S)

    def _create_conv_layers(self, in_channels, architecture):
        layers = []
        
        for l in architecture:
            if isinstance(l, tuple):
                layers.append(ConvBlock(in_channels, l[1], kernel_size=l[0],
                                        stride=l[2], padding=l[3]))
                in_channels = l[1]

            elif isinstance(l, list):
                for idx in range(l[-1]):
                    layers.append(nn.Sequential(
                        ConvBlock(in_channels, l[0][1], kernel_size=l[0][0],
                                  stride=l[0][2], padding=l[0][3])))
                    in_channels = l[0][1]
                    layers.append((
                        ConvBlock(in_channels, l[1][1], kernel_size=l[1][0],
                                  stride=l[1][2], padding=l[1][3])
                    ))
                    in_channels = l[1][1]
            elif l == 'M':
                layers.append(nn.MaxPool2d(2, 2))

        return nn.Sequential(*layers)
    
    def _create_fc_layers(self, S):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096, True),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.prms_and_clss * S * S)
        )
    
    def forward(self, x):
        x = self.fc(self.conv(x))
        return x


# Define ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5  
        self.lambda_noobj = 0.5  

    def forward(self, predictions, target):
        predictions = predictions.view(-1, self.S, self.S, self.C + self.B * 5)
        
        pred_boxes = predictions[..., self.C:self.C + 5*self.B].view(
            -1, self.S, self.S, self.B, 5)
        
        pred_classes = predictions[..., :self.C]
        
        target_boxes = target[..., self.C:self.C + 5*self.B].view(
            -1, self.S, self.S, self.B, 5)
        
        target_classes = target[..., :self.C]
        
        pred_box_coords = pred_boxes[..., :4]
        pred_confidences = pred_boxes[..., 4]

        target_box_coords = target_boxes[..., :4]
        target_confidences = target_boxes[..., 4]
        
        box_loss = torch.sum(
            target_confidences *
            nn.functional.mse_loss(pred_box_coords,
                                   target_box_coords,
                                   reduction='none'),
        )
        
        conf_loss_obj = torch.sum(
            target_confidences *
            nn.functional.mse_loss(pred_confidences,
                                   target_confidences,
                                   reduction='none')
        )
        
        conf_loss_noobj = torch.sum(
            (1 - target_confidences) *
            nn.functional.mse_loss(pred_confidences,
                                   target_confidences,
                                   reduction='none')
        )
        
        class_loss = nn.functional.mse_loss(pred_classes,
                                            target_classes, reduction='sum')
        
        loss = (
            self.lambda_coord * box_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            class_loss
        )
        
        return loss


model = YOLOV1(3, architecture_config)
x = torch.randn(2, 3, 448, 448)
test_output = model(x)

# Check output shape
print("Output shape:", test_output.shape)