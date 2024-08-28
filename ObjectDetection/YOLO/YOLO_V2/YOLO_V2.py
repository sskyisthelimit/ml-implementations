import torch
import torch.nn as nn

darknet19 = [
    (32, 3, 1, 1),
    'M',
    (64, 3, 1, 1),
    'M',
    (128, 3, 1, 1),
    (64, 1, 1, 1),
    (128, 3, 1, 1),
    'M',
    (256, 3, 1, 1),
    (128, 1, 1, 1),
    (256, 3, 1, 1),
    'M',
    (512, 3, 1, 1),
    (256, 1, 1, 1),
    (512, 3, 1, 1),
    (256, 1, 1, 1),
    (512, 3, 1, 1),
    'M',
    (1024, 3, 1, 1),
    (512, 1, 1, 1),
    (1024, 3, 1, 1),
    (512, 1, 1, 1),
    (1024, 3, 1, 1),
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class DarkNet19(nn.Module):
    def __init__(self):
        super(DarkNet19, self).__init__()

        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=1, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        )
        self.conv4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        )
        self.conv5 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        )
        self.conv6 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        C4 = self.conv5(x)
        C5 = self.conv6(self.pool(C4))
        return (C4, C5)


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalLayer, self).__init__()
        self.conv = ConvBlock(in_channels, 1024, kernel_size=3,
                              padding=1, stride=1)
        self.detect = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.detect(out)
        return out


class PassthroughLayer(nn.Module):
    def __init__(self, in_channels):
        super(PassthroughLayer, self).__init__()
        self.feature_dims = (64, 1024)
        # 1. reduce channels while saving spatial info
        self.conv1 = ConvBlock(in_channels[0], self.feature_dims[0],
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            ConvBlock(in_channels[1], 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, self.feature_dims[1], kernel_size=3,
                      stride=1, padding=1),
            )
        
    def forward(self, x):
        C4, C5 = x
        P4 = self.conv1(C4)
        P4 = torch.cat([P4[..., ::2, ::2], P4[..., 1::2, ::2],
                        P4[..., ::2, 1::2], P4[..., 1::2, 1::2]], dim=1)
        P5 = self.conv2(C5)
        return torch.cat((P4, P5), dim=1)


class YoloV2(nn.Module):
    def __init__(self, input_size, n_classes, anchors):
        super(YoloV2, self).__init__()
        self.stride = 32
        self.n_boxes = 5
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_attributes = n_classes + 5
        self.backbone = DarkNet19()
        self.feat_dims = (512, 1024)
        self.passthrough = PassthroughLayer(self.feat_dims)
        self.final_layer = FinalLayer(
            in_channels=self.passthrough.feature_dims[0] * 4
            + self.passthrough.feature_dims[1],
            out_channels=self.n_attributes * self.n_boxes)
        
        self.anchors = torch.tensor(anchors)
        self.set_grid_x_y()

    def forward(self, x):
        bs = x.shape[0]
        x = self.backbone(x)
        x = self.passthrough(x)
        x = self.final_layer(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2).view((
            bs, -1, self.n_boxes, self.n_attributes))
        
        preds = torch.sigmoid(x[..., [0]])
        pred_txty = torch.sigmoid(x[..., 1:3])
        pred_twth = torch.sigmoid(x[..., 3:5])
        pred_cls = torch.sigmoid(x[..., 5:])
        
        return torch.cat((preds, pred_txty, pred_twth, pred_cls), dim=-1) 

    def set_grid_x_y(self):
        self.grid_size = self.input_size // self.stride
    
        grid_x, grid_y = torch.meshgrid(
            (torch.arange(self.grid_size),
             torch.arange(self.grid_size)), indexing='ij')
        
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))


if __name__ == '__main__':
    input_size = 416
    n_classes = 20
    anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
               (9.42, 5.11), (16.62, 10.52)]

    model = YoloV2(input_size=input_size, n_classes=n_classes, anchors=anchors)

    dummy_input = torch.randn(1, 3, input_size, input_size)

    output = model(dummy_input)

    print("Output shape:", output.shape)