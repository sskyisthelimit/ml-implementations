import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        # MSE for bbox coords
        self.mse = nn.MSELoss()
        # bce
        self.bce = nn.BCEWithLogitsLoss()
        # cross entropy for clsf. loss
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, preds, target, anchors):
        # Determine where object exists and where it does not
        I_obj = target[..., 0] == 1
        I_noobj = target[..., 0] == 0

        # Reshape anchors to fit the prediction tensor dimensions
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)

        # COORDS LOSS
        # Apply sigmoid to x and y predictions
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        # Apply log to the width and height of target bounding boxes
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        coords_loss = self.mse(preds[..., 1:5][I_obj], target[..., 1:5][I_obj])

        # ================== #
        #  NO OBJECT LOSS     #
        # ================== #

        no_object_loss = self.bce(
            preds[..., 0:1][I_noobj], target[..., 0:1][I_noobj]
        )

        # OBJECT LOSS   
        # Calculate bounding boxes using anchors
        bboxes = torch.cat(
            [self.sigmoid(preds[..., 1:3]),
             torch.exp(preds[..., 3:5]) * anchors], dim=-1
        )
        # Calculate IoU for predicted and target boxes
        ious = intersection_over_union(bboxes[I_obj],
                                       target[..., 1:5][I_obj]).detach()
        object_loss = self.bce(preds[..., 0:1][I_obj],
                               ious * target[..., 0:1][I_obj])

        # CLASS LOSS
        class_loss = self.entropy(
            preds[..., 5:][I_obj], target[..., 5][I_obj].long()
        )

        # Combine all losses with their respective lambda values
        return (
            self.lambda_box * coords_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
    

def test():
    # Define the dimensions
    batch_size = 2
    grid_size = 13  
    num_anchors = 3
    num_classes = 20  
    bbox_attrs = 5 + num_classes  

    preds = torch.randn((batch_size, num_anchors, grid_size,
                         grid_size, bbox_attrs))

    target = torch.zeros((batch_size, num_anchors, grid_size,
                          grid_size, bbox_attrs))
    
    for i in range(batch_size):
        for j in range(num_anchors):
            x_idx = torch.randint(0, grid_size, (1,)).item()
            y_idx = torch.randint(0, grid_size, (1,)).item()
            target[i, j, x_idx, y_idx, 0] = 1  
            target[i, j, x_idx, y_idx, 1:5] = torch.tensor(
                [0.5, 0.5, 1.0, 1.0])
            target[i, j, x_idx, y_idx, 5 + torch.randint(
                0, num_classes, (1,)).item()] = 1  

    # Define random anchors
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

    # Initialize the loss function
    loss_fn = YoloLoss()

    # Calculate the loss
    loss = loss_fn(preds, target, anchors)
    print("Loss:", loss.item())


if __name__ == '__main__':
    test()