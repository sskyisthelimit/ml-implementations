import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5, label_smoothing=0.0):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(reduction='sum') 

        self.entropy = nn.CrossEntropyLoss(reduction='sum',
                                           label_smoothing=label_smoothing)
        self.sigmoid = nn.Sigmoid()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, target, anchors):
        I_obj = target[..., 0] == 1
        I_noobj = target[..., 0] == 0

        # anchor box dimensions to match prediction shape
        anchors = anchors.reshape(1, 5, 1, 1, 2)

        # COORDS LOSS
        # Sigmoid the predictions for tx and ty and apply
        #  exponentiation for tw and th
        pred_boxes = torch.cat([
            self.sigmoid(preds[..., 1:3]),  # tx, ty
            torch.exp(preds[..., 3:5]) * anchors  # tw, th
        ], dim=-1)

        target_boxes = target[..., 1:5]

        # COORDS LOSS
        coords_loss = self.lambda_coord * (
            self.mse(pred_boxes[..., :2][I_obj],
                     target_boxes[..., :2][I_obj]) +
            self.mse(pred_boxes[..., 2:][I_obj],
                     target_boxes[..., 2:][I_obj])
        )

        # OBJECTNESS LOSS
        iou_scores = intersection_over_union(pred_boxes[I_obj],
                                             target_boxes[I_obj]).detach()
        obj_loss = self.mse(self.sigmoid(preds[..., 0][I_obj]), iou_scores)
        noobj_loss = self.mse(self.sigmoid(preds[..., 0][I_noobj]),
                              target[..., 0][I_noobj])

        objectness_loss = obj_loss + self.lambda_noobj * noobj_loss

        # CLASS LOSS
        class_loss = self.entropy(preds[..., 5:][I_obj],
                                  target[..., 5:][I_obj].argmax(dim=-1))

        # Sum up all the losses
        total_loss = coords_loss + objectness_loss + class_loss

        return total_loss
    
    
def test():
    # Define the dimensions
    batch_size = 2
    grid_size = 13  
    num_anchors = 5
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

    # random anchors
    anchors = torch.tensor([[10, 13], [16, 30],
                            [33, 23], [11, 26], [16, 19]], dtype=torch.float32)

    loss_fn = YoloLoss()

    loss = loss_fn(preds, target, anchors)
    print("Loss:", loss.item())


if __name__ == '__main__':
    test()