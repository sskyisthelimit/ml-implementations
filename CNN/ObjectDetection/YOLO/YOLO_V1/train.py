import torch
from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader


from ObjectDetection.YOLO.YOLO_V1.YOLO_V1 import Yolov1, YoloLoss
from ObjectDetection.YOLO.YOLO_V1.utils import (get_bboxes,
                                                mean_average_precision)


def get_syntetic_dls(sz, bs, C, shuffle=True):

    images = torch.randn(32, 3, sz, sz)
    targets = torch.randn(32, 7, 7, C + 5)

    val_images = torch.randn(8, 3, sz, sz)
    val_targets = torch.randn(8, 7, 7, C + 5)

    train_ds = TensorDataset(images, targets)
    train_dl = DataLoader(train_ds, bs, shuffle=shuffle)

    val_ds = TensorDataset(val_images, val_targets)
    val_dl = DataLoader(val_ds, bs, shuffle=False)

    return train_dl, val_dl


def train():
    S, B, C = 7, 2, 20
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    opt = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    dls = get_syntetic_dls(448, 2, C)

    tr_loop = tqdm(enumerate(dls[0]), total=len(dls[0])) 
    for bc_idx, (data, targets) in tr_loop:
        opt.zero_grad()
        predictions = model(data)
        loss = YoloLoss(S, B, C)(predictions, targets)
        loss.backward()
        opt.step()
        tr_loop.set_postfix(loss=loss.item())
    model.eval()

    with torch.no_grad():
        pred_boxes, target_boxes = get_bboxes(
            dls[1], model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Val mAP: {mean_avg_prec}")


if __name__ == '__main__':
    train()