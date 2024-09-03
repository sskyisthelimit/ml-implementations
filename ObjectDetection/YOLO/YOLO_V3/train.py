import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    check_class_accuracy,
    get_loaders,  
)

from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")


def train(loader, model, loss_fn, optim, scaled_anchors):
    losses = []

    loop = tqdm(loader)

    for batch_idx, (x, y) in enumerate(loop):
        y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE),
                      y[2].to(config.DEVICE))
        x = x.to(config.DEVICE)

        pred = model(x)
        loss = (loss_fn(pred[0], y0, scaled_anchors[0]) +
                loss_fn(pred[1], y1, scaled_anchors[1]) +
                loss_fn(pred[2], y2, scaled_anchors[2]))
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

        mean_loss = sum(losses) / len(losses)
        tqdm.set_postfix(loss=mean_loss)


def main():
    n_classes = config.NUM_CLASSES
    model = YOLOv3(n_classes=n_classes)
    loss_fn = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    train_loader, test_loader, tr_eval_loader = get_loaders(
        config.DATASET + 'train.csv', config.DATASET + 'test.csv')

    scaled_anchors = (
        torch.tensor(config.ANCHORS) *
        torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    
    train(train_loader, model, loss_fn, optimizer, scaled_anchors)
    if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename="checkpoint.pth.tar")

    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")


if __name__ == '__main__':
    main()

        
