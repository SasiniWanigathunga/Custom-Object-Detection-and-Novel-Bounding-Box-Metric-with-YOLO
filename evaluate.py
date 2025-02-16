import os
import argparse
import yaml
from tqdm import tqdm
from copy import deepcopy
import cv2
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from models.common import *
from models.yolo import Model, Detect
from utils import non_max_suppression, scale_coords, box_iou, clip_coords, ap_per_class
from datasets import CustomDataset, get_data_path
from loss import ComputeLoss
from utils import xywh2xyxy, xyxy2xywh, xywhn2xyxy
from pathlib import Path


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # dak = [k for k, v in da.items()]
    # for i in dak:
    #     print(i)
    # dbk = [k for k, v in db.items()]
    # for i in dbk:
    #     print(i)
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def evaluate(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True):
    # Config Data
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Initialize/load model and set device, create dateloader
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        model = deepcopy(model)  # copy it
    else:
        # Device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Model
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
        model = Model(opt.cfg, ch=3).to(device)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        # state_dict = ckpt['model'].float().state_dict()  # official model, to FP32
        state_dict = ckpt.float().state_dict()  # self model
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load param
        # Val dataloader
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        _, _, val_img_path, val_label_path = get_data_path(data_dict)
        val_dataset = CustomDataset(val_img_path, val_label_path)
        nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])
        dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=nw,
                                pin_memory=True,
                                collate_fn=CustomDataset.collate_fn)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    nc = int(data_dict['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    pbar = tqdm(dataloader, total=len(dataloader))
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss_boc = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for step, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        # Disable gradients
        with torch.no_grad():
            inf_out, train_out = model(imgs)  # forward

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss_callable = ComputeLoss(model)
                loss_boc += loss_callable([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        info = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        pbar.set_description(info)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    return mp, mr, map50, map


if __name__ == '__main__':
    ''' not test in the main, just call evaluate while training '''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/test_images/dog.jpg', help='source')
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--save', action='store_true', help='save inference image')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_yolo_voc.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yolo_voc.yaml', help='hyperparameters yaml path')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        evaluate(data='data/yolo_voc.yaml', weights='result/epoch6_2025-02-16_07-03-49_model.pth')