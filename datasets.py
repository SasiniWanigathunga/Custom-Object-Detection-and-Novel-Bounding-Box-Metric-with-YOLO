import os
import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, '..')
from utils import xyxy2xywh, xywhn2xyxy
# from general import xyxy2xywh, xywhn2xyxy


class CustomDataset(Dataset):
    def __init__(self, img_path: list, label_path: list, img_size=640):
        '''
        :param img_path: all image path in a list
        :param label_path: all image label in a list
        :param img_size: net input size
        :param augment: image augment
        :param hyp: hyp params for image augment
        :param radam_perspect: use random_perspective func or not
        '''
        self.img_path = img_path
        self.label_path = label_path
        assert len(self.img_path) == len(self.label_path) or len(self.img_path) > 0, \
            "image count:{} != label cnt:{}".format(len(self.img_path), len(self.label_path))
        self.len = len(self.img_path)
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff']  # acceptable image suffixes
        for img in self.img_path:
            img_fmt = img.split('.')[-1]
            assert img_fmt in self.img_formats, "{} img format is not acceptable".format(img)  # check image format
        self.img_size = img_size


    def load_labels(self, index):
        # loads labels of 1 image from dataset, returns labels(np) [[] []...]
        path = self.label_path[index]

        labels = []
        tree = ET.parse(path)  # Replace with the correct path to your XML file
        root = tree.getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            if obj_name == 'dog':
                cls = 0
            elif obj_name == 'cat':
                cls = 1
            
            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            labels.append([cls, float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)])

        # with open(path) as f:
        #     labels_str = f.readlines()
        # for lstr in labels_str:
        #     l = lstr.strip().split(" ")
        #     labels.append([float(i) for i in l])
        return np.float32(labels)

    def load_image(self, index):
        # loads 1 image from dataset, returns img(np), original hw, resized hw
        path = self.img_path[index]
        # print("path: ", path)

        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def __getitem__(self, index):
        # return No.index data(tensor)
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        # Letterbox
        shape = (self.img_size, self.img_size)  # final letterboxed shape, use net input img shape
        img, ratio, pad = letterbox(img, shape, auto=False)  # img, ratio, (dw, dh)

        # Load labels
        labels = self.load_labels(index).copy()
        # if labels.shape:  # normalized xywh to pixel xyxy format
        #     # print("labels.shape: ", labels.shape)
        #     labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert pixel xyxy to pixel xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1


        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR to RGB, to 3x416x416(3x640x640), norm
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img).float(), labels_out

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
              stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def get_data_path(data: dict):
    # train
    train_img_path = data['train']
    train_label_path = train_img_path.replace('images', 'labels')
    train_img_path = [os.path.join(train_img_path, f) for f in os.listdir(train_img_path)]
    train_img_path.sort()
    train_label_path = [os.path.join(train_label_path, f) for f in os.listdir(train_label_path)]
    train_label_path.sort()
    # val
    val_img_path = data['val']
    val_label_path = val_img_path.replace('images', 'labels')
    val_img_path = [os.path.join(val_img_path, f) for f in os.listdir(val_img_path)]
    val_img_path.sort()
    val_label_path = [os.path.join(val_label_path, f) for f in os.listdir(val_label_path)]
    val_label_path.sort()
    return train_img_path, train_label_path, val_img_path, val_label_path


if __name__ == '__main__':
    # img
    train_sample_dataset_path = "./data/train_sample_dataset"
    train_img_path = os.path.join(train_sample_dataset_path, "images", "train")
    train_img_path = [os.path.join(train_img_path, f) for f in os.listdir(train_img_path)]
    train_img_path.sort()
    # label
    train_label_path = os.path.join(train_sample_dataset_path, "labels", "train")
    train_label_path = [os.path.join(train_label_path, f) for f in os.listdir(train_label_path)]
    train_label_path.sort()
    # Hyperparameters

    # CustomDataset
    dataset = CustomDataset(train_img_path, train_label_path)
    for i in range(len(dataset)):
        img, labels = dataset.__getitem__(i)
        print("img shape:{}, labels shape:{}".format(img.shape, labels))