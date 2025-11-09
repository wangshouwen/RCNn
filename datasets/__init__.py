from PIL import ImageDraw
import numpy as np
import random

import torchvision.transforms as transforms
from randaugment import RandAugment

from .coco import CocoDetection
from .nuswide import Nuswide
from .caption import Caption

dataset_dict = {'mscoco': CocoDetection,
                'nuswide': Nuswide,
                }

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def get_transforms(cfg):
    transform = dict()
    transform['test'] = transforms.Compose([
        transforms.Resize(cfg.INPUT.TEST_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    transform['train'] = transforms.Compose([
        transforms.Resize(cfg.INPUT.TRAIN_SIZE),
        CutoutPIL(cutout_factor=cfg.INPUT.CUTOUT_FACTOR),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform

def build_dataset(cfg, data_set):
    print(' -------------------- Building Dataset ----------------------')
    print('data_set = %s' % data_set)

    transforms = get_transforms(cfg)
    if 'train' in data_set or 'Train' in data_set:
        transform = transforms['train']
    else:
        transform = transforms['test']
        
    return dataset_dict[cfg.DATASET.NAME](cfg, cfg.DATASET.IMG_PATH, cfg.DATASET.ANNO_PATH, data_set=data_set, transform=transform)