import os
import pickle
from pycocotools.coco import COCO
from PIL import Image

import torch
from torchvision import datasets as datasets


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, cfg, img_path, anno_path, data_set='train', transform=None):
        self.classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                           "orange",
                           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        self.cfg = cfg
        self.img_path = img_path + '/' + '{}2014'.format(data_set)
        self.transform = transform
        cls_id = pickle.load(open(os.path.join(anno_path, "cls_ids.pickle"), "rb"))
        self.seen_cls_idx = list(cls_id['train'])
        self.seen_cls_idx.sort()
        self.seen_names = [self.classnames[x] for x in self.seen_cls_idx]

        if data_set == 'test':
            self.img_path = self.img_path.replace('test', 'val')
            self.unseen_cls_idx = list(cls_id['test'])
            self.unseen_cls_idx.sort()
            self.unseen_names = [self.classnames[x] for x in self.unseen_cls_idx]
            annFile = os.path.join(anno_path, 'instances_val2014_gzsi_48_17.json')
            cls_id = self.seen_cls_idx + self.unseen_cls_idx # 65
            cls_id.sort()
        else:
            annFile = os.path.join(anno_path, 'instances_train2014_seen_48_17.json')
            cls_id = self.seen_cls_idx # 48

        self.coco = COCO(annFile)
        self.data_set = data_set
        self.ids = list(self.coco.imgToAnns.keys()) # img_list

        self.cat2cat = dict()
        cats_keys = [*self.coco.cats.keys()]
        cats_keys.sort()
        for cat, cat2 in zip(cats_keys, cls_id):
            self.cat2cat[cat] = cat2

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros(len(self.classnames), dtype=torch.long)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        if self.transform is not None:
            img_ = self.transform(img)
        
        seen_label = 2 * target[self.seen_cls_idx] - 1
        if self.data_set == 'test':
            unseen_label = 2 * target[self.unseen_cls_idx] - 1
            if self.cfg.cutimage:
                img_list = CutImageplus(img, count_list=self.cfg.count_list)
                img_list = [self.transform(image) for image in img_list]
                return img_, seen_label, unseen_label, img_list
            else:
                return img_, seen_label, unseen_label

        return img_, seen_label
    

def CutImageplus(img, img_size=224, count_list=[2,3]):
    img = img.resize((img_size, img_size))
    box_list = []
    for count in count_list:
        temp_width = int(img_size / count)
        for i in range(0, count):
            for j in range(0, count):
                box = (j*temp_width, i*temp_width, (j+1)*temp_width, (i+1)*temp_width)
                box_list.append(box)
    image_list = [img.crop(box) for box in box_list]
    return image_list
