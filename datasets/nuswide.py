import os
import pickle
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def save_pickle(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def load_labels_81(filename, tag81):
    label_tags = []
    for tag in tag81:
        with open(filename + 'Labels_{}.txt'.format(tag),"r") as file: 
            label_tag = np.array(file.read().splitlines(), dtype=np.float32)[:, np.newaxis]
            label_tags.append(label_tag)
    label_tags = np.concatenate(label_tags, axis=1)
    return label_tags

def get_seen_unseen_classes(file_tag1k, file_tag81):
    with open(file_tag1k, "r") as file: 
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81,"r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx, unseen_cls_idx, tag1k, tag81

def load_id_label_imgs(id_filename, data_partition, label1k_filename, label81_human_filename, tag81):
    with open(id_filename, "r") as file:
        id_imgs = file.readlines()
        id_imgs = [id_img.rstrip().replace('\\', '/') for id_img in id_imgs]
        
    with open(data_partition, "r") as file:
        idxs_partition = file.readlines()
        idxs_partition = [idx.rstrip().replace('\\', '/') for idx in idxs_partition]
        
    with open(label1k_filename, "r") as file:
        label1k_imgs = file.readlines()
    
    dict_img_id = {}
    for idx, id_img in enumerate(id_imgs):
        key = id_img.split('/')[-1]
        dict_img_id[key] = idx
    
    label1k_imgs = [np.array(label_img[:-2].split('\t'), dtype = np.float32) for label_img in label1k_imgs]
    label81_imgs = load_labels_81(label81_human_filename, tag81)
    return dict_img_id, idxs_partition, label1k_imgs, label81_imgs

def get_labels(anno_path, img_path, data_set):
    file_tag1k = os.path.join(anno_path, 'TagList1k.txt')
    file_tag81 = os.path.join(anno_path, 'Concepts81.txt')
    seen_cls_idx, unseen_cls_idx, tag1k, tag81 = get_seen_unseen_classes(file_tag1k, file_tag81)

    id_filename = os.path.join(anno_path, 'Imagelist.txt')
    label1k_filename = os.path.join(anno_path, 'AllTags1k.txt')
    label81_human_filename = os.path.join(anno_path, 'AllLabels/')
    data_partition = os.path.join(anno_path, '{}Imagelist.txt'.format(data_set.title()))
    dict_img_id, idxs_partition, label1k_imgs, label81_imgs = load_id_label_imgs(id_filename, data_partition, label1k_filename,
                                                                                 label81_human_filename, tag81)
    
    seen_dict = {}
    if data_set == 'test':
        unseen_dict = {}
        for img_id in idxs_partition:
            img_path_ = os.path.join(img_path, img_id)
            if not os.path.exists(img_path_):
                continue
            img_id_ = img_id.split('/')[-1]
            idx_dict = dict_img_id[img_id_]
            label81 = 2*label81_imgs[idx_dict]-1
            label1k = 2*label1k_imgs[idx_dict]-1
            label1k[unseen_cls_idx] = 0

            seen_dict[img_id] = label1k
            unseen_dict[img_id] = label81
        save_pickle(seen_dict, os.path.join(anno_path, '{}_seen_labels.pkl'.format(data_set)))
        save_pickle(unseen_dict, os.path.join(anno_path, '{}_unseen_labels.pkl'.format(data_set)))
        print('Seen and unseen labels have been saved!')
    
    else:
        for img_id in idxs_partition:
            img_path_ = os.path.join(img_path, img_id)
            if not os.path.exists(img_path_):
                continue
            img_id_ = img_id.split('/')[-1]
            idx_dict = dict_img_id[img_id_]
            label1k = 2*label1k_imgs[idx_dict]-1

            seen_dict[img_id] = label1k
        save_pickle(seen_dict, os.path.join(anno_path, '{}_seen_labels.pkl'.format(data_set)))
        print('Seen labels have been saved!')


class Nuswide(Dataset):
    def __init__(self, cfg, img_path, anno_path, data_set='train', transform=None):
        super(Nuswide, self).__init__()
        file_tag1k = os.path.join(anno_path, 'TagList1k.txt')
        file_tag81 = os.path.join(anno_path, 'Concepts81.txt')
        seen_label_path = os.path.join(anno_path, '{}_seen_labels.pkl'.format(data_set))
        seen_cls_idx, unseen_cls_idx, tag1k, tag81 = get_seen_unseen_classes(file_tag1k, file_tag81)

        self.cfg = cfg
        self.img_path = img_path
        self.data_set = data_set
        self.transform = transform
        self.seen_cls_idx = seen_cls_idx
        self.seen_names = tag1k[seen_cls_idx] #925

        if not os.path.exists(seen_label_path):
            get_labels(anno_path, img_path, data_set)
        self.seen_label_dict = load_pickle(seen_label_path)

        if data_set == 'test':
            unseen_label_path = os.path.join(anno_path, '{}_unseen_labels.pkl'.format(data_set))
            self.unseen_label_dict = load_pickle(unseen_label_path)
            self.unseen_names = tag81
        
        self.img_names = list(self.seen_label_dict.keys())

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img_ = self.transform(img)

        seen_label = np.int32(self.seen_label_dict[img_name])
        seen_label = torch.from_numpy(seen_label[self.seen_cls_idx]) # 925/1000
        if self.data_set == 'test':
            unseen_label = np.int32(self.unseen_label_dict[img_name]) # 81
            unseen_label = torch.from_numpy(unseen_label)
            if self.cfg.cutimage:
                img_list = CutImageplus(img, count_list=self.cfg.count_list)
                img_list = [self.transform(image) for image in img_list]
                return img_, seen_label, unseen_label, img_list
            else:
                return img_, seen_label, unseen_label 
        return img_, seen_label
    
    def __len__(self):
        return len(self.img_names)


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