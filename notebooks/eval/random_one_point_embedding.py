import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("../..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import pandas as pd
from tqdm import tqdm

# CONSTANTS
device = 'cuda:0'
LABELS = """Background
Hat
Hair
Glove
Sunglasses
UpperClothes
Dress
Coat
Socks
Pants
Jumpsuits
Scarf
Skirt
Face
Left-arm
Right-arm
Left-leg
Right-leg
Left-shoe
Right-shoe""".split('\n')


class PeoplePosesDataset(Dataset):
    def __init__(self, mode="train", img_size=1024):
        assert mode in ("train", "val")
        self.mode = mode
        self.root = "../../datasets/people_poses"
        self.image_dir = os.path.join(self.root, f"{self.mode}_images")
        self.mask_dir = os.path.join(self.root, f"{self.mode}_segmentations")
        self.embed_dir = os.path.join(self.root, f"{self.mode}_embeds")
        with open(os.path.join(self.root, f"{self.mode}_id.txt"), 'r') as lf:
            self.data_list = [ s.strip() for s in lf.readlines() ]
        self.img_size = img_size

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """ embed: (256, 64, 64)
            label: (H, W)
        """
        data = np.load(os.path.join(self.embed_dir, self.data_list[index] + ".npz"))
        embed = data['embed']
        embed = torch.as_tensor(embed)
        
        label = cv2.imread(os.path.join(self.mask_dir, self.data_list[index] + ".png"))

        return self.data_list[index], embed, label[:,:,0]


def compute_pix_acc(predicted, target):
    return (predicted == target).mean()

def compute_IOU(predicted, target):
    intersection = np.logical_and(target, predicted).sum()
    union = np.logical_or(target, predicted).sum()
    assert union > 0
    return intersection / union

def compute_metric(name, masks, label):
    """ name: data_id
        mask: { label_id: numpy.ndarray(shape=(H, W)) }
        label: np.ndarray(shape=(H, W)) --> numbers from 0 to 19
    """
    pix_acc_metric = { "name": name }
    iou_metric = { "name": name }
    empty = np.zeros_like(label)
    for i, label_name in enumerate(LABELS):
        mask_i = masks.get(i, empty)
        label_i = (label == i)
        if label_i.sum() == 0:
            # pandas dataframe automatically skips nan
            # when computing .count() and .mean()
            pix_acc_metric[label_name] = np.nan
            iou_metric[label_name] = np.nan
        else:
            pix_acc_metric[label_name] = compute_pix_acc(mask_i, label_i)
            iou_metric[label_name] = compute_IOU(mask_i, label_i)
    
    return pix_acc_metric, iou_metric


def set_embedding(predictor, embed, label):
    predictor.is_image_set = True
    predictor.features = embed
    predictor.original_size = label.shape
    # TODO(jiahang): fix magic numbers
    predictor.input_size = (1024, 1024)


def gt_to_anns(mask_gt):
    """ mask_gt: (H, W) """
    labels = np.unique(mask_gt)
    anns = []
    for label in labels:
        mask = (mask_gt == label)
        # 1 ramdon point from mask
        num_point = 1
        indices = np.argwhere(mask)
        # swap x y
        indices[:,[1,0]] = indices[:,[0,1]]
        # sample on random point
        point = np.asarray(indices[np.random.randint(indices.shape[0], size=num_point)])
        # all point for test
        points = np.asarray(indices)
        anns.append({
            'area': np.sum(mask),
            'segmentation': mask,
            'label': label,
            'random_point_1':point,
            'points':points,
        })
    return anns

def main():
    sam_checkpoint = "../../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    dataset = PeoplePosesDataset(mode="val")

    miou_table = []
    pix_acc_table = []
    # evaluation takes about 7 minutes on 4060 GPU
    for name, embed, label in tqdm(dataset):
        embed = embed.to(device)
        with torch.no_grad():
            # B, 20, 256, 256
            anns = gt_to_anns(label)
            set_embedding(predictor, embed, label)
            masks = {}
            for ann in anns:
                input_point = ann['random_point_1']
                input_label = np.array([1])
                masks[ann['label']], _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
        miou, pix_acc = compute_metric(name, masks, label)
        miou_table.append(miou)
        pix_acc_table.append(pix_acc)
    
    miou_table = pd.DataFrame(miou_table, columns=miou_table[0].keys()).set_index('name')
    pix_acc_table = pd.DataFrame(pix_acc_table, columns=pix_acc_table[0].keys()).set_index('name')

    miou_table.to_csv('random_miou.csv')
    pix_acc_table.to_csv('random_pix_acc.csv')

    print('miou:', miou_table.mean(axis=None))
    print('miou per class:\n', miou_table.mean())
    print()
    print('pix_acc:', pix_acc_table.mean(axis=None))
    print('pix_acc per class:\n', pix_acc_table.mean())
    



if __name__ == '__main__':
    main()
