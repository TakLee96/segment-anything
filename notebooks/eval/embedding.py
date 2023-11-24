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
import random
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
    assert predicted.shape == target.shape
    assert len(predicted.shape) == 2
    return (predicted == target).mean()

def compute_IOU(predicted, target):
    assert predicted.shape == target.shape
    assert len(predicted.shape) == 2
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
            iou_metric[label_name] = np.nan
            pix_acc_metric[label_name] = np.nan
        else:
            iou_metric[label_name] = compute_IOU(mask_i, label_i)
            pix_acc_metric[label_name] = compute_pix_acc(mask_i, label_i)
    
    return iou_metric, pix_acc_metric


def set_embedding(predictor, embed, label):
    predictor.is_image_set = True
    predictor.features = embed[None,]
    # image_batch, mask_batch, height, width
    assert len(predictor.features.shape) == 4
    predictor.original_size = label.shape
    # NOTE(jiahang): resized but not cropped
    predictor.input_size = predictor.transform.get_preprocess_shape(label.shape[0], label.shape[1], 1024)

def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
    if full_prob > 0 and random.random() < full_prob:
        return obj_mask

    padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        prob_map = dt.flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        click_indx = np.random.choice(len(prob_map), p=prob_map)
        click_coords = np.unravel_index(click_indx, dt.shape)
        return np.array([click_coords])

def gt_to_anns(mask_gt):
    """ mask_gt: (H, W) """
    labels = np.unique(mask_gt)
    anns = []
    for label in labels:
        mask = (mask_gt == label)
        # center point
        center_point = get_point_candidates(mask,1 + 1e-6)
        center_point[:,[1,0]] = center_point[:,[0,1]]
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
            'center_point_1': center_point[None,0,:],
            'points':points,
        })
    return anns

def FindOracle(masks, label):
    result_iou = -1
    for i in range(len(masks)):
        iou = compute_IOU(masks[i],label)
        if result_iou < iou:
            result_iou = iou
            result_mask = masks[i]    
    return result_mask

def GetMask(predictor, input_point,oracle = False):
    input_label = np.array([1])
    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=oracle,
    )
    if oracle:
        return mask
    else:
        return mask[0]
    

    
def main():
    sam_checkpoint = "../../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    dataset = PeoplePosesDataset(mode="val")

    miou_table_map = {}
    pix_acc_table_map = {}

    eval_names = ["random", "random_oracle" ,"center", "center_oracle"]


    for e_name in eval_names:
        miou_table_map[e_name] = []
        pix_acc_table_map[e_name] = []
    

    # evaluation takes about 7 minutes on 4060 GPU
    for img_name, embed, label in tqdm(dataset):
        embed = embed.to(device)
        with torch.no_grad():
            # B, 20, 256, 256
            anns = gt_to_anns(label)
            set_embedding(predictor, embed, label)
            masks_map = {}
            for e_name in eval_names:
                masks_map[e_name] = {}
            for ann in anns:
                masks_map['random'][ann['label']] = GetMask(predictor, input_point = ann['random_point_1'])
                masks_map['random_oracle'][ann['label']] = FindOracle(GetMask(predictor, input_point = ann['random_point_1'],oracle=True),ann['segmentation'])
                masks_map['center'][ann['label']] = GetMask(predictor, input_point = ann['center_point_1'])
                masks_map['center_oracle'][ann['label']] = FindOracle(GetMask(predictor, input_point = ann['center_point_1'],oracle=True),ann['segmentation'])
    
        for e_name in eval_names:
            miou, pix_acc = compute_metric(img_name, masks_map[e_name], label)
            miou_table_map[e_name].append(miou)
            pix_acc_table_map[e_name].append(pix_acc)
    
    for e_name in eval_names:
        miou_table_map[e_name] = pd.DataFrame(miou_table_map[e_name], columns=miou_table_map[e_name][0].keys()).set_index('name')
        pix_acc_table_map[e_name] = pd.DataFrame(pix_acc_table_map[e_name], columns=pix_acc_table_map[e_name][0].keys()).set_index('name')
    
    for e_name in eval_names:
        miou_table_map[e_name].to_csv('result/' + e_name + '_miou.csv')
        pix_acc_table_map[e_name].to_csv('result/' + e_name + '_pix_acc.csv')


    for e_name in eval_names:
        print('---------------------')
        print(e_name,":")
        print()
        print('miou:', miou_table_map[e_name].mean(axis=None))
        print('miou per class:')
        print( miou_table_map[e_name].mean())
        print('--------')
        print('pix_acc:', pix_acc_table_map[e_name].mean(axis=None))
        print('pix_acc per class:')
        print(pix_acc_table_map[e_name].mean())
        print('---------------------')

    
    print(eval_names)
    print('miou:')
    print([miou_table_map[e_name].mean(axis=None) for e_name in eval_names])
    print('pix_acc:')
    print([pix_acc_table_map[e_name].mean(axis=None) for e_name in eval_names])
if __name__ == '__main__':
    main()
