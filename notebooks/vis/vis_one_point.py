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
    def __init__(self, img_size=1024):
        self.mode = "val"
        self.root = "../../datasets/people_poses"
        self.image_dir = os.path.join(self.root, f"{self.mode}_images")
        self.mask_dir = os.path.join(self.root, f"{self.mode}_segmentations")
        self.embed_dir = os.path.join(self.root, f"{self.mode}_embeds")
        with open(os.path.join(self.root, f"val_id.short.txt"), 'r') as lf:
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


def convert_to_anns(name, mask, label):
    """ recover mask at original resolution and compute mIoU
        name: data id
        mask: { label_id: numpy.ndarray(shape=(H, W)) }
        label: np.ndarray(shape=(H, W)) --> numbers from 0 to 19
    """
    anns = []
    for i, label_name in enumerate(LABELS):
        if i not in mask:
            continue
        mask_i  = mask[i]
        label_i = (label == i)
        if label_i.sum() == 0:
            continue
        
        anns.append({
            "segmentation": mask_i,
            "label": i,
            "gt": label_i,
        })

    return anns


def set_embedding(predictor, embed, label):
    predictor.is_image_set = True
    predictor.features = embed[None,]
    # image_batch, mask_batch, height, width
    assert len(predictor.features.shape) == 4
    predictor.original_size = label.shape
    # NOTE(jiahang): resized but not cropped
    predictor.input_size = predictor.transform.get_preprocess_shape(label.shape[0], label.shape[1], 1024)


def get_point_candidates(obj_mask, k=1.7):
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
        # 1 ramdon point from mask
        num_point = 1
        indices = np.argwhere(mask)
        center_point = get_point_candidates(mask, 1 + 1e-6)
        # swap x y
        indices[:,[1,0]] = indices[:,[0,1]]
        center_point[:,[1,0]] = center_point[:,[0,1]]
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

def main():
    sam_checkpoint = "../../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    dataset = PeoplePosesDataset()
    result_table = {}

    for name, embed, label in tqdm(dataset):
        embed = embed.to(device)
        with torch.no_grad():
            # B, 20, 256, 256
            anns = gt_to_anns(label)
            set_embedding(predictor, embed, label)
            masks = {}
            for ann in anns:
                input_point = ann['center_point_1']
                input_label = np.array([1])
                mask, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                masks[ann['label']] = mask[0]
        anns = convert_to_anns(name, masks, label)
        result_table[name] = anns
    
    np.save('vis_one_point.npy', result_table)


if __name__ == '__main__':
    main()
