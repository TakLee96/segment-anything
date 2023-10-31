import numpy as np
import torch
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# numpy version
def pixAcc(predicted, target):
    same = (predicted == target).sum()
    w, h = target.shape
    return same / (w * h)

# input: bool matrix
def IOU(predicted , target):
    intersection = np.logical_and(target, predicted).sum()
    union = np.logical_or(target, predicted).sum()
    if union == 0:
        iou_score = 0
    else :
        iou_score = intersection / union
    return iou_score
def gt_to_anns(mask_gt):
    labels = np.unique(mask_gt)
    anns = []
    for label in labels:
        # skip background
        if label == 0:
            continue
        mask = np.all(mask_gt == label, axis=-1)
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

sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)
predictor = SamPredictor(sam)
# numpy metrics
import os
from tqdm import tqdm
root = "../datasets/people_poses/"
with open(os.path.join(root, f"val_id.txt"), 'r') as lf:
    data_list = [ s.strip() for s in lf.readlines() ]


num_valid_case, sum_miou, sum_pixAcc = 0,0,0
for data_name in (pbar := tqdm(data_list)):
    image = cv2.imread(root +'val_images/' + data_name + '.jpg')
    if image is None:
        print("image is None", data_name)
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_gt = cv2.imread(root + 'val_segmentations/' + data_name + '.png')
    if mask_gt is None:
        print("mask_gt is None", data_name)
        continue
    anns = gt_to_anns(mask_gt)
    predictor.set_image(image)
    img_miou_sum , img_pixacc_sum, num_class = 0, 0, len(anns)
    for ann in anns:
        target = ann['segmentation']
        input_point = ann['random_point_1']
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        iou = IOU(masks[0],ann['segmentation'])
        pixacc = pixAcc(masks[0],ann['segmentation'])
        img_miou_sum += iou
        img_pixacc_sum += pixacc
    if num_class > 0:
        img_iou = img_miou_sum/num_class
        img_pixacc = img_pixacc_sum/num_class
        sum_miou += img_iou
        sum_pixAcc += img_pixacc
        num_valid_case += 1
        pbar.set_description("current pixAcc: {:.3f}, mIoU: {:.3f}".format(
                    img_pixacc, img_iou))
    else:
        print(data_name)
    
print("num_valid_case:", num_valid_case)
print("mIoU:", sum_miou / num_valid_case)
print("Acc:", sum_pixAcc / num_valid_case)