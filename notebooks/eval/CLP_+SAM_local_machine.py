import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.jit import Error

# numpy metrics
import os
from tqdm import tqdm

import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Convert Mask's Boundary Box from XYWH to XYXY format
def convert_box_xywh_to_xyxy(box):
  x1 = box[0]
  y1 = box[1]
  x2 = box[0] + box[2]
  y2 = box[1] + box[3]
  if(box[2]==0 or box[3]==0):
    print(box[2],box[3],[x1, y1, x2, y2])
  return [x1, y1, x2, y2]

# Show Only the Segmented Part in the Given Image
def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def gt_to_anns_of_label_mask(mask_gt):
  labels = np.unique(mask_gt)
  anns = []
  for label in labels:
    # skip background
      if label == 0:
          continue
      mask = np.all(mask_gt == label, axis=-1)
      anns.append({
        'area': np.sum(mask),
        'segmentation': mask,
        'label': label,
      })
  return anns

@torch.no_grad()
def retriev(image_features: list, search_text: str) -> int:
    # preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    # stacked_images = torch.stack(preprocessed_images)
    # image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)
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
def evaluate(sam_generator, data_cnt=100):
  root = "../../datasets/people_poses/"
  prompt = "The object of "
  with open(os.path.join(root, f"val_id.txt"), 'r') as lf:
      data_list = [ s.strip() for s in lf.readlines() ]
  if data_cnt > len(data_list):
    evaluate_data = data_list
  else:
    evaluate_data = data_list[:data_cnt]
  try:
    miou_table = []
    pix_acc_table = []
    for data_name in (pbar := tqdm(evaluate_data)):
      img_path = root +'val_images/' + data_name + '.jpg'
      seg_path = root + 'val_segmentations/' + data_name + '.png'
      # Read Image and Ground truth mask
      image = cv2.imread(img_path)
      if image is None:
          print("\nimage is None", data_name)
          continue
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask_gt = cv2.imread(seg_path)
      if mask_gt is None:
          print("\nmask_gt is None", data_name)
          continue

      # Generate masks for all object by SAM
      masks = sam_generator.generate(image)

      # Cut out all masks
      input_img = Image.open(img_path)
      cropped_boxes = []

      for mask in masks:
        crop_box = convert_box_xywh_to_xyxy(mask['bbox'])
        if(crop_box[0]==crop_box[2] or crop_box[1]==crop_box[3]):
          continue
        cropped_boxes.append(segment_image(input_img, mask["segmentation"]).crop(crop_box))

      preprocessed_images = [preprocess(img).to(device) for img in cropped_boxes]
      stacked_images = torch.stack(preprocessed_images)
      image_features = model.encode_image(stacked_images)

      # Get Mask By Label Id
      anns = gt_to_anns_of_label_mask(mask_gt)
      img_miou_sum , img_pixacc_sum, num_class = 0, 0, len(anns)
      predict_masks = {}
      for ann in anns:
        scores = retriev(image_features, prompt+LABELS[ann['label']])
        ## Get Label Index with Highest Score
        predict_idx = np.argmax(scores.cpu())
        predict_idx = predict_idx.cpu()
        predict_masks[ann['label']] = masks[predict_idx]["segmentation"]
      miou, pix_acc = compute_metric(data_name, predict_masks, mask_gt[:,:,0])
      miou_table.append(miou)
      pix_acc_table.append(pix_acc)
    return miou_table, pix_acc_table

  except Exception as e:
    print(e)
    print(miou_table)
    print(pix_acc_table)

def export_csv(miou_table, pix_acc_table, miou_csv_name="random_miou.csv", pix_acc_csv_name="random_pix_acc.csv", export = True):
  miou_table_ = pd.DataFrame(miou_table, columns=miou_table[0].keys()).set_index('name')
  pix_acc_table_ = pd.DataFrame(pix_acc_table, columns=pix_acc_table[0].keys()).set_index('name')
  if export:
    miou_table_.to_csv('clip_sam_result/'+miou_csv_name)
    pix_acc_table_.to_csv('clip_sam_result/'+pix_acc_csv_name)


  # print('miou:\n', miou_table_.mean(axis=None))
  # print('miou per class:\n', miou_table_.mean())
  print()
  # print('pix_acc:\n', pix_acc_table_.mean(axis=None))
  # print('pix_acc per class:\n', pix_acc_table_.mean())
  return miou_table_.mean(), pix_acc_table_.mean()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
sam_checkpoint = "../../sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)

# mask_generator_default = SamAutomaticMaskGenerator(sam)
LABELS = ["Background","Hat","Hair","Glove",
        "Sunglasses","UpperClothes","Dress","Coat","Socks","Pants",
        "Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]

# 12 GB GPU might out of memory
mask_generator_default = SamAutomaticMaskGenerator(
    sam,
    points_per_side = 32,
    points_per_batch = 64,
    pred_iou_thresh = 0.88,
    stability_score_thresh = 0.98, # 0.95 
    stability_score_offset = 1.0,
    box_nms_thresh = 0.7,
    crop_n_layers = 0,
    crop_nms_thresh = 0.7,
    crop_overlap_ratio = 512 / 1500,
    crop_n_points_downscale_factor = 1,
    point_grids = None,
    min_mask_region_area = 10, # 0 
    output_mode = "binary_mask"
  )
miou_table, pix_acc_table = evaluate(mask_generator_default,10000)
miou_mean_default_100, pixacc_mean_default_100 = export_csv(miou_table, pix_acc_table, miou_csv_name="random_miou_default_100.csv", pix_acc_csv_name="random_pix_acc_default_100.csv")
print('miou per class\n', miou_mean_default_100,10)