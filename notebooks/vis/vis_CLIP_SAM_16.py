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
import traceback
from tqdm import tqdm

import sys
sys.path.append("../..")

import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip

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

def visualize(sam_generator, file_list="val_id.short.txt"):
  root = "../../datasets/people_poses/"
  prompt = "The object of "
  with open(os.path.join(root, file_list), 'r') as lf:
      data_list = [ s.strip() for s in lf.readlines() ]
  evaluate_data = data_list
  
  data_name = None
  result_table = {}
  try:
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
      sam_generator.predictor.reset_image()
      masks = sam_generator.generate(image)

      # Cut out all masks
      input_img = Image.open(img_path)
      cropped_boxes = []

      if len(masks) > 48:
        print(f'{data_name} has {len(masks)} masks, shrunk to 48')
        masks = masks[:48]
      if len(masks) == 0:
        print(f'{data_name} has no masks, skipped for now')
        continue
      for mask in masks:
        crop_box = convert_box_xywh_to_xyxy(mask['bbox'])
        if(crop_box[0]==crop_box[2] or crop_box[1]==crop_box[3]):
          continue
        cropped_boxes.append(segment_image(input_img, mask["segmentation"]).crop(crop_box))

      preprocessed_images = [preprocess(img).to(device) for img in cropped_boxes]
      stacked_images = torch.stack(preprocessed_images)
      with torch.no_grad():
        image_features = model.encode_image(stacked_images)

      # Get Mask By Label Id
      anns = gt_to_anns_of_label_mask(mask_gt)
      img_miou_sum , img_pixacc_sum, num_class = 0, 0, len(anns)
      predict_anns = []
      for ann in anns:
        scores = retriev(image_features, prompt+LABELS[ann['label']])
        ## Get Label Index with Highest Score
        predict_idx = np.argmax(scores.cpu())
        predict_idx = predict_idx.cpu()
        predict_anns.append({
            'segmentation': masks[predict_idx]["segmentation"],
            'label': ann['label'],
            'gt': ann['segmentation'],
        })
      result_table[data_name] = predict_anns
      
  except Exception as e:
    print(data_name)
    traceback.print_exc()

  return result_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
sam_checkpoint = "../../sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)

LABELS = ["Background","Hat","Hair","Glove",
        "Sunglasses","UpperClothes","Dress","Coat","Socks","Pants",
        "Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]

# 12 GB GPU might out of memory
mask_generator_default = SamAutomaticMaskGenerator(
    sam,
    points_per_side = 16,
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
result_table = visualize(mask_generator_default, 'val_id.short.txt')
np.save('vis_clip_sam_16.npy', result_table)
