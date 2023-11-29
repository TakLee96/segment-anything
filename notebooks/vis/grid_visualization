from collections import defaultdict
from mpl_toolkits.axes_grid1 import ImageGrid
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
from PIL import Image, ImageDraw
import clip
import sys

mask_root = "/content/drive/MyDrive/CSCI567/segment-anything/notebooks/Cindy_data/" # Your path to npy and image savings
img_root = "/content/drive/MyDrive/CSCI567/segment-anything/datasets/people_poses/val_images/" # Your path to validation images
files = os.listdir(mask_root)
results = {}

def visualize(img_lst, result, exp):
  fig = plt.figure(figsize=(12., 12.))
  grid1 = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )
  grid2 = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )
  originals = []
  layers = []
  max_h, max_w = -1, -1
  for img_name in img_lst:
    image = cv2.imread(img_root+img_name+".jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    layer = np.ones((image.shape[0], image.shape[1], 4))
    max_h = max(max_h, image.shape[0])
    max_w = max(max_w, image.shape[1])
    layer[:,:,3] = 0

    for item in result[img_name]:
      m = item['segmentation']
      layer[m] = colors[item['label']]

    originals.append(image)
    layers.append(layer)
  
  for ax, im in zip(grid1, originals):
    ax.axis('off')
    ax.imshow(im)
  cnt = 0
  for ax, im in zip(grid2, layers):
    ax.set_title(f'({cnt})')
    ax.axis('off')
    ax.imshow(im)
    cnt += 1

  plt.axis('off')
  plt.savefig(mask_root+exp+"png") 

for f in files:
  if f.endswith(".npy"):
    print(mask_root+f)
    results[f.replace(".npy","")] = np.load(mask_root+f, allow_pickle=True).item()


for exp, result in results.items():
  img_lst = result.keys()
  visualize(img_lst, result, exp)
