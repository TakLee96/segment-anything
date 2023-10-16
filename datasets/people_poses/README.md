# Dataset Overview

## 1. Overview
Look into Person (LIP) is a new large-scale dataset, focus on semantic understanding of person. Following are the detailed descriptions.

### 1.1 Volume
The dataset contains 50,000 images with elaborated pixel-wise annotations with 19 semantic human part labels and 2D human poses with 16 key points.

### 1.2 Diversity
The annotated 50,000 images are cropped person instances from [COCO dataset](http://mscoco.org/home/) with size larger than 50 * 50.The images collected from the real-world scenarios contain human appearing with challenging poses and views, heavily occlusions, various appearances and low-resolutions. We are working on collecting and annotating more images to increase diversity.

## 2. Download
### 2.1 Single Person
We have divided images into three sets. 30462 images for training set, 10000 images for validation set and 10000 for testing set.The dataset is available at [Google Drive (no it actually is not available)](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing) and [Baidu Drive](http://pan.baidu.com/s/1nvqmZBN).

Besides we have another large dataset mentioned in "Human parsing with contextualized convolutional neural network." ICCV'15, which focuses on fashion images. You can download the dataset including 17000 images as extra training data.

## Label Order

1. Hat
2. Hair
3. Glove
4. Sunglasses
5. UpperClothes
6. Dress
7. Coat
8. Socks
9. Pants
10. Jumpsuits
11. Scarf
12. Skirt
13. Face
14. Left-arm
15. Right-arm
16. Left-leg
17. Right-leg
18. Left-shoe
19. Right-shoe


# Download Instructions

From @TakLee96: you should download these files:
```
Testing_images.zip
TrainVal_images.zip
TrainVal_parsing_annotations.zip
TrainVal_pose_annotations.zip
```

Unzip them to obtain
```
pose_annotations/
  lip_train_set.csv
  lip_val_set.csv
  README.md
  vis_annotation.py

test_id.txt
testing_images/*.jpg

train_id.txt
train_images/*.jpg
train_segmentations/*.png

val_id.txt
val_images/*.jpg
val_segmentations/*.png
```

# Data Format

```python
import cv2
m = cv2.imread('val_segmentations/547336_495052.png')
m.shape
# (414, 372, 3)

import numpy as np
np.unique(m)
# array([ 0,  2,  5,  9, 13, 14, 15], dtype=uint8)

m[0, 278]
# array([2, 2, 2], dtype=uint8)

im = cv2.imread('val_images/547336_495052.jpg')        
im.shape
# (414, 372, 3)
```
