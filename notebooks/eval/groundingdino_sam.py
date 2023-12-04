

import pandas as pd
import torch
import os
from PIL import Image
import copy

from tqdm import tqdm
repo_root = "../../"
sam_path = repo_root + "sam_vit_h_4b8939.pth"
Dataset_path = repo_root + "datasets/people_poses/"
HOME = "/home/hongxin/"
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append(os.path.join(HOME, "GroundingDINO"))
from groundingdino.util.inference import Model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


from typing import List

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = sam_path
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def getPredictions(promptValue, image):
  prompts = [promptValue]
  # inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
  # # predict
  # with torch.no_grad():
  #   outputs = model(**inputs)
  BOX_TRESHOLD = 0.35
  TEXT_TRESHOLD = 0.25
  #image = cv2.imread(SOURCE_IMAGE_PATH)

  pil_image = image.convert('RGB')
  open_cv_image = np.array(image)
  # Convert RGB to BGR
  open_cv_image = open_cv_image[:, :, ::-1].copy()

  detections = grounding_dino_model.predict_with_classes(
  image=open_cv_image,
  classes=enhance_class_name(class_names=prompts),
  box_threshold=BOX_TRESHOLD,
  text_threshold=TEXT_TRESHOLD
  )


  if len(detections.xyxy) > 0:
    return detections.xyxy[0]
  return [0, 0, 0, 0]


def make_square_by_padding(img, fill_color=(0, 0, 0)):
    # img = Image.open(requests.get(url, stream=True).raw)
    width, height = img.size

    # Determine the size for the square
    new_size = max(width, height)

    # Create a new image with the desired size and fill color
    new_img = Image.new("RGB", (new_size, new_size), fill_color)

    # Paste the original image onto the center of the new image
    new_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))

    return new_img
import cv2
from PIL import Image
import numpy as np
def processPredictionImage(pred, img):
    processed_tensor = torch.sigmoid(torch.reshape(pred, (352, 352)))
    image_np = (processed_tensor.detach().numpy() * 255).astype(np.uint8)

    _, binary_image = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)

    desired_width, desired_height = img.size
    resized_image = cv2.resize(binary_image, (max(desired_width, desired_height),  max(desired_width, desired_height)), interpolation=cv2.INTER_AREA)

    # Get dimensions of the binary image
    height, width = resized_image.shape[:2]

    # Check if the desired crop size is smaller than the original image size
    if desired_width <= width and desired_height <= height:
        # Calculate the top-left corner of the crop
        x = width // 2 - desired_width // 2
        y = height // 2 - desired_height // 2

        # Crop the image
        cropped_image = resized_image[y:y+desired_height, x:x+desired_width]
        return cropped_image

    else:
        print("failed to crop image")
        return resized_image
    
def getBoundingBox(predictionImage):
    contours, _ = cv2.findContours(predictionImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        list_of_pts = []
        for ctr in contours:
            list_of_pts += [pt[0] for pt in ctr]
        ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
        # largest_contour = max(cv2.convexHull(ctr), key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cv2.convexHull(ctr))

        # print(f"Bounding Box: x={x}, y={y}, width={(w)}, height={(h)}")
        return [x, y, (w + x), (h + y)]

    else:
        # print("No contours found")
        return [0, 0, 0, 0]
def getSAMPreditction(image, box):

    input_box = np.array(box)
    if(input_box[0] == 0 and input_box[1] == 0 and input_box[2] == 0 and input_box[3] == 0):
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=False,
        )
    else:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    return masks[0]
def pixAcc(predicted, target):
    same = (predicted == target).sum()
    w, h = target.shape
    print("Target seg shape: {}, Predicted seg shape: {}, #Same pixels: {}".format(target.shape, predicted.shape, same))
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
# *****
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
def getFinalPredictions(images, promptVals):
  masks = []
  emptyPrompt = ""
  for img in (pbar := tqdm(images)):
    imageMasks = {}
    image_np = np.array(img)
    image2 = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    predictor.set_image(image2)

    for prompt in promptVals:
      bbox = getPredictions(prompt, img)
      if(bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0 and emptyPrompt == ""):
        mask = getSAMPreditction(img, bbox)
        imageMasks[prompt] = mask
        emptyPrompt = prompt
      elif(bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0):
        imageMasks[prompt] = imageMasks[emptyPrompt]
      else:
        mask = getSAMPreditction(img, bbox)
        imageMasks[prompt] = mask

    emptyPrompt = ""
    masks.append(imageMasks)
  return  masks

def getImages(num):
    images = []
    truthMasks = []
    data_list = []
    root = Dataset_path
    imageFile = "val_images/"
    segmentationFile = "val_segmentations/"
    with open(os.path.join(root, f"val_id.txt"), 'r') as lf:
        data_list = [ s.strip() for s in lf.readlines() ]

    try:
        for data_name in (pbar := tqdm(data_list[:num])):
            img_path = root + imageFile + data_name + '.jpg'
            seg_path = root +  segmentationFile+ data_name + '.png'
            # Read Image and Ground truth mask
            img = copy.deepcopy(Image.open(img_path))

            # display(img)
            if img is None:
                print("\nimage is None", data_name)
                continue
            else:
                images.append(img)
            mask_gt = cv2.imread(seg_path)
            if mask_gt is None:
                print("\nmask_gt is None", data_name)
                continue
            else:
                truthMasks.append(mask_gt)
    except Exception as e:
        print("ERROR")
        print(e)
    return images, truthMasks

def get_masked_image(original_image, segmentation):
  # Visualize

  overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
  overlay_color = (255, 0, 0, 200)

  draw = ImageDraw.Draw(overlay_image)
  segmentation_mask_image = Image.fromarray(segmentation.astype('uint8') * 255)
  draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

  return Image.alpha_composite(original_image.convert('RGBA'), overlay_image)



def evaluate(images, masks, truthMasks):
  # Get confidence scores for each masks generated by SAM for
  # each object label existing in the given image

  results = []
  label_name = ["Background","Hat","Hair","Glove",
        "Sunglasses","UpperClothes","Dress","Coat","Socks","Pants",
        "Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]


  for i in range(len(truthMasks)):
    anns = gt_to_anns_of_label_mask(truthMasks[i])
    for ann in anns:
      mask = masks[i]
      image = images[i]
      label =  label_name[ann['label']]
      print(label_name[ann['label']])
      # Get Corresponding gt Mask, generated Mask for Evaluation
      print(mask[label])
      iou = IOU(mask[label],ann['segmentation'])
      print(iou)
      result_image = get_masked_image(image, mask[label])
      # display(result_image)
      pixacc = pixAcc(mask[label],ann['segmentation'])
      result = {
          "iou": iou,
          "pixacc": pixacc,
          "label_num": ann["label"],
          "label_name": label,
          "masked_img": result_image,
          "pred_mask": mask[label],
          "gt_mask": ann
      }
      print("iou:{}, pixacc:{}, label num:{}, label_name:{}".format(iou, pixacc, ann["label"], label))
      results.append(result)

  return results

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
    LABELS = ["Background","Hat","Hair","Glove",
        "Sunglasses","UpperClothes","Dress","Coat","Socks","Pants",
        "Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]

    for i, label_name in enumerate(LABELS):
        mask_i = masks.get(label_name, empty)
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


def getImages2(start_idx,end_idx):
    images = []
    truthMasks = []
    data_list2 = []
    root = Dataset_path
    imageFile = "val_images/"
    segmentationFile = "val_segmentations/"
    with open(os.path.join(root, f"val_id.txt"), 'r') as lf:
        data_list = [ s.strip() for s in lf.readlines() ]
    try:
        for data_name in (pbar := tqdm(data_list[start_idx:end_idx])):
            img_path = root + imageFile + data_name + '.jpg'
            seg_path = root +  segmentationFile+ data_name + '.png'

            # Read Image and Ground truth mask
            img = copy.deepcopy(Image.open(img_path))

            # display(img)
            if img is None:
                print("\nimage is None", data_name)
                continue
            else:
                data_list2.append(data_name)
                images.append(img)
            mask_gt = cv2.imread(seg_path)
            if mask_gt is None:
                print("\nmask_gt is None", data_name)
                continue
            else:
                truthMasks.append(mask_gt)
    except Exception as e:
        print("ERROR")
        print(e)
    return images, truthMasks, data_list2




def main():

    miou_table = []
    pix_acc_table = []
    num_slice = 10
    num_data_size = 10000
    batch_size = int( num_data_size / num_slice)
    for i in range(num_slice):
        start = i * batch_size
        end = (i+1) * batch_size
        images, truthMasks, data_list = getImages2(start,end)
        bodyPrompts = ["Background","Hat","Hair","Glove", "Sunglasses","UpperClothes","Dress","Coat","Socks","Pants", "Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]
        masks = getFinalPredictions(images, bodyPrompts)
        for data_idx in range(len(data_list)):
            miou, pix_acc = compute_metric(data_list[data_idx], masks[data_idx], truthMasks[data_idx][:,:,0])
            miou_table.append(miou)
            pix_acc_table.append(pix_acc)
    miou_table = pd.DataFrame(miou_table, columns=miou_table[0].keys()).set_index('name')
    miou_table.to_csv('groundingdino_sam_result/miou.csv')

    pix_acc_table = pd.DataFrame(pix_acc_table, columns=pix_acc_table[0].keys()).set_index('name')
    pix_acc_table.to_csv('groundingdino_sam_result/pix_acc.csv')
if __name__ == '__main__':
    main()