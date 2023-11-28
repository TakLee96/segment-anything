import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("../..")
from segment_anything.modeling import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom

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


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim,
        transformer,
        num_multimask_outputs,
        activation = nn.GELU,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
        ])

    def forward(
        self,
        image_embeddings,
        image_pe,
    ):
        tokens = self.mask_tokens.weight.unsqueeze(0) # 1, 20, 256
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        mask_tokens_out, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_layer = PositionEmbeddingRandom(128)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=20,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
        )

    def forward(self, img_embed):
        """ img_embed: (B, 256, 64, 64)
            @returns (B, C, 256, 256) logits
        """
        image_pe = self.pe_layer((64, 64)).unsqueeze(0)
        return self.mask_decoder(img_embed, image_pe)


class PeoplePosesDataset(Dataset):
    def __init__(self, mode="val", img_size=1024):
        assert mode == "val"
        self.mode = mode
        self.root = "../../datasets/people_poses"
        self.image_dir = os.path.join(self.root, f"{self.mode}_images")
        self.mask_dir = os.path.join(self.root, f"{self.mode}_segmentations")
        self.embed_dir = os.path.join(self.root, f"{self.mode}_embeds")

        # HACK(jiahang): use a different file list
        with open(os.path.join(self.root, "val_id.short.txt"), 'r') as lf:
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

def convert_to_anns(name, mask, label):
    """ recover mask at original resolution and compute mIoU
        name: data id
        mask: torch.Tensor(size=[20, 256, 256])
        label: np.ndarray(shape=(H, W)) --> numbers from 0 to 19
    """
    h, w = label.shape
    size = max(h, w)
    mask = F.interpolate(
        mask.unsqueeze(0),
        (size, size),
        mode="bilinear",
        align_corners=False,
    )
    mask = torch.permute(mask, [0, 2, 3, 1]).cpu()[0, :h, :w]
    mask = mask.numpy().argmax(axis=-1)

    anns = []
    for i, label_name in enumerate(LABELS):
        mask_i  = (mask == i)
        label_i = (label == i)
        if label_i.sum() == 0:
            continue
        
        anns.append({
            "segmentation": mask_i,
            "label": i,
            "gt": label_i,
        })

    return anns


def collate_fn(data):
    names, embeds, labels = zip(*data)
    return names, torch.stack(embeds, axis=0), labels


def main():
    model = Decoder()
    model.load_state_dict(torch.load('../train/v4/model_19.pth'))
    model.to(device)
    model.eval()

    dataset = PeoplePosesDataset(mode="val")
    # batch size 64 uses about 5GB of GPU RAM
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    result_table = {}
    # evaluation takes about 7 minutes on 4060 GPU
    for names, embeds, labels in tqdm(dataloader):
        embeds = embeds.to(device)
        with torch.no_grad():
            # B, 20, 256, 256
            masks = model(embeds)
            masks = masks.cpu()
        
        for i in range(len(labels)):
            anns = convert_to_anns(names[i], masks[i], labels[i])
            result_table[names[i]] = anns

    np.save('vis_v4.npy', result_table)

if __name__ == '__main__':
    main()
