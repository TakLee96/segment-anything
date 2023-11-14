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

import os
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter


# test smaller weight on background and longer epoches
MODEL = 'v5'
device = 'cuda:0'


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
        """ embed: (1, 256, 64, 64)
            label: (1, 256, 256)
        """
        data = np.load(os.path.join(self.embed_dir, self.data_list[index] + ".npz"))
        embed = data['embed']
        embed = torch.as_tensor(embed)
        label = data['label']
        # NOTE(jiahang): uint8 cannot handle -1, other int fails F.interpolate
        label = torch.as_tensor(label).to(torch.float32)
        label = F.interpolate(label[None, None, ...], (256, 256), mode='nearest')
        return embed, label[0][0].to(torch.int64)


# TODO(jiahang): fix magic numbers
LOSS_WEIGHT = np.ones(shape=(20,), dtype=np.float32)
LOSS_WEIGHT[0] = 0.1
LOSS_WEIGHT = torch.as_tensor(LOSS_WEIGHT).to(device)
def loss_fn(logits, labels):
    """ logits/labels: (B, C, 1024, 1024) """
    return F.cross_entropy(logits, labels, weight=LOSS_WEIGHT, ignore_index=-1, reduction='mean')


global_step = [ 0 ]
def train_one_epoch(epoch_index):
    losses = []
    for data in (pbar := tqdm(dataloader)):
        embeds, labels = data
        embeds = embeds.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embeds)
        loss = loss_fn(logits, labels)
        loss_t = loss.detach().cpu().numpy()
        losses.append(loss_t)
        writer.add_scalar('loss', loss_t, global_step[0])
        global_step[0] += 1
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss_t}')
    print('Average Loss: {}'.format(np.mean(losses)))
    scheduler.step()


dataset = PeoplePosesDataset()
dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

model = Decoder()
model = model.to(device)

os.makedirs(MODEL, exist_ok=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.9)
writer = SummaryWriter(log_dir=MODEL)

for epoch in range(40):
    print('EPOCH {}: LR={}'.format(epoch, optimizer.param_groups[0]['lr']))

    model.train(True)
    train_one_epoch(epoch)

    model_path = f'{MODEL}/model_{epoch}.pth'
    torch.save(model.state_dict(), model_path)
