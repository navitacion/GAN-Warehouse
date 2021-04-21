import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# CelebA Dataset ---------------------------------------------------------------------------
class CelebDataset(Dataset):
    def __init__(self, data_dir, ano, transform, phase='train'):
        self.data_dir = data_dir
        self.img_ids = ano['image_id'].values
        self.ano = ano
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.ano)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.data_dir, 'img_align_celeba', 'img_align_celeba', img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = self.transform(img, self.phase)

        return img


# CelebA-HQ Dataset ---------------------------------------------------------------------------
class CelebHQDataset(Dataset):
    def __init__(self, data_dir, transform, phase='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.img_paths = glob.glob(os.path.join(self.data_dir, 'celeba_hq', '**/*.jpg'), recursive=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = self.transform(img, self.phase)

        return img


# Monet Dataset ---------------------------------------------------------------------------
class CycleGanDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths, transform, phase='train'):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        base_img = cv2.imread(base_img_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        style_img = cv2.imread(style_img_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        base_img = self.transform(base_img, self.phase)
        style_img = self.transform(style_img, self.phase)

        return base_img, style_img
