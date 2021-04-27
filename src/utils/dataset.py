import cv2
import numpy as np
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    def __init__(self, img_paths, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img, self.phase)

        return img



# CycleGAN Dataset ---------------------------------------------------------------------------
class CycleGANDataset(Dataset):
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
