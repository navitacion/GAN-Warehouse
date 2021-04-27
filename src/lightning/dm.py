import os, glob, random
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.utils.dataset import CycleGANDataset, SingleImageDataset

# DataModule ---------------------------------------------------------------------------
class SingleImageDataModule(pl.LightningDataModule):
    def __init__(self, img_paths, transform, cfg):
        super(SingleImageDataModule, self).__init__()
        self.img_paths = img_paths
        self.transform = transform
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = SingleImageDataset(self.img_paths, self.transform, phase='train')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)



# DataModule ---------------------------------------------------------------------------
class CycleGANDataModule(pl.LightningDataModule):
    def __init__(self, base_img_paths, style_img_paths, transform, cfg, phase='train', seed=0):
        super(CycleGANDataModule, self).__init__()
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.cfg = cfg
        self.phase = phase
        self.seed = seed

    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = CycleGANDataset(self.base_img_paths[:self.cfg.train.step_per_epoch],
                                             self.style_img_paths[:self.cfg.train.step_per_epoch],
                                             self.transform, self.phase)

        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.cyclegan.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True
                          )

