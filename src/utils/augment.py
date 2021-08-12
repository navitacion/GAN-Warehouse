from abc import ABCMeta
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        self.transform = None

    def __call__(self, img, phase='train'):
        transformed = self.transform[phase](image=img)
        transformed_img = transformed['image']

        return transformed_img


class ImageTransform(BaseTransform):
    def __init__(self, cfg):
        super(ImageTransform, self).__init__()

        aug_config = cfg.aug

        transform_train_list = [getattr(A, name)(**kwargs) for name, kwargs in dict(aug_config.aug_train).items()]
        transform_train_list.append(ToTensorV2())
        transform_val_list = [getattr(A, name)(**kwargs) for name, kwargs in dict(aug_config.aug_val).items()]
        transform_val_list.append(ToTensorV2())
        transform_test_list = [getattr(A, name)(**kwargs) for name, kwargs in dict(aug_config.aug_test).items()]
        transform_test_list.append(ToTensorV2())

        self.transform = {
            'train': A.Compose(transform_train_list, p=1.0),
            'val': A.Compose(transform_val_list, p=1.0),
            'test': A.Compose(transform_test_list, p=1.0)
        }

class PROGANImageTransform(BaseTransform):
    def __init__(self, img_size):
        super(PROGANImageTransform, self).__init__()

        transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ]

        self.transform = {
            'train': A.Compose(transforms),
            'val': A.Compose(transforms),
            'test': A.Compose(transforms)
        }