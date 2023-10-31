import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision import transforms as tvtransforms
from .gaussian_blur import GaussianBlur
from . import new_transforms as transforms
from ..configuration import Config


class CustomDataset(Dataset):
    def __init__(self, x, config: Config):
        self.x = x
        self.image_size = config.data.image_size
        self.s = config.data.s

        self.angle_bins = np.linspace(-90, 90, 36)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = Image.fromarray(self.x[item])

        color_jitter_ = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        random_resized_crop = transforms.RandomResizedCrop(size=self.image_size)
        random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.RandomApply([color_jitter_], p=0.8)
        random_grayscale = transforms.RandomGrayscale(p=0.2)
        gaussian_blur = GaussianBlur(kernel_size=int(0.1 * self.image_size))
        to_tensor = transforms.ToTensor()

        t = tvtransforms.Compose([
            random_resized_crop,
            random_horizontal_flip,
            color_jitter,
            random_grayscale,
            gaussian_blur
        ])

        x1 = t(x)
        x2 = t(x)

        max_shear = 25
        random_affine = transforms.RandomAffine(
            degrees=90, translate=(0.25, 0.25), scale=(0.7, 1.3),
            shear=[-max_shear, max_shear, -max_shear, max_shear]
        )

        x1t, params = random_affine(x1)
        angle, (tx, ty), scale, shear = params
        angle = angle / 360
        tx = tx / self.image_size
        ty = ty / self.image_size
        shearx = shear[0] / max_shear
        sheary = shear[1] / max_shear

        param_vec = torch.from_numpy(np.array([
            angle, tx, ty, scale, shearx, sheary
        ])).float()

        x1 = to_tensor(x1)
        x2 = to_tensor(x2)
        x1t = to_tensor(x1t)

        return x1, x2, x1t, param_vec
