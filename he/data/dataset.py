import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision import transforms as tvtransforms
from .gaussian_blur import GaussianBlur
from . import new_transforms as transforms
from ..configuration import Config


class AffineDataset(Dataset):
    def __init__(self, paths, config: Config):
        self.paths = paths
        self.image_size = config.data.image_size
        self.s = config.data.s
        self.config = config

        self.angle_bins = np.linspace(-90, 90, 36)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        x = Image.open(self.paths[item]).convert('RGB')

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

        if self.config.data.shear:
            max_shear = 25
            shear = [-max_shear, max_shear, -max_shear, max_shear]
        else:
            shear = None

        if self.config.data.rotation:
            degrees = 90
        else:
            degrees = 0

        if self.config.data.translation:
            translate = (0.25, 0.25)
        else:
            translate = None

        if self.config.data.scale:
            scale = (0.7, 1.3)
        else:
            scale = None

        random_affine = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear
        )

        x1t, params = random_affine(x1)
        angle, (tx, ty), scale, shear = params

        param_vec_elements = []
        if self.config.data.rotation:
            angle = angle / 360
            param_vec_elements.append(angle)
        if self.config.data.translation:
            tx = tx / self.image_size
            ty = ty / self.image_size
            param_vec_elements.append(tx)
            param_vec_elements.append(ty)
        if self.config.data.scale:
            param_vec_elements.append(scale)
        if self.config.data.shear:
            shearx = shear[0] / max_shear
            sheary = shear[1] / max_shear
            param_vec_elements.append(shearx)
            param_vec_elements.append(sheary)

        assert len(param_vec_elements) > 0, 'One of rotation, translation, scale, shear should be enabled.'

        param_vec = torch.from_numpy(np.array(param_vec_elements)).float()

        x1 = to_tensor(x1)
        x2 = to_tensor(x2)
        x1t = to_tensor(x1t)

        return x1, x2, x1t, param_vec


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


class DefaultDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = Image.open(self.paths[item]).convert('RGB')
        dummy = 1
        return [self.transform(image), self.transform(image)], dummy
