import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as tvF

from he.cfg import Config
from he.constants import STD, MEAN
from he.data.augmentations import get_maximal_crop


class SimCLRNumPyDataset(Dataset):
    def __init__(self, images, transform, config: Config):
        self.images = images
        self.transform = transform
        self.config = config
        size = config.data.image_size

        self.affine_transform = T.RandomAffine(degrees=0)

        self.crop_resize = T.Resize((size, size))

        self.norm = T.Normalize(MEAN, STD)

        self.tt = T.ToTensor()

        self.tp = T.ToPILImage()

    def __len__(self):
        return len(self.images)

    def generate_affine_sample(self, x):
        ret = self.affine_transform.get_params(
            degrees=[-self.config.data.max_degrees, self.config.data.max_degrees],
            translate=[self.config.data.max_translate, self.config.data.max_translate],
            scale_ranges=[self.config.data.min_scale, self.config.data.max_scale],
            shears=[
                -self.config.data.max_shear, self.config.data.max_shear,
                -self.config.data.max_shear, self.config.data.max_shear
            ],
            img_size=x.size
        )
        angle, (tx, ty), scale, shear = ret

        if self.config.data.crop_after_transform:
            x_affine = self.crop_resize(get_maximal_crop(x, angle, scale, (tx, ty), shear))
        else:
            x_affine = tvF.affine(x, *ret, interpolation=tvF.InterpolationMode.NEAREST, fill=0, center=None)

        angle = angle / self.config.data.max_degrees
        tx = tx / self.config.data.image_size
        ty = ty / self.config.data.image_size
        shearx = shear[0] / self.config.data.max_shear
        sheary = shear[1] / self.config.data.max_shear

        affine_params = torch.from_numpy(np.array([
            angle, tx, ty, scale, shearx, sheary
        ])).float()

        return self.tt(x), self.tt(x_affine), affine_params

    def __getitem__(self, item):
        image = Image.fromarray(self.images[item])

        x1t, x2t = self.transform(image)
        x1, x2 = self.tp(x1t), self.tp(x2t)

        x1t, x1at, affine_params1 = self.generate_affine_sample(x1)
        x2t = self.tt(x2)

        return x1t, x1at, affine_params1, x2t
