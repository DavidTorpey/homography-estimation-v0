from torchvision.transforms import transforms

from .gaussian_blur import GaussianBlur
from ..configuration import Config


def get_simclr_data_transforms(config: Config):
    image_size = config.data.image_size
    s = config.data.s

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * image_size)),
                                          transforms.ToTensor()])
    return data_transforms
