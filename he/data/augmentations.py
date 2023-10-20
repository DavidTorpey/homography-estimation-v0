import logging

import numpy as np
from PIL import Image
from lightly.transforms import SimCLRTransform
import torchvision.transforms as T
from skimage import transform

from he.cfg import Config
from he.constants import MEAN, STD
from he.data.maxrect import get_maximal_rectangle, get_intersection

NORMALIZE = T.Normalize(MEAN, STD)


def get_corners(im):
    rows, cols = im.shape[0], im.shape[1]
    return np.array([
        [0, 0],
        [0, rows - 1],
        [cols - 1, rows - 1],
        [cols - 1, 0]
    ])


def get_rot_mat(deg, im):
    shift_y, shift_x = np.array(im.shape[:2]) / 2.
    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(deg))
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    return tf_shift + (tf_rotate + tf_shift_inv)


def get_maximal_crop(image, angle, scale, translations, shear):
    image_np = np.asarray(image)
    height, width = image_np.shape[:2]

    transformation_matrix = get_rot_mat(angle, image_np) + transform.AffineTransform(
        scale=scale, translation=translations, shear=np.deg2rad(shear[0])
    )

    warped = transform.warp(image_np, transformation_matrix.inverse)

    original_corners = get_corners(image_np)
    transformed_corners = transformation_matrix(original_corners)

    coordinates1 = [(x, y) for x, y in transformed_corners]
    _, coordinates = get_intersection([coordinates1, ])
    (x4, y4), (x2, y2) = get_maximal_rectangle(list(coordinates))
    x1 = int(x4)
    y1 = int(y2)
    x3 = int(x2)
    y3 = int(y4)
    aa = (np.clip(x1, 0, width), np.clip(y1, 0, height))
    bb = (np.clip(x3, 0, width), np.clip(y3, 0, height))
    return Image.fromarray(warped[bb[1]:aa[1], aa[0]:bb[0]].astype('uint8'))


def get_transform(transform_name, config: Config):
    logging.info('Initialising transform: %s', transform_name)

    if transform_name == 'simclr':
        transform = SimCLRTransform(
            input_size=config.data.image_size,
            normalize=None
        )
    else:
        raise NotImplementedError(f'Transform {transform_name} not supported.')

    return transform
