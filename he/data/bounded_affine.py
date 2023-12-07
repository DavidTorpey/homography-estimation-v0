import traceback

import numpy as np
import torch
from PIL import Image
from skimage import transform
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import perspective
import cv2 as cv

from he.data.maxrect import get_intersection, get_maximal_rectangle


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


class BoundedRandomAffineTransformation:
    def __init__(self, bounded_homog_p: float, distortion_scale: float):
        self.bounded_homog_p = bounded_homog_p
        self.distortion_scale = distortion_scale
        self.rp = T.RandomAffine(distortion_scale=self.distortion_scale)

    def __call__(self, image_pil):
        image_init_np = np.array(image_pil)

        width, height = image_pil.size

        resize = T.Resize(image_pil.size)

        points = None
        image_warped = image_pil
        if torch.rand(1) < self.bounded_homog_p:
            points = self.rp.get_params(width, height, self.distortion_scale)
            startpoints, endpoints = points
            image_warped = perspective(image_pil, startpoints, endpoints, InterpolationMode.BILINEAR, 0)
        image_warped_np = np.array(image_warped)

        proj_mat = np.zeros(9)

        output = image_warped_np
        if points is not None:
            startpoints, endpoints = points
            homog = transform.ProjectiveTransform(matrix=cv.getPerspectiveTransform(
                np.array(startpoints).astype('float32'), np.array(endpoints).astype('float32')
            ))
            proj_mat = homog.params.ravel()
            proj_mat = proj_mat / np.linalg.norm(proj_mat)
            original_corners = get_corners(image_init_np)
            transformed_corners = homog(original_corners)
            coordinates1 = [(x, y) for x, y in transformed_corners]
            _, coordinates = get_intersection([coordinates1, ])
            try:
                (x4, y4), (x2, y2) = get_maximal_rectangle(list(coordinates))
                x1 = int(x4)
                y1 = int(y2)
                x3 = int(x2)
                y3 = int(y4)
                height, width = image_init_np.shape[:2]
                aa = (np.clip(x1, 0, width), np.clip(y1, 0, height))
                bb = (np.clip(x3, 0, width), np.clip(y3, 0, height))
                bounded_crop_np = image_warped_np[bb[1]:aa[1], aa[0]:bb[0]]
                bounded_crop = resize(Image.fromarray(bounded_crop_np))

                output = np.array(bounded_crop)
            except Exception:
                traceback.print_exc()
                proj_mat = np.zeros(9)
                return image_pil, proj_mat

        return Image.fromarray(output), proj_mat
