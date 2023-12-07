import numpy as np
from PIL import Image
from skimage import transform
from torchvision.transforms.functional import _get_inverse_affine_matrix

from he.data.maxrect import get_intersection, get_maximal_rectangle


def get_corners(im):
    rows, cols = im.shape[0], im.shape[1]
    return np.array([
        [0, 0],
        [0, rows - 1],
        [cols - 1, rows - 1],
        [cols - 1, 0]
    ])


def bounded_affine(image, angle, translation, scale, shear):
    image_np = np.asarray(image)
    height, width = image_np.shape[:2]

    center = (image.size[0] * 0.5 + 0.5, image.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translation, scale, shear, inverted=False)

    transformation_matrix = transform.AffineTransform(
        matrix=np.array(matrix + [0, 0, 1]).reshape(3, 3)
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
    new = Image.fromarray((warped[bb[1]:aa[1], aa[0]:bb[0]] * 255.0).astype('uint8'))
    return new
