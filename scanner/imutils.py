# scanner/imutils.py
# Image utility functions.

from typing import List, Optional, Tuple

import cv2
import numpy as np


def resize(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize the image using a calculated ratio between the new and old width/height.

    Args:
        image (np.ndarray): The input image.
        width (int, optional): The new width of the image. Defaults to None.
        height (int, optional): The new height of the image. Defaults to None.
        interpolation (int): The interpolation method. Defaults to INTER_AREA.

    Returns:
        The resized image.
    """
    if width is None and height is None:
        return image

    dim = None
    h, w = image.shape[:2]
    if width is None:
        r = height / h
        dim = (int(w * r), height)
    else:
        r = width / w
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=interpolation)


def rotate_without_cropping(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image without cropping.

    Args:
        image (np.ndarray): The input image.
        angle (float): The angle of the rotation in the counterclockwise direction.

    Returns:
        The rotated image.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy

    return cv2.warpAffine(image, M, (nw, nh))


def center_crop(image: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
    """Return center cropped image.

    Args:
        image (np.ndarray): The input image.
        dim (Tuple[int, int]): The dimensions (width, height) of the center.

    Returns:
        The center cropped image.
    """
    ih, iw = image.shape[:2]
    cw, ch = dim

    if iw < cw or ih < ch:
        raise Exception(
            "The center of the image cannot be larger than the image itself."
        )

    x = (iw - cw) // 2
    y = (ih - ch) // 2

    return image[y : y + ch, x : x + cw]


def vconcat_images(images: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    """Concatenate the images vertically.

    Args:
        images (List[np.ndarray]): The list of images.

    Returns:
        A new image and heights of the inner images after resizing.
    """
    min_width = min(img.shape[1] for img in images)
    resized_images = [resize(img, width=min_width) for img in images]
    heights = [img.shape[0] for img in resized_images]
    return cv2.vconcat(resized_images), heights
