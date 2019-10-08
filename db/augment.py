"""Augmentation function

author: Haixin wang
e-mail: haixinwa@gmail.com
"""

import cv2
import random
import numpy as np


def mirror(image):
    image_m = image[:, ::-1]

    return image_m


def flip(image):
    image_f = image[::-1, :]

    return image_f


def rotation(image, range):
    _h, _w = image.shape[0: 2]
    center = (_w // 2, _h // 2)
    rot = random.uniform(range[0], range[1])
    M = cv2.getRotationMatrix2D(center, rot, 1)
    image_r = cv2.warpAffine(image, M, (_w, _h), borderMode=cv2.BORDER_REPLICATE)

    return image_r


def shift(image, dis):
    _h, _w = image.shape[0:2]
    y_s = random.uniform(dis[0], dis[1])
    x_s = random.uniform(dis[0], dis[1])
    M = np.float32([[1, 0, x_s], [0, 1, y_s]])
    image_s = cv2.warpAffine(image, M, (_w, _h), borderMode=cv2.BORDER_REPLICATE)

    return image_s


def lighting_adjust(image, k, b):
    slope = random.uniform(k[0], k[1])
    bias = random.uniform(b[0], b[1])
    image = image * slope + bias
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def normalize_(image, mean, std):
    image -= mean
    image /= std


def crop(image, crop_size):
    height, width, _ = image.shape
    x_offset = random.randint(0, width - crop_size[0])
    y_offset = random.randint(0, height - crop_size[1])

    return image[y_offset: y_offset+crop_size[1], x_offset: x_offset+crop_size[0]]
