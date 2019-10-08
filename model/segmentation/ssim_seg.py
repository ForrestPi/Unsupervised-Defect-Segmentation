from skimage.measure import compare_ssim
import cv2
import numpy as np


def ssim_seg(ori_img, re_img, win_size=11, gaussian_weights=False):
    """
    input:
    threhold:
    return: s_map: mask
    """
    # convert the images to grayscale
    if len(ori_img.shape) == 3:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    if len(re_img.shape) == 3:
        re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)

    # compute ssim , s: The value of ssim, d: the similar map
    (s, s_map) = compare_ssim(ori_img, re_img,  win_size=win_size, full=True, gaussian_weights=gaussian_weights)
    s_map = np.clip(s_map, 0, 1)

    return s_map


def seg_mask(s_map, threshold=64):
    s_map = (s_map * 255).astype("uint8")
    mask = s_map.copy()
    mask[s_map < threshold] = 255
    mask[s_map >= threshold] = 0
    return mask

