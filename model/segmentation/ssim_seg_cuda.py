#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: kristine
# data: 2019.08.04

import torch
import torch.nn.functional as F
from model.loss.SSIM_loss import create_window

def ssim_seg(img1, img2, window_size=11,threshold=128):
    (_, channel_1, _, _) = img1.size()
    (_, channel_2, _, _) = img1.size()
    BGR = torch.Tensor([[[[0.114,0.587,0.299]]]]).reshape((1,-1,1,1))
    if channel_1 == 3:
        img1 = torch.sum(img1 * BGR, dim=1,keepdim=True)

    if channel_2 == 3:
        img2 = img2 * BGR
        img2 = torch.sum(img2 * BGR, dim=1,keepdim=True)


    channel = 1

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    # window = window.type_as(img1)
    mu1 = F.conv2d(img1.type_as(window), window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2.type_as(window), window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d((img1 * img1).type_as(window), window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d((img2 * img2).type_as(window), window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d((img1 * img2).type_as(window), window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mask = torch.randn_like(ssim_map, dtype=torch.float)

    #mask
    mask[ssim_map >= threshold/255] = 0
    mask[ssim_map < threshold/255] = 255

    return mask


