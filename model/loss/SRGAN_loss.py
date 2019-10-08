#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author: kristine
# data:   2019.07.29
import torch
from torch import nn


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GeneratorLoss(nn.Module):
    def __init__(self, vgg16):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.vgg_net = vgg16
        self.vgg_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = (1 - out_labels).mean()
        # VGG Loss
        vgg_loss = self.vgg_loss(self.vgg_net(out_images), self.vgg_net(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)

        return image_loss, vgg_loss, adversarial_loss, tv_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, real_out, fake_out):
        # d_loss = -1 * (torch.log(real_out) + torch.log(1 - fake_out))
        d_loss = 1 - real_out + fake_out

        return d_loss.mean()