"""Residual Encoder-Decoder Networks

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import torch.nn as nn


class BNConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BNConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BNDeConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BNDeConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                       output_padding=out_padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RED_Net(nn.Module):
    def __init__(self, code_dim, img_channel):
        super(RED_Net, self).__init__()
        self.conv0 = BNConv(in_planes=img_channel, out_planes=32, kernel_size=3, stride=2, padding=1)
        self.conv1 = BNConv(in_planes=32, out_planes=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = BNConv(in_planes=64, out_planes=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = BNConv(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = BNConv(in_planes=64, out_planes=128, kernel_size=1, stride=1, padding=0)
        self.conv5 = BNConv(in_planes=128, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv6 = BNConv(in_planes=128, out_planes=128, kernel_size=1, stride=1, padding=0)
        self.conv7 = BNConv(in_planes=128, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv8 = BNConv(in_planes=128, out_planes=128, kernel_size=1, stride=1, padding=0)
        self.conv9 = BNConv(in_planes=128, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv10 = BNConv(in_planes=128, out_planes=code_dim, kernel_size=1, stride=1, padding=0)
        self.conv_f = BNConv(in_planes=code_dim, out_planes=code_dim, kernel_size=3, stride=1, padding=1)
        self.deconv0 = BNDeConv(in_planes=code_dim, out_planes=128, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.deconv1 = BNDeConv(in_planes=128, out_planes=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = BNDeConv(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.deconv3 = BNDeConv(in_planes=128, out_planes=128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = BNDeConv(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.deconv5 = BNDeConv(in_planes=128, out_planes=128, kernel_size=4, stride=2, padding=1)
        self.deconv6 = BNDeConv(in_planes=128, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.deconv7 = BNDeConv(in_planes=64, out_planes=64, kernel_size=4, stride=2, padding=1)
        self.deconv8 = BNDeConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.deconv9 = BNDeConv(in_planes=64, out_planes=32, kernel_size=4, stride=2, padding=1)
        self.deconv10 = BNDeConv(in_planes=32, out_planes=32, kernel_size=4, stride=2, padding=1)
        self.deconv_f = BNDeConv(in_planes=32, out_planes=img_channel, kernel_size=1, stride=1, padding=0, relu=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # conv and down-sampling
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        c4 = self.conv4(x)
        x = self.conv5(c4)
        c6 = self.conv6(x)
        x = self.conv7(c6)
        c8 = self.conv8(x)
        x = self.conv9(c8)
        c10 = self.conv10(x)
        x = self.conv_f(c10)

        # deconv and up-sampling
        x = self.deconv0(x + self.pool0(c10))
        x = self.deconv1(x)
        x = self.deconv2(x + self.pool2(c8))
        x = self.deconv3(x)
        x = self.deconv4(x + self.pool4(c6))
        x = self.deconv5(x)
        x = self.deconv6(x + self.pool6(c4))
        x = self.deconv7(x)
        x = self.deconv8(x)
        x = self.deconv9(x)
        x = self.deconv10(x)
        x = self.deconv_f(x)
        out = self.activate(x)

        return out
