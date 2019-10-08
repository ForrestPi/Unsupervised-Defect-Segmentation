"""Autoencoder with ssim loss

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import torch
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


class Encoder(nn.Module):
    def __init__(self, code_dim, img_channel):
        super(Encoder, self).__init__()
        self.conv1 = BNConv(in_planes=img_channel, out_planes=32, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = BNConv(in_planes=32, out_planes=32, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv3 = BNConv(in_planes=32, out_planes=32, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv4 = BNConv(in_planes=32, out_planes=64, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation4 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv5 = BNConv(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation5 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv6 = BNConv(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=1, relu=False)
        self.activation6 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv7 = BNConv(in_planes=64, out_planes=code_dim, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation7 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        x = self.activation3(self.conv3(x))
        x = self.activation4(self.conv4(x))
        x = self.activation5(self.conv5(x))
        x = self.activation6(self.conv6(x))
        x = self.activation7(self.conv7(x))

        return x


class Decoder(nn.Module):
    def __init__(self, code_dim, img_channel):
        super(Decoder, self).__init__()
        self.deconv1 = BNDeConv(in_planes=code_dim, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv2 = BNDeConv(in_planes=64, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv3 = BNConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv4 = BNDeConv(in_planes=64, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation4 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv5 = BNDeConv(in_planes=64, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation5 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv6 = BNConv(in_planes=64, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation6 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv7 = BNDeConv(in_planes=32, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation7 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv8 = BNDeConv(in_planes=32, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation8 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv9 = BNConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation9 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv10 = BNConv(in_planes=32, out_planes=img_channel, kernel_size=1, stride=1, padding=0, relu=False)

    def forward(self, x):
        x = self.activation1(self.deconv1(x))
        x = self.activation2(self.deconv2(x))
        x = self.activation3(self.deconv3(x))
        x = self.activation4(self.deconv4(x))
        x = self.activation5(self.deconv5(x))
        x = self.activation6(self.deconv6(x))
        x = self.activation7(self.deconv7(x))
        x = self.activation8(self.deconv8(x))
        x = self.activation9(self.deconv9(x))
        x = self.deconv10(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class SSIM_Net(nn.Module):
    def __init__(self, code_dim, img_channel):
        super(SSIM_Net, self).__init__()
        self.encoder = Encoder(code_dim, img_channel)
        self.decoder = Decoder(code_dim, img_channel)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
