"""Data set tool of MVTEC

author: mingliangbai
e-mail: mingliangbai@outlook.com
"""
import torch.nn as nn
from torch.autograd import Variable

'''class encoder

class reparametrize

class decoder'''


class VAE_Net0(nn.Module):
    def __init__(self, code_dim,phase):
        super(VAE_Net0, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3, stride=1)
        self.conv51 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.conv52 = nn.Conv2d(48, 96, kernel_size=3, stride=1)

        self.deconv6 = nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1)
        self.deconv7 = nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1)
        self.deconv8 = nn.ConvTranspose2d(24, 12, kernel_size=3, stride=1)
        self.deconv9 = nn.ConvTranspose2d(12, 6, kernel_size=5, stride=1)
        self.deconv10 = nn.ConvTranspose2d(6, 3, kernel_size=5, stride=1)
        self.Leakyrelu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
        self.phase=phase

    def encode(self, x):
        h1 = self.Leakyrelu(self.conv1(x))
        h2 = self.Leakyrelu(self.conv2(h1))
        h3 = self.Leakyrelu(self.conv3(h2))
        h4 = self.Leakyrelu(self.conv4(h3))
        return self.conv51(h4), self.conv52(h4)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h6 = self.Leakyrelu(self.deconv6(z))
        h7 = self.Leakyrelu(self.deconv7(h6))
        h8 = self.Leakyrelu(self.deconv8(h7))
        h9 = self.Leakyrelu(self.deconv9(h8))
        return self.sigmoid(self.deconv10(h9))

    def forward(self, x):
        mu, logvar = self.encode(x)  #
        z = self.reparametrize(mu, logvar)
        imag1_mu_logvar = [self.decode(z), mu, logvar]
        if self.phase=='train':
            return imag1_mu_logvar
        elif self.phase=='inference':
            return self.decode(z)
        else:
            raise Exception("Wrong phase")
