###############################################################################
# basic components for building gan networks.
# such as, discriminator, generator, discriminator3D, generator3D
###############################################################################

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Defines the generator G for testing
def define_G_Z(batchSize, OUTPUT_DIM, ngf, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    netG = Generator_Z(batchSize, OUTPUT_DIM, ngf, gpu_ids=gpu_ids)
    
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


##############################################################################
# Classes
##############################################################################


# The generator G
class Generator(nn.Module):
    def __init__(self, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*ngf),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*ngf, 2*ngf, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*ngf, ngf, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(ngf, 1, 10, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.ngf, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        # output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        # return output.view(-1, self.OUTPUT_DIM)
        return output


# The generator G
class Generator3D(nn.Module):
    def __init__(self, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator3D, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*4*ngf),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose3d(4*ngf, 2*ngf, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose3d(2*ngf, ngf, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose3d(ngf, 1, 10, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.ngf, 4, 4, 4)
        # print 'o1: ',output.size()
        output = self.block1(output)
        # print 'o2: ',output.size()
        # output = output[:, :, :7, :7, :7]
        # print 'o3: ',output.size()
        output = self.block2(output)
        # print 'o4: ',output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        # print 'generator3d: ',output.size()
        # return output.view(-1, self.OUTPUT_DIM)
        return output


# The discriminator D
# ndf: number of discriminator filters
class Discriminator(nn.Module):
    def __init__(self, ndf, gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.gpu_ids = gpu_ids

        main = nn.Sequential(
            nn.Conv2d(1, ndf, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(ndf, 2*ndf, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*ndf, 4*ndf, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*ndf, 1)

    def forward(self, input):
        # input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        # print 'o1.shape: ',out.shape
        out = out.view(-1, 4*4*4*self.ndf)
        # print 'o2.shape: ',out.shape
        out = self.output(out)
        # print 'o3.shape: ',out.shape
        # return out.view(-1)
        return out


# The discriminator 3D
# ndf: number of discriminator filters
class Discriminator3D(nn.Module):
    def __init__(self, ndf, gpu_ids=[]):
        super(Discriminator3D, self).__init__()
        self.ndf = ndf
        self.gpu_ids = gpu_ids

        main = nn.Sequential(
            nn.Conv3d(1, ndf, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(),
            nn.Conv3d(ndf, 2*ndf, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            nn.Conv3d(2*ndf, 2*ndf, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        # it is annoying here, we have to change the size if input size changed
        self.output = nn.Linear(2*4*4*4*ndf, 1)

    def forward(self, input):
        #input = input.view(-1, 1, 64, 64, 64)
        out = self.main(input)
        # print 'out1.shape: ',out.shape
        out = out.view(-1, 2*4*4*4*self.ndf)
        # print 'out2.shape: ', out.shape
        out = self.output(out)
        # print 'out3.shape: ',out.shape
        # return out.view(-1)
        return out
    


class NoiseZ(nn.Module):
    def __init__(self, batchSize):
        super(NoiseZ, self).__init__()
        self.Z = nn.Parameter(torch.randn(batchSize, 128), requires_grad=True)

    def forward(self, input):
        out = self.Z * input
        return out

# The generator G for testing
# ngf: number of generator filters
class Generator_Z(nn.Module):
    def __init__(self, batchSize, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator_Z, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*ngf),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*ngf, 2*ngf, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*ngf, ngf, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(ngf, 1, 8, stride=2, padding=1)

        self.noise = NoiseZ(batchSize)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # TODO: multiple GPU
        input_noise = self.noise(input)
        output = self.preprocess(input_noise)
        output = output.view(-1, 4*self.ngf, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        # output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        # return output.view(-1, self.OUTPUT_DIM)
        return output
    
# 3D version The generator G for testing
# ngf: number of generator filters
class Generator3D_Z(nn.Module):
    def __init__(self, batchSize, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator3D_Z, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*4*ngf),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose3d(4*ngf, 2*ngf, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose3d(2*ngf, ngf, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose3d(ngf, 1, 8, stride=2, padding=1)

        self.noise = NoiseZ(batchSize)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # TODO: multiple GPU
        input_noise = self.noise(input)
        output = self.preprocess(input_noise)
        output = output.view(-1, 4*self.ngf, 4, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        # output = output[:, :, :7, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        # return output.view(-1, self.OUTPUT_DIM)
        return output