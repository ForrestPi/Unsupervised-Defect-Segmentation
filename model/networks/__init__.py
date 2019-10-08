# Encoder
from .SSIM.ssim import SSIM_Net as SSIM_Net
from .SSIM.ssim_lite import SSIM_Net as SSIM_Net_Lite
from .RED_Net.red_net_2skips import RED_Net as RED_Net_2skips
from .RED_Net.red_net_3skips import RED_Net as RED_Net_3skips
from .RED_Net.red_net_4skips import RED_Net as RED_Net_4skips
from .VAE.VAE import VAE_Net0

# GAN
from .SRGAN.SRGAN import Generator as SR_G
from .SRGAN.SRGAN import Discriminator as SR_D
from .SRGAN.vgg16 import VGG16