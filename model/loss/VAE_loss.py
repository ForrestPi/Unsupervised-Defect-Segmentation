import torch
import torch.nn as nn



class VAE_loss(nn.Module):
    def __init__(self):#
        super(VAE_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self, imag1_mu_logvar, imag2):#imag1, imag2, mu, logvar
        #reconstruction_function.size_average = False
        #imag1_mu_logvar[0]=imag1  imag1_mu_logvar[0]=mu  imag1_mu_logvar[0]=logvar
        BCE = self.bce_loss(imag1_mu_logvar[0].view(-1,256*256*3), imag2.view(-1,256*256*3))#####
        KLD_element = imag1_mu_logvar[1].pow(2).add_(imag1_mu_logvar[2].exp()).mul_(-1).add_(1).add_(imag1_mu_logvar[2])
        KLD = torch.mean(KLD_element).mul_(-0.5)
        return BCE + KLD