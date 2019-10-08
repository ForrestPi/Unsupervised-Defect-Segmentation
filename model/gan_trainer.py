"""training container

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, g_net, d_net, g_loss, d_loss):
        super(Network, self).__init__()
        self.g_net = g_net
        self.d_net = d_net
        self.g_loss = g_loss
        self.d_loss = d_loss

    def forward(self, input_tensor, phase):
        if phase == 'discriminate':
            hr_img = input_tensor
            sr_img = self.g_net(hr_img)
            real_out = self.d_net(hr_img)
            fake_out = self.d_net(sr_img)
            d_loss = self.d_loss(real_out, fake_out)

            return d_loss

        elif phase == 'generate':
            hr_img = input_tensor
            sr_img = self.g_net(hr_img)
            fake_out = self.d_net(sr_img)
            image_loss, vgg_loss, adversarial_loss, tv_loss = self.g_loss(fake_out, sr_img, hr_img)

            return image_loss, vgg_loss, adversarial_loss, tv_loss

        else:
            raise Exception("Wrong phase!")


class Trainer:
    def __init__(self, g_net, d_net, g_loss, d_loss, optimizerG, optimizerD, ngpu):
        self.g_net = g_net.train()
        self.d_net = d_net.train()
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.network = torch.nn.DataParallel(Network(self.g_net, self.d_net, g_loss, d_loss),
                                             device_ids=list(range(ngpu)))
        self.network.cuda()
        self.loss_value = None

    def save_params(self, save_path):
        print("saving model to {}".format(save_path))
        with open((save_path % 'GNet'), "wb") as f:
            params = self.g_net.state_dict()
            torch.save(params, f)
        with open((save_path % 'DNet'), "wb") as f:
            params = self.g_net.state_dict()
            torch.save(params, f)

    def set_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizerG.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizerD.param_groups:
            param_group["lr"] = lr

    def train(self, input_tensor, phase):
        # Update Discriminator
        if phase == 'discriminate':
            self.optimizerD.zero_grad()
            d_loss = self.network(input_tensor, phase=phase)
            d_loss = d_loss.mean()
            d_loss.backward()
            self.optimizerD.step()
            self.loss_value = [d_loss.item()]

        # Update Generator
        elif phase == 'generate':
            self.optimizerG.zero_grad()
            image_loss, vgg_loss, adversarial_loss, tv_loss = self.network(input_tensor, phase=phase)
            g_loss = image_loss + (6e-3 * vgg_loss) + (1e-3 * adversarial_loss) + (2e-8 * tv_loss)
            g_loss = g_loss.mean()
            g_loss.backward()
            self.optimizerG.step()
            self.loss_value += [image_loss.mean().item(), vgg_loss.mean().item(),
                                adversarial_loss.mean().item(), tv_loss.mean().item()]

        else:
            raise Exception("Wrong phase!")

    def get_loss_message(self):
        mes = 'D loss:{:.4f}; image_loss:{:.4f}; vgg_loss:{:.4f}; adversarial_loss:{:.4f}; tv_loss:{:.4f}'.format(self.loss_value[0],
                                                                                                                  self.loss_value[1],
                                                                                                                  self.loss_value[2],
                                                                                                                  self.loss_value[3],
                                                                                                                  self.loss_value[4],)

        return mes