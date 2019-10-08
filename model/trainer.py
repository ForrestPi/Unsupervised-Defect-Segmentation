"""training container

author: Haixin wang
e-mail: haixinwa@gmail.com
"""


import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, images):
        preds = self.model(images)
        loss = self.loss(preds, images)

        return loss


class Trainer():
    def __init__(self, net, loss, loss_name, optimizer, ngpu):
        self.net = net
        self.loss = loss
        self.loss_name = loss_name
        self.loss_value = None
        self.optimizer = optimizer
        self.network = torch.nn.DataParallel(Network(self.net, self.loss), device_ids=list(range(ngpu)))
        self.network.train()
        self.network.cuda()
        torch.backends.cudnn.benchmark = True

    def save_params(self, save_path):
        print("saving model to {}".format(save_path))
        with open(save_path, "wb") as f:
            params = self.net.state_dict()
            torch.save(params, f)

    def load_params(self, path):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        w_dict = torch.load(path)
        for k, v in w_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def set_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, input_tensor):
        if self.loss_name == 'SSIM_loss' or self.loss_name == 'VAE_loss':
            self.optimizer.zero_grad()
            loss = self.network(input_tensor)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.loss_value = loss.item()

        elif self.loss_name == 'Multi_SSIM_loss':
            self.loss_value = list()
            total_loss = list()
            self.optimizer.zero_grad()
            loss_multi = self.network(input_tensor)
            for loss in loss_multi:
                loss = loss.mean()
                total_loss.append(loss)
                self.loss_value.append(loss.item())
            total_loss = torch.stack(total_loss, 0).sum()
            total_loss.backward()
            self.optimizer.step()

        else:
            raise Exception('Wrong loss name')

    def get_loss_message(self):
        if self.loss_name == 'SSIM_loss' or self.loss_name == 'VAE_loss':
            mes = 'ssim loss:{:.4f};'.format(self.loss_value)

        elif self.loss_name == 'Multi_SSIM_loss':
            mes = ''
            for k, loss in enumerate(self.loss_value):
                mes += 'size{:d} ssim loss:{:.4f}; '.format(k, loss)
        else:
            raise Exception('Wrong loss name')

        return mes