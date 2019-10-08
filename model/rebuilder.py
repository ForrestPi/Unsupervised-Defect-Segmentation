"""inference container

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np



class Network(nn.Module):
    def __init__(self, net, seg):
        super(Network, self).__init__()
        self.net = net
        self.seg = seg

    def forward(self, x):
        out = self.net(x)

        return out


class Rebuilder:
    def __init__(self, net, gpu_id=0, fp16=False):
        self.net = net
        self.gpu_id = gpu_id
        self.fp16 = fp16
        if fp16 is True:
            self.net.half()
        self.network = Network(self.net, None)
        self.network.eval()
        self.network.cuda(gpu_id)

        torch.backends.cudnn.benchmark = True

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

    def inference(self, input_tensor):
        # inference
        with torch.no_grad():
            input_tensor = input_tensor.cuda(self.gpu_id)
            if self.fp16 is True:
                input_tensor = input_tensor.half()
            out = self.network(input_tensor)
        out = out * 255
        out = out.cpu().numpy()[0]
        out = out.astype(np.uint8)
        return out

