import torch.nn as nn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


class VGG16(nn.Module):
    def __init__(self, in_channels, batch_norm=False):
        super(VGG16, self).__init__()
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        for k in range(len(self.layers)):
            x = self.layers[k](x)

        return x


if __name__ == '__main__':
    vgg16 = VGG16(in_channels=3)
    weight_path = 'D:/DeeCamp/Pro/Defect_Seg/Cascade_rebuilder/weights/vgg16_reducedfc.pth'
    import torch
    dict = torch.load(weight_path)
    vgg16.layers.load_state_dict(dict)
    pass


