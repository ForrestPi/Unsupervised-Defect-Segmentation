import torch


def load_params(net, path):
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
    net.load_state_dict(new_state_dict)


def load_data_set_from_factory(configs, phase):
    if configs['db']['name'] == 'mvtec':
        from db import MVTEC, MVTEC_pre, MVTEC_with_val
        if configs['db']['use_validation_set'] == True:
            if phase == 'train':
                set_name = configs['db']['train_split']
                preproc = MVTEC_pre(resize=tuple(configs['db']['resize']))
            elif phase == 'validation':
                set_name = configs['db']['validation_split']
                preproc = None
            elif phase == 'test':
                set_name = configs['db']['val_split']
                preproc = None
            else:
                raise Exception("Invalid phase name")
            set = MVTEC_with_val(root=configs['db']['data_dir'], set=set_name, preproc=preproc)
        elif configs['db']['use_validation_set'] == False:
            if phase == 'train':
                set_name = configs['db']['train_split']
                preproc = MVTEC_pre(resize=tuple(configs['db']['resize']))
            elif phase == 'validation':
                pass
            elif phase == 'test':
                set_name = configs['db']['val_split']
                preproc = None
            else:
                raise Exception("Invalid phase name")
            set = MVTEC(root=configs['db']['data_dir'], set=set_name, preproc=preproc)
        else:
            raise Exception("Invalid input")
    elif configs['db']['name'] == 'chip':
        from db import CHIP, CHIP_pre
        if phase == 'train':
            set_name = configs['db']['train_split']
            preproc = CHIP_pre(resize=tuple(configs['db']['resize']))
        elif phase == 'test':
            set_name = configs['db']['val_split']
            preproc = None
        else:
            raise Exception("Invalid phase name")
        set = CHIP(root=configs['db']['data_dir'], set=set_name, preproc=preproc)
    else:
        raise Exception("Invalid set name")

    return set


def load_training_net_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'SSIM_Net_lite':
        from model.networks import SSIM_Net_Lite
        net = SSIM_Net_Lite(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'RED_Net_2skips':
        from model.networks import RED_Net_2skips
        net = RED_Net_2skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'RED_Net_3skips':
        from model.networks import RED_Net_3skips
        net = RED_Net_3skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'RED_Net_4skips':
        from model.networks import RED_Net_4skips
        net = RED_Net_4skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'VAE_Net0':
        from model.networks import VAE_Net0
        net = VAE_Net0(code_dim=configs['model']['code_dim'],phase='train')
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    elif configs['model']['name'] == 'SRGAN':
        from model.networks import SR_G, SR_D
        sr_G = SR_G(configs['model']['upscale_factor'])
        sr_D = SR_D()
        optimizerG = torch.optim.Adam(sr_G.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(sr_D.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return sr_G, sr_D, optimizerG, optimizerD
    else:
        raise Exception("Invalid model name")


def load_loss_from_factory(configs):
    if configs['op']['loss'] == 'SSIM_loss':
        from model.loss import SSIM_loss
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=configs['model']['img_channel'])

        return loss

    elif configs['op']['loss'] == 'Multi_SSIM_loss':
        from model.loss import Multi_SSIM_loss
        loss = Multi_SSIM_loss(window_sizes=configs['op']['window_size'], channel=configs['model']['img_channel'])

        return loss

    elif configs['op']['loss'] == 'VAE_loss':
        from model.loss import VAE_loss
        loss = VAE_loss()

        return loss

    elif configs['op']['loss'] == 'SRGAN_loss':
        from model.loss import SRGAN_Gloss, SRGAN_Dloss
        from model.networks import VGG16
        vgg16 = VGG16(in_channels=configs['model']['img_channel'])
        load_params(vgg16.layers, configs['op']['vgg16_weight_path'])
        g_loss = SRGAN_Gloss(vgg16)
        d_loss = SRGAN_Dloss()

        return g_loss, d_loss

    else:
        raise Exception('Wrong loss name')


def load_training_model_from_factory(configs, ngpu):
    if configs['model']['type'] == 'Encoder':
        from model.trainer import Trainer
        net, optimizer = load_training_net_from_factory(configs)
        loss = load_loss_from_factory(configs)
        trainer = Trainer(net, loss, configs['op']['loss'], optimizer, ngpu)
    elif configs['model']['type'] == 'GAN':
        from model.gan_trainer import Trainer
        sr_G, sr_D, optimizerG, optimizerD = load_training_net_from_factory(configs)
        g_loss, d_loss = load_loss_from_factory(configs)
        trainer = Trainer(sr_G, sr_D, g_loss, d_loss, optimizerG, optimizerD, ngpu)
    else:
        raise Exception("Wrong model type!")

    return trainer


def load_test_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'SSIM_Net_lite':
        from model.networks import SSIM_Net_Lite
        net = SSIM_Net_Lite(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_2skips':
        from model.networks import RED_Net_2skips
        net = RED_Net_2skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_3skips':
        from model.networks import RED_Net_3skips
        net = RED_Net_3skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_4skips':
        from model.networks import RED_Net_4skips
        net = RED_Net_4skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'VAE_Net0':
        from model.networks import VAE_Net0
        net = VAE_Net0(code_dim=configs['model']['code_dim'],phase='inference')
    else:
        raise Exception("Invalid model name")

    return net
