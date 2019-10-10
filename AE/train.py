import torch
from torch import optim,nn
from torchvision.utils import save_image
from autoencoder1 import ResNet_autoencoder,Bottleneck,DeconvBottleneck,weights_init_kaiming
from dataset import get_data_loader
from help_fucntions import get_cuda
import time
import os
import numpy as np
import pickle
import pytorch_ssim


# parameters
im_size = 256
img_path = "/mnt/mfs/yiling/EL_surface"
batch_size = 48
save_path = "ae_model.pth"
train_epoch = 80
resume_training = False
z_size = 512

def load_model_from_checkpoint():
    global ae_model
    checkpoint = torch.load(save_path)
    vae.load_state_dict(checkpoint['ae_model'])
    return checkpoint['epoch']



# load dataSet
train_loader = get_data_loader(img_size=im_size,img_path=img_path,shuffle = True,batch_size=batch_size,num_workers=8)

loss_function = pytorch_ssim.SSIM()



def main():
    #build Neural Network
    ae_model = ResNet_autoencoder(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 3)
    ae_model.apply(weights_init_kaiming)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        ae_model.cuda()
    ae_model.train()
    start_epoch = 0
    if resume_training:
        start_epoch = load_model_from_checkpoint()
        
    lr = 0.0005
    ae_model_optimizer = optim.Adam(ae_model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5) 
    device_ids = range(torch.cuda.device_count())
    ae_model = nn.DataParallel(ae_model, device_ids)
    
    for epoch in range(start_epoch,train_epoch):
        ae_model.train()
        ssim_loss = 0
        epoch_start_time = time.time()        
        if (epoch + 1) % 20 == 0:
            ae_model_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")
        i = 0
        for x, _ in train_loader:
            ae_model.train()
            ae_model.zero_grad()
            x = get_cuda(x)
            rec = ae_model(x)
            loss =1 - loss_function(x, rec[1])
            loss.backward()
            ae_model_optimizer.step()
            ssim_loss += loss.item()
            os.makedirs('results_rec', exist_ok=True)
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                ssim_loss /= m
                print('\n[%d/%d] - ptime: %.2f, ssim_loss: %.9f' % ((epoch + 1), train_epoch, per_epoch_ptime, ssim_loss))
                ssim_loss = 0
                with torch.no_grad():
                    ae_model.eval()
                    x_rec = ae_model(x)
                    resultsample = torch.cat([x, x_rec[1]])
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
        torch.save({
            'epoch': epoch,
            "ae_model": model.state_dict(),
            }, save_path)
          
    print("Training finish!... save training results")
    torch.save(model.state_dict(), "ae_model.pkl")

if __name__ == '__main__':
    main()