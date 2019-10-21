# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
# from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator, Discriminator_WGANGP
# from Unet3d_pytorch import UNet3D
from nnBuildUnits import  adjust_learning_rate, calc_gradient_penalty
from ganComponents import Generator, Generator_Z, Discriminator
import time
import SimpleITK as sitk


# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID", type=int, default=5, help="how to normalize the data")
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)
parser.add_argument("--isWDist", action="store_true", help="is adversarial loss with WGAN-GP distance?", default=True)
parser.add_argument("--lambda_AD", default=1, type=float, help="weight for AD loss, Default: 0.05")
parser.add_argument("--lambda_D_WGAN_GP", default=10, type=float, help="weight for gradient penalty of WGAN-GP, Default: 10")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters")
parser.add_argument("--outputSizeOfG", type=int, default=32, help="the output size of G network: x*x*x")
parser.add_argument("--finalSize", type=int, default=64, help="the size of generated data")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showValPerformanceEvery", type=int, default=1000, help="number of iterations to show validation performance")
parser.add_argument("--showTestPerformanceEvery", type=int, default=5000, help="number of iterations to show test performance")
parser.add_argument("--lr_G", type=float, default=1e-3, help="Learning Rate for G. Default=1e-4")
parser.add_argument("--lr_D", type=float, default=5e-3, help="Learning Rate for D. Default=1e-4")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="prob to drop neurons to zero: 0.2")
parser.add_argument("--decLREvery", type=int, default=10000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/shenlab/lab_stor5/dongnie/anoDetectGAN/model/anoDetect_WDistGP_lr5e3_1106_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub10_", type=str, help="prefix of the to-be-saved predicted filename")

parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--isGDL", action="store_true", help="do we use GDL loss?", default=True)
parser.add_argument("--gdlNorm", default=2, type=int, help="p-norm for the gdl loss, Default: 2")
parser.add_argument("--lambda_gdl", default=0.05, type=float, help="Weight for gdl loss, Default: 0.05")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--isMultiSource", action="store_true", help="is multiple modality used?", default=False)
parser.add_argument("--numOfChannel_singleSource", type=int, default=5, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=5, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--RT_th", default=0.005, type=float, help="Relative thresholding: 0.005")

global opt, model 
opt = parser.parse_args()

def main():    
    print opt    

    # load/define networks
    netG = Generator(opt.finalSize*opt.finalSize, opt.ngf, gpu_ids=opt.gpuID)
#     netG = define_G(opt.fineSize*opt.fineSize, opt.ngf,opt.init_type, opt.gpuID)
    optimizerG = optim.Adam(netG.parameters(),lr=opt.lr_G)                            
    netG.apply(weights_init)
    netG.cuda()
    
    if opt.isAdLoss:
        netD = Discriminator(opt.ndf, opt.gpuID)
        netD.apply(weights_init)
        netD.cuda()
        optimizerD = optim.Adam(netD.parameters(),lr=opt.lr_D)
        
    if opt.isWDist:
        netD = Discriminator(opt.ndf, opt.gpuID)
        netD.apply(weights_init)
        netD.cuda()
        optimizerD = optim.Adam(netD.parameters(),lr=opt.lr_D)
        
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    params = list(netG.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())

    
    
    path_test ='/shenlab/lab_stor5/dongnie/3T7T/3t7tHistData/'
    path_patients_h5 = '/shenlab/lab_stor5/dongnie/3T7T/histH5Data'
    path_patients_h5_test ='/shenlab/lab_stor5/dongnie/3T7T/histH5Data'


    data_generator = Generator_2D_slices_oneKey(path_patients_h5,opt.batchSize, inputKey='data7T')
    data_generator_test = Generator_2D_slices_oneKey(path_patients_h5_test,opt.batchSize, inputKey='data7T')
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['model'])
            opt.start_epoch = 100000
            opt.start_epoch = checkpoint["epoch"] + 1
            # net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        #print('iter %d'%iter)
        
        # we use a 128-dim vector as the noise of the input
        noise = torch.randn(opt.batchSize, 128)
        if opt.gpuID!=None:
            noise = noise.cuda()

        labels = data_generator.next() # labels means real images
        labels = np.squeeze(labels) #64x64
        labels = np.resize(labels, [opt.batchSize, opt.outputSizeOfG,opt.outputSizeOfG]) # here, we take 32 as output size
#         labels = labels.astype(int)


        inputs = noise

        labels = labels.astype(float)
        labels = torch.from_numpy(labels)
        labels = labels.float()

        source = inputs

        source = source.cuda()
#         residual_source = residual_source.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        source, labels = Variable(source), Variable(labels)

        
        ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if opt.isAdLoss:
#             outputG = net(source,residual_source) #5x64x64->1*64x64

            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
            # print 'labels.shape: ',labels.shape
            outputD_real = netD(labels)
            outputD_real = F.sigmoid(outputD_real)

            outputG = netG(source) #1x64x64->1*64x64
            # print 'outputG.shape: ',outputG.shape
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            outputD_fake = netD(outputG)
            outputD_fake = F.sigmoid(outputD_fake)

            ## update D network
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            #print(real_label.size())
            real_label = Variable(real_label)
            # print 'outputD_real.shape: ',outputD_real.shape,' outputD_fake.size(): ',outputD_fake.shape
            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)

            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake

            optimizerD.step()
            
        if opt.isWDist:
            one = torch.FloatTensor([1])
            mone = one * -1
            one = one.cuda()
            mone = mone.cuda()
            
            netD.zero_grad()
            
            outputG = netG(source) #5x64x64->1*64x64
            
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)

            
            batch_size = inputs.size(0)
            
            D_real = outputD_real.mean()
            # print D_real
            D_real.backward(mone)
        
        
            D_fake = outputD_fake.mean()
            D_fake.backward(one)
        
            gradient_penalty = opt.lambda_D_WGAN_GP*calc_gradient_penalty(netD, labels.data, outputG.data)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            
            optimizerD.step()
        
        
        ## (2) update G network: maximize the D(G(x))
        netG.zero_grad()
        
        if opt.isAdLoss:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            outputG = netG(source) #5x64x64->1*64x64
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            outputD = F.sigmoid(outputD)
            lossG_D = opt.lambda_AD*criterion_bce(outputD,real_label) #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
            
        if opt.isWDist:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            #outputG = net(inputs)
            outputG = netG(source) #5x64x64->1*64x64
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD_fake = netD(outputG)

            outputD_fake = outputD_fake.mean()
            
            lossG_D = opt.lambda_AD*outputD_fake.mean() #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward(mone)
        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizerG.step() #update network parameters

        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
#             print 'running loss is ',running_loss
#             print 'average running loss for generator between iter [%d, %d] is: %.5f'%(iter - 100 + 1,iter,running_loss/100)

            if opt.isAdLoss:
                print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',torch.mean(outputD_real).data[0], 'outputD_fake is',torch.mean(outputD_fake).data[0]
                print 'lossG_D is ', lossG_D.data[0]
                print 'loss for discriminator is %f'%lossD.data[0]
                
            if opt.isWDist:
                print 'loss_real is ',D_real.data[0],'loss_fake is ',D_fake.data[0], 'outputD_real is',torch.mean(outputD_real).data[0], 'outputD_fake is',torch.mean(outputD_fake).data[0]
                print 'lossG_D is ', lossG_D.data[0]
                print 'loss for discriminator is %f'%Wasserstein_D.data[0], ' D cost is %f'%D_cost
            
            print 'cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start)
            print '************************************************'
            running_loss = 0.0
            start = time.time()
        if iter%opt.saveModelEvery==0: #save the model
            state = {
                'epoch': iter+1,
                'model': netG.state_dict()
            }
            torch.save(state, opt.prefixModelName+'%d.pt'%iter)
            print 'save model: '+opt.prefixModelName+'%d.pt'%iter

            if opt.isAdLoss or opt.isWDist:
                torch.save(netD.state_dict(), opt.prefixModelName+'netD_%d.pt'%iter)
        if iter%opt.decLREvery==0:
            opt.lr_G = opt.lr_G*0.5
            adjust_learning_rate(optimizerG, opt.lr_G)
            if opt.isAdLoss or opt.isWDist:
                opt.lr_D = opt.lr_D*0.2
                adjust_learning_rate(optimizerD, opt.lr_D)
                
        if iter%opt.showValPerformanceEvery==0: #test one subject
            # to test on the validation dataset in the format of h5
            
            # we use a 128-dim vector as the noise of the input
            noise = torch.randn(opt.batchSize, 128)
            if opt.gpuID!=None:
                noise = noise.cuda()
            inputs = noise
             
#             inputs,exinputs,labels = data_generator_test.next()
            labels = data_generator_test.next()
            labels = np.squeeze(labels)

            labels = torch.from_numpy(labels)
            labels = labels.float()

            source = inputs
            source = source.cuda()
            source = Variable(source)
#             residual_source = residual_source.cuda()
            labels = labels.cuda()
            labels = Variable(labels)
            
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
            outputD_real = netD(labels)
            
            outputG = netG(source) #noise -> img
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            outputD_fake = netD(outputG)
            print 'outputD_real is ', torch.mean(outputD_real).data[0], ' outputD_real is ', torch.mean(outputD_real).data[0]


        if iter % opt.showTestPerformanceEvery == 0:  # test one subject
            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'S10to1_3t.nii.gz'))


            spacing = mr_test_itk.GetSpacing()
            origin = mr_test_itk.GetOrigin()
            direction = mr_test_itk.GetDirection()

            mrnp=sitk.GetArrayFromImage(mr_test_itk)

            ##### specific normalization #####
            mu = np.mean(mrnp)

            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(mrnp)

            #for training data in pelvicSeg
            if opt.how2normalize == 2:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(mrnp)

            #for training data in pelvicSegRegH5
            if opt.how2normalize== 3:
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)

            if opt.how2normalize == 4:
                maxLPET = 149.366742
                maxPercentLPET = 7.76
                minLPET = 0.00055037
                meanLPET = 0.27593288
                stdLPET = 0.75747500

                matLPET = (mrnp - minLPET) / (maxPercentLPET - minLPET)

            if opt.how2normalize == 5:
                # for rsCT
                maxCT = 27279
                maxPercentCT = 1320
                minCT = -1023
                meanCT = -601.1929
                stdCT = 475.034

                print 'ct, max: ', np.amax(mrnp), ' ct, min: ', np.amin(mrnp)

                matLPET = mrnp


            if opt.how2normalize == 6:
                maxPercentPET, minPercentPET = np.percentile(mrnp, [99.5, 0])
                matLPET = (mrnp - minPercentPET) / (maxPercentPET - minPercentPET)
 
            matFA = matLPET
#                 matGT = hpetnp

            print 'matFA shape: ',matFA.shape
            pred = testOneSubject4Cla(matFA,[1,64,64],[1,32,32],netD, opt.prefixModelName+'%d.pt'%iter)
            print 'predicted result is ', pred

        
    print('Finished Training')
    
if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)  
    main()
    