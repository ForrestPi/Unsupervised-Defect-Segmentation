# -*- coding: utf-8 -*-
"""
@author: WellenWoo
"""
import torch
#from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from nets import NetD, NetG, weights_init
from loss import l2_loss
from reporter import print_current_performance, Clf_Reporter
from utils import data2np, prob2int,files2sample,loader2tensor,get_primitive_th,get_eval_th
import torchsnooper

rpt = Clf_Reporter()

class Ganomaly(object):
    @staticmethod
    def name():
        return "Ganomaly"

    def __init__(self, batchsize = 64, isize = 32, real_label = 1, fake_label = 0):
        super(Ganomaly, self).__init__()
        
        self.batchsize = batchsize      
        self.isize = isize #input image size    

        # -- Discriminator attributes.
        self.fake = None #
        self.latent_i = None
        self.latent_o = None

        # -- Generator attributes.
        
        self.w_bce = 1 #alpha to weight bce loss
        self.w_rec = 50 #alpha to weight reconstruction loss
        self.w_enc = 1 #alpha to weight encoder loss

        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = l2_loss
                
        self.real_label = real_label
        self.fake_label = fake_label
        
    def set_device(self, device = None):
        """设置计算设备;"""
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            self.device = "cpu"
        elif use_cuda and device==None:
            self.device = "cuda:0"
        else:
            self.device = device
        return self.device
    
    def set_nets(self, nc = 3, nz = 100, ngf = 64, ndf = 64, ngpu = 1, extralayers = 0):
        """设置网络架构的基本参数,构建生成器和判别器网络,
        args:
            ----
            nc: 图像通道数,int, 
            nz: 下潜空间向量的长度,int,
            ngpu: 使用的GPU个数,int,
            extralayers: 额外网络层数,int;"""
        self.nc = nc #input images channels
        self.nz = nz#size of the latent z vector 下潜空间向量的长度
        self.ngf = ngf #生成器使用的特征图深度
        self.ndf = ndf #设置判别器使用的特征图的深度
        
        self.ngpu = ngpu
        self.extralayers = extralayers
        
        self.netg = NetG(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers).to(self.device)
        self.netd = NetD(self.isize, self.nc, self.ngf, self.ngpu, self.extralayers).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
    
    def update_nets(self, netd_path, netg_path):
        """更新网络,加载已经训练好的本地模型state_dict,
        args:
            ----
            netd_path: 判别器网络的路径, str,
            netg_path: 生成器网络的路径, str,"""
        self.netd = self.load_weight(self.netd, netd_path).to(self.device)
        self.netg = self.load_weight(self.netg, netg_path).to(self.device)

    def put_data2device(self, datas):
        """将数据加载到模型所在的设备上,
        args:
            ----
            datas:存放输入数据的可迭代容器, list or tuple,
        
        return:
            ----
            datas:存放输出数据的可迭代容器, list or tuple."""
        for i, data in enumerate(datas):
            datas[i] = data.to(self.device)
        return datas     

    def optimize(self, x_train):
        """优化损失,
        args:
            ----
            x_train: 图像数据,torch.Tensor, [batch, channel, h, w]."""
        latent_i, latent_o, err_d_real, err_d_fake = self.update_netd(x_train)
        self.update_netg(x_train, latent_i, latent_o)
        
        if err_d_real.item() < 1e-5 or err_d_fake.item() < 1e-5:
            self.reinitialize_netd()

    def update_netd(self, x_train):
        """更新判别器网络,
        Update D network: Ladv = |f(real) - f(fake)|_2
        args:
            ----
            x_train: 训练的样本,torch.Tensor,[batch,channel,h,w],
        return:
            ----
            latent_i:从生成网络编码器中出来的向量,torch.Tensor,
            latent_o:从生成网络重构编码器中出来的向量,torch.Tensor,
            err_d_real: Ladv,
            err_d_fake: Ladv.
        """
        current_batchsize = x_train.shape[0]
        label = torch.empty(size = (current_batchsize, ), dtype = torch.float32, device = self.device)
        self.netd.zero_grad()
        
        #train with real
        label.data.resize_(current_batchsize).fill_(self.real_label) #用1填满便签张量
        _, feat_real = self.netd(x_train)
        
        #train with fake
        label.data.resize_(current_batchsize).fill_(self.fake_label) #用0填满便签张量
        self.fake, latent_i, latent_o = self.netg(x_train)
        _, feat_fake = self.netd(self.fake.detach())
        
        err_d = l2_loss(feat_real, feat_fake)
        err_d_real = err_d
        err_d_fake = err_d
        err_d.backward()
        self.optimizer_d.step() 

        return latent_i, latent_o, err_d_real, err_d_fake
    
    def update_netg(self, x_train, latent_i, latent_o):
        """更新生成器网络,
        # ============================================================ #
        # (2) Update G network: log(D(G(x)))  + ||G(x) - x||           #
        # ============================================================ #
        args:
            ----
            x_train:训练的样本,torch.Tensor,[batch,channel,h,w],
            latent_i:从生成网络编码器中出来的向量,torch.Tensor,
            latent_o:从生成网络重构编码器中出来的向量,torch.Tensor,            
        """
        current_batchsize = x_train.shape[0]
        label = torch.empty(size = (current_batchsize, ), dtype = torch.float32, device = self.device)
        
        self.netg.zero_grad()
        
        label.data.resize_(current_batchsize).fill_(self.real_label)
        out_g, _ = self.netd(self.fake)
        
        err_g_bce = self.bce_criterion(out_g, label)
        err_g_l1l = self.l1l_criterion(self.fake, x_train)
        err_g_enc = self.l2l_criterion(latent_o, latent_i)
        err_g = err_g_bce * self.w_bce + err_g_l1l * self.w_rec + err_g_enc * self.w_enc
        
        err_g.backward(retain_graph = True)
        self.optimizer_g.step()
    
    def reinitialize_netd(self):
        """重初始化判别器网络的权重参数;"""
        self.netd.apply(weights_init)
    
    def save_weight(self, netd, netg, netd_name = "netD.pth", netg_name = "netG.pth"):
        """保存模型权重参数;
        args:
            ----
            netd: 已经训练好的判别器网络模型,nn.Module,
            netg: 已经训练好的生成器网络模型,nn.Module,
            netd_name: 保存判别器的路径及名称,str,
            netg_name: 保存生成器的路径及名称,str;"""
            
        torch.save(netd.state_dict(), netd_name)
        torch.save(netg.state_dict(), netg_name)        
        
    def load_weight(self, model_frame, model_path):
        """加载预训练模型;
        args:
            model_frame: 网络结构, nn.Module,
            model_path: 预训练模型的路径, str;
        return:
            model_frame: 已读入权重参数的预训练模型;"""
        pretrained_dict = torch.load(model_path)
        model_frame.load_state_dict(pretrained_dict)
        return model_frame
    
class Ganomaly4loader(Ganomaly):
    """针对dataloader类型输入数据的模型;
    usage:
        ----        
        #读入样本数据
        x1 = files2sample(r"train", (32, 32), drop = False)
        x2 = files2sample(r"test", (32, 32), drop = False)
        x3 = files2sample(r"eval", (32, 32), drop = False)
                
        #训练模型
        clf = Ganomaly4loader()
        clf.set_device("cuda:0")
        clf.set_nets()
        clf.fit(x1, x2)
        
        #获取适当的预测阈值
        y_test,p_test,ps_test = clf.predict_classes(x2)
        y_test_ = data2np(y_test)
        
        ps_test_ = data2np(ps_test)
        th_test, v_test, d_test = rpt.get_threshold(y_test_, ps_test_, plot = False)
        ps_test_ = prob2int(ps_test_, float(th_test))
        
        #打印此阈值对应的混淆矩阵及准确率
        print(rpt.matrix(y_test_, ps_test_))
        print(rpt.clf_report(y_test_, ps_test_))
        
        #获取原始预测值的阈值
        prm_th_test = get_primitive_th(float(th_test), p_test)
        prm_th_test_ = data2np(prm_th_test)
        
        y_eval, p_eval, ps_eval = clf.predict_classes(x3)
        y_eval_ = data2np(y_eval)
        ps_eval_ = data2np(ps_eval)
        th_eval, v_eval, d_eval = rpt.get_threshold(y_eval_, ps_eval_, plot = False)
        
        th_eval2 = get_eval_th(float(th_test), p_eval)
        
        #保存模型权重参数
        clf.save_weight(clf.netd, clf.netg, 
                        r"gan4loader_mnist_0_1_netd.pth",
                        r"gan4loader_mnist_0_1_netg.pth")
        #加载本地模型
        gan = Ganomaly4loader()
        gan.set_device("cuda:0")
        gan.set_nets()
        gan.update_nets(r"gan4loader_mnist_0_1_netd.pth",
                        r"gan4loader_mnist_0_1_netg.pth")        
        """
    
    def unpack_loader(self, loader):
        """从输入数据中提取出图像数据及其便签;
        args:
            loader: 含两个元素的list,两个元素均为Tensor,
                    size分别为[batch, n_channel, h, w], [batch],
        return:
            data: 图像数据, Tensor, size=[batch, n_channel, h, w],
            label: 便签, Tensor, size = [batch];"""
        data = loader[0].to(self.device)
        label = loader[1].to(self.device)
        return data, label
    
    def fit(self, train_loader, test_loader, lr = 0.0002, niter = 15, beta = 0.5):
        """训练模型,dataloader类型输入数据的接口,
        args:
            ----
            train_loader: 已经包含标签的训练样本,torch.utils.data.dataloader.DataLoader,
            test_loader: 已经包含标签的测试样本,torch.utils.data.dataloader.DataLoader,
            lr:学习率,float,
            niter:迭代次数,int,
            beta:float."""
        self.netg.train()
        self.netd.train()
        
        #设置优化器
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr = lr, betas = (beta, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr = lr, betas = (beta, 0.999))
        
        best_auc = 0
        
        for epoch in range(0, niter):
            self.train_epoch(train_loader, epoch, niter)
            labels, _, scaled_pred = self.predict_classes(test_loader)
            auc = rpt.roc(labels, scaled_pred)
            print_current_performance(auc, best_auc)
            print(auc)
            if auc > best_auc:
                best_auc = auc
                self.save_weight(self.netg, self.netd, 
                                 "epoch{n}netg.pth".format(n = epoch), 
                                 "epoch{n}netd.pth".format(n = epoch))
            torch.cuda.empty_cache()
        print(">> Training model Gan.[Done]" ) 
    
    def train_epoch(self, train_loader, epoch, niter):
        self.netg.train()
        
        for i, loader in enumerate(train_loader, 0):
            data, label = self.unpack_loader(loader)
            self.optimize(data)
        
        print(">> Training model. Epoch %d/%d" % (epoch+1, niter))
            
#    @torchsnooper.snoop()
    def predict_classes(self, test_loader):
        """预测样本;这种预测数据的方式对test集的文件夹存放顺序有严格的要求,
        正样本(OK样本,即与train 集相同的样本)必须为test集的第一个文件夹,
        否则得到的标签会有误;另外,当此函数用于预测"全负样本"的数据集时(即
        test文件夹中只有一个文件夹且该文件夹中的全部图像都为负样本),
        需要手动构建标签集,y_test = np.repeat(1, len(x_test)),
        因为此时本函数所返回的是一个错误的标签集,里面的标签全部为0;
        args:
        ----------
            test_loader: 已经包含标签的测试样本,torch.utils.data.dataloader.DataLoader;
            
        return:
            ----
            labels:测试样本真实的标签,torch.Tensor,
            scaled_prd:经过缩放后的预测值,torch.Tensor,
            pred: 原始预测值,torch.Tensor.
        """
#        self.netg.eval()
        with torch.no_grad():
            batch_size = test_loader.batch_size
            nsample = len(test_loader.dataset.imgs)
            if test_loader.drop_last:
                nsample = nsample - nsample % batch_size
                
            pred = torch.zeros(size = (nsample, ), dtype = torch.float32, device = self.device)
            labels = torch.zeros(size = (nsample, ) , dtype = torch.long, device = self.device)
            
            for i, loader in enumerate(test_loader, 0):
                x_data, y_data = self.unpack_loader(loader)
                _, latent_i, latent_o = self.netg(x_data)
                
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim = 1)
                
                pred[i * self.batchsize: i * self.batchsize + error.size(0)] = error.reshape(error.size(0))
                labels[i * self.batchsize: i * self.batchsize + error.size(0)] = y_data.reshape(error.size(0))
                
            scaled_pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred)) #将置信度缩放到0-1之间;  
            return labels, pred, scaled_pred
