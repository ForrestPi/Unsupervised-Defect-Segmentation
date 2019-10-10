# -*- coding: utf-8 -*-
"""
@author: WellenWoo
"""
import torchvision as tv
import torch
import numpy as np

def files2sample(fpath,img_size = 26,batch_size = 64, workers = 4, drop = False):
    """将文件夹中不同子文件夹的图片转为数据集;
    传入的参数为r'sample';
    子文件夹分别为r'sample/ok_sam',r'sample/ng_sam';
    args:
        fpath: str,图片存放的路径,
        img_size: seq or int, 图片缩放的尺寸,
        batch_size: int,每次批图片的数量,
        workers: int, 进程数量,
    return:
        torch.utils.data.dataloader.DataLoader;
    """
    trans = tv.transforms.Compose([
            tv.transforms.Resize(size = img_size),# 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
#            tv.transforms.CenterCrop(img_size),# 从图片中间切出img_size*img_size的图片
            tv.transforms.ToTensor(),# 将图片(Image)转成Tensor，归一化至[0, 1]
            tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])# 标准化至[-1, 1]，规定均值和标准差
    
    dataset = tv.datasets.ImageFolder(fpath, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers= workers,
                                             drop_last = drop) #drap_last = True:如果样本总量除于batch_size后有余数,则丢弃余数部分
    return dataloader

def loader2tensor(loader):
    """从dataloader中提取图像数据和便签数据,
    再组装成torch.Tensor返回.
    args:
        ----
        loader: 已经包含标签的数据,torch.utils.data.dataloader.DataLoader,
        
    return:
        ----
        x: 图像数据,torch.Tensor,[batch,channel,h, w],
        y: 便签数据,torch.Tensor,[batch]."""
    xs, ys = [], []
    for i, data in enumerate(loader):
        xs.append(data[0])
        ys.append(data[1])
    x = torch.cat(tuple(xs))
    y = torch.cat(tuple(ys))
    return x, y

def data2np(data):
    """将cuda或cpu中的Tensor 数据转为cpu中的np数据;
    args:
        ----
        data: torch.Tensor,
            
    return:
        ----
        data: np.nadrray."""
    device = torch.device("cpu")
    if isinstance(data, torch.Tensor):
        if not data.device == device:
            data = data.cpu()
        data = data.numpy()
    return data
    
def prob2int(y_pred, threshold):
    """将预测的概率值转为类别;
    args:
        y_pred: 预测标签的概率值, np.ndarray,
            
        threshold: 阈值, float,[0, 1],
    return:
        val: 预测的标签, np.ndarray, dtype=np.int."""
    val = np.empty_like(y_pred)
    val[y_pred >= threshold] = 1
    val[y_pred < threshold] = 0
    return val
    
def get_primitive_th(th, pred):
    """根据缩放后的预测阈值来计算出原始预测阈值,
    args:
        ----
        th: 缩放后的预测阈值,float,
        pred: 原始的预测概率序列, torch.Tensor.
        
    return:
        ----
        th: 原始预测阈值,float."""
    return (torch.max(pred) - torch.min(pred)) * th + torch.min(pred)

def get_eval_th(th, pred):
    """根据测试集的原始预测阈值计算验证集的缩放预测阈值,
    args:
        ----
        th: 测试集的原始预测阈值,float,
        pred: 验证集的原始预测概率序列,torch.Tensor,
    
    return:
        ----
        th:验证集的缩放预测阈值,float."""
    return (th - torch.min(pred))/(torch.max(pred) - torch.min(pred))