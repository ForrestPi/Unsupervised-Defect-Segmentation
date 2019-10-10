# -*- coding: utf-8 -*-
"""
@author: WellenWoo
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
import time

class Clf_Reporter(object):
    """用于分类模型的预测结果评估器;"""    
    def clf_report(self, y_test, y_pred):
        """分类报告;
        只能用于分类任务,
        不能用于回归任务;"""
        score = classification_report(y_test, y_pred)
        return score
    
    def matrix(self, y_test, y_pred):
        """混淆矩阵;
        用于分类任务;"""
        score = confusion_matrix(y_test, y_pred)
        return score

    def roc(self, labels, scores):
        """Compute ROC curve and ROC area for each class"""
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        labels = labels.cpu()
        scores = scores.cpu()

        # True/False Positive Rates.\
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc
            
    def plot_cfm(self, cfm, classes = ["ng", "ok"],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        绘制混淆矩阵图;
        args:
            cfm: 混淆矩阵, np.ndarray,
            classes: 类别列表, list, i.e. ["ng", "ok"],
            normalize: 是否归一化,bool,归一化则以百分比显示;
        """
        if normalize:
            cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cfm)

        plt.imshow(cfm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cfm.max() / 2.
        for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
            plt.text(j, i, format(cfm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cfm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    def get_threshold(self, y_test, pred, 
                            threshold_rang = [0.05,0.9], step = 0.01, 
                            plot = True):
        """获取最佳置信度,目前仅适用于二分类.
        
        args:
        ----------
            y_test: 真实的分类标签, np.ndarray,
            
            pred: 预测的便签, np.ndarray,
            
            threshold_rang: 置信度范围, list,
            
            step: 步长, float;
        retrun:
        ----------
            val: 最佳置信度对应的正确分类样本数,
            
            data[val]: 最佳置信度."""
        pred_true_0 = []
        pred_true_1 = []
        pred_true_sum = []
        
        threshold_rang = np.arange(*threshold_rang, step)
        data = dict()
        
        for th in threshold_rang:
            tmp = self.prob2int(pred, th)
            mat = self.matrix(y_test, tmp)
            
            pred_true_0.append(mat[0][0])
            pred_true_1.append(mat[1][1])
            
            sums = np.sum([mat[0][0], mat[1][1]])
            pred_true_sum.append(sums)
            
            data[str(th)] = sums
        if plot:
            plot_true_pred(threshold_rang, pred_true_0, pred_true_1, pred_true_sum)
            
        th, val = get_dict_max_val(data)
        return th, val, data

def plot_true_pred(rang, true0, true1, true_sum):
    fig,ax = plt.subplots()
    ax.plot(rang, true0, label = "true0")
    ax.plot(rang, true1, label = "true1")
    ax.plot(rang, true_sum, label = "true_sum")
        
    ax.set_xlabel("confidence")
    ax.set_ylabel("samples")
        
    plt.grid()
    plt.legend()
    plt.show() 
    
def get_dict_max_val(d):
    """获取字典中的最大值及其对应的键值;
    args:
        ----
        d: 字典, dict,
    
    return:
        ---
        val: 最大值所对应的键, str,
        
        max_val: 最大值, int or float;"""
    max_val = max(d.values())
    for i in d.keys():
        if d[i] == max_val:
            break
    return i, max_val

def print_current_performance(performance, best):
    """ Print current performance results.
    Args:
        performance ([OrderedDict]): Performance of the model
        best ([int]): Best performance.
    """
    message = '   '
    log_name = r"train_log.txt"
    dt = time.gmtime()
    message += str(dt.tm_year) +"-"+str(dt.tm_mon)+"-"+str(dt.tm_mday)+"-"+str(dt.tm_hour+8)+":"+str(dt.tm_min)+":"+str(dt.tm_sec)
    message += "roc: %.3f#######" % performance
    message += 'max AUC: %.3f' % best

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)
