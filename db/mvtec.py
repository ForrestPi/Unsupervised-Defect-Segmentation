"""Data set tool of MVTEC

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import os
import re
import torch
import numpy as np
import torch.utils.data as data
from collections import OrderedDict
from .augment import *
from .eval_func import *


class Preproc(object):
    """Pre-procession of input image includes resize, crop & data augmentation

    Arguments:
        resize: tup(int width, int height): resize shape
        crop: tup(int width, int height): crop shape
    """
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        # random transformation
        p = random.uniform(0, 1)
        if (p > 0.33) and (p <= 0.66):
            image = mirror(image)
        else:
            image = flip(image)
        # light adjustment
        p = random.uniform(0, 1)
        if p > 0.5:
            image = lighting_adjust(image, k=(0.95, 1.05), b=(-10, 10))

        # image normal
        image = image.astype(np.float32) / 255.
        # normalize_(tile, self.mean, self.std)
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


class MVTEC_with_val(data.Dataset):
    """A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection with validation set

    Arguments:
        root (string): root directory to mvtec folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.set = set

        if set == 'train':
            self.ids = list()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                img_dir = os.path.join(item_path, set, 'good')
                imag_count = 0
                for img in os.listdir(img_dir):
                    imag_count += 1
                    if 1 <= imag_count <= int(0.9 * len(os.listdir(img_dir))):  # 90% data in training set for training
                        self.ids.append(os.path.join(img_dir, img))
                    elif imag_count > int(0.9 * len(os.listdir(img_dir))):
                        pass
                    else:
                        raise Exception("Invalid image number")
        elif set == 'validation':
            self.val_dict = OrderedDict()
            for _item in os.listdir(root):
                self.ids_val = list()
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                img_dir = os.path.join(item_path, 'train', 'good')
                imag_count = 0
                for img in os.listdir(img_dir):
                    imag_count += 1
                    if 1 <= imag_count <= int(0.9 * len(os.listdir(img_dir))):
                        pass
                    elif imag_count > int(0.9 * len(os.listdir(img_dir))):  # 10% data in training set for validation
                        self.ids_val.append(os.path.join(img_dir, img))
                    else:
                        raise Exception("Invalid image number")
                self.val_dict[_item] = self.ids_val
        elif set == 'test':
            self.test_len = 0
            self.test_dict = OrderedDict()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                self.test_dict[_item] = OrderedDict()
                type_dir = os.path.join(item_path, set)
                for type in os.listdir(type_dir):
                    img_dir = os.path.join(item_path, set, type)
                    ids = list()
                    for img in os.listdir(img_dir):
                        if re.search('.png', img) is None:
                            continue
                        ids.append(os.path.join(img_dir, img))
                        self.test_len += 1
                    self.test_dict[_item][type] = ids
        else:
            raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.preproc is not None:
            img = self.preproc(img)

        return img

    def __len__(self):
        if self.set == 'train':
            return len(self.ids)
        else:
            return self.test_len

    def eval(self, eval_dir,threshold_dict):
        summary_file = open(os.path.join(eval_dir, 'summary.txt'), 'w')
        for item in self.test_dict:
            summary_file.write('--------------{}--------------\n'.format(item))
            labels = list()
            paccs = list()
            ious = list()
            type_good_index = 0
            type_bad_index = 0
            num_good = 0
            num_bad = 0

            FPR_list = list()
            TPR_list = list()
            gt_re_list = list()
            gt_dir = os.path.join(self.root, item, 'ground_truth')
            res_dir = os.path.join(eval_dir, item, 'mask')
            log_file = open(os.path.join(eval_dir, item, 'result.txt'), 'w')
            log_file.write('Item: {}\n'.format(item))

            for type in os.listdir(res_dir):
                log_file.write('--------------------------\nType: {}\n'.format(type))
                type_dir = os.path.join(res_dir, type)
                type_ious = list()
                type_paccs = list()
                for mask in os.listdir(type_dir):
                    mask_id = mask.split('.')[0]
                    gt_id = '{}_mask'.format(mask_id)
                    if type != 'good':
                        gt = cv2.imread(os.path.join(gt_dir, type, '{}.png'.format(gt_id)))
                        mask = cv2.imread(os.path.join(type_dir, mask))
                        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        _h, _w = gt.shape
                        mask = cv2.resize(mask, (_w, _h))
                        labels.append(0)

                        type_ious.append(cal_iou(mask, gt))
                        type_bad_index+=(1-cal_good_index(mask,400))
                        gt_re_list.append(gt.reshape(_w*_h, 1))
                        num_bad += 1

                    else:
                        mask = cv2.imread(os.path.join(type_dir, mask))
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        gt = np.zeros(shape=mask.shape, dtype=np.uint8)
                        labels.append(1)
                        num_good += 1
                        type_good_index += (cal_good_index(mask, 400))
                    type_paccs.append(cal_pixel_accuracy(mask, gt))
                if type == 'good':
                    log_file.write('mean IoU: nan\n')
                else:
                    log_file.write('mean IoU:{:.2f}\n'.format(np.array(type_ious).mean() * 100))
                log_file.write('mean Pixel Accuracy:{:2f}\n'.format(np.array(type_paccs).mean() * 100))
                ious += type_ious
                paccs += type_paccs
            mIoU = np.array(ious).mean()
            mPAc = np.array(paccs).mean()
            s_map_all = np.array(threshold_dict[item]).reshape(-1, 1)
            gt_re = np.array(gt_re_list)
            gt_re = gt_re.reshape(-1,1)
            for threshold in np.arange(0,1,0.005):
                FPR_list.append(cal_FPR(s_map_all, gt_re,threshold))
                TPR_list.append(cal_TPR(s_map_all, gt_re,threshold))

            auc = cal_AUC(TPR_list, FPR_list)
            plt.figure()
            plt.plot(FPR_list, TPR_list, '.-')
            plt.savefig('./eval_result/ROC_curve/' + item + '.jpg')
            acc_good = type_good_index / num_good
            acc_bad = type_bad_index / num_bad
            log_file.write('--------------------------\n')
            log_file.write('Total mean IoU:{:.2f}\n'.format(mIoU*100))
            log_file.write('Total mean Pixel Accuracy:{:.2f}\n'.format(mPAc*100))
            log_file.write('AUC of defect samples: {:.2f}\n'.format(auc * 100))
            log_file.write('acc of good samples: {:.2f}\n'.format(acc_good * 100))
            log_file.write('acc of bad samples: {:.2f}\n'.format(acc_bad*100))
            summary_file.write('mIoU:{:.2f}     mPAcc:{:.2f}    auc:{:.2f}  acc_good:{:.2f}  acc_bad:{:.2f}\n'.format(mIoU*100, mPAc*100,auc*100,acc_good*100,acc_bad*100))

            log_file.write('\n')
            log_file.close()
            pass



class MVTEC(data.Dataset):
    """A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

    Arguments:
        root (string): root directory to mvtec folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.set = set

        if set == 'train':
            self.ids = list()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                img_dir = os.path.join(item_path, set, 'good')
                for img in os.listdir(img_dir):
                    self.ids.append(os.path.join(img_dir, img))
        elif set == 'test':
            self.test_len = 0
            self.test_dict = OrderedDict()
            for _item in os.listdir(root):
                item_path = os.path.join(root, _item)
                if os.path.isfile(item_path):
                    continue
                self.test_dict[_item] = OrderedDict()
                type_dir = os.path.join(item_path, set)
                for type in os.listdir(type_dir):
                    img_dir = os.path.join(item_path, set, type)
                    ids = list()
                    for img in os.listdir(img_dir):
                        if re.search('.png', img) is None:
                            continue
                        ids.append(os.path.join(img_dir, img))
                        self.test_len += 1
                    self.test_dict[_item][type] = ids
        else:
            raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.preproc is not None:
            img = self.preproc(img)

        return img

    def __len__(self):
        if self.set == 'train':
            return len(self.ids)
        else:
            return self.test_len

    def eval(self, eval_dir,threshold_dict):
        summary_file = open(os.path.join(eval_dir, 'summary.txt'), 'w')
        for item in self.test_dict:
            summary_file.write('--------------{}--------------\n'.format(item))
            labels = list()
            paccs = list()
            ious = list()
            type_good_index = 0
            type_bad_index = 0
            num_good = 0
            num_bad = 0

            FPR_list = list()
            TPR_list = list()
            gt_re_list = list()
            gt_dir = os.path.join(self.root, item, 'ground_truth')
            res_dir = os.path.join(eval_dir, item, 'mask')
            log_file = open(os.path.join(eval_dir, item, 'result.txt'), 'w')
            log_file.write('Item: {}\n'.format(item))

            for type in os.listdir(res_dir):
                log_file.write('--------------------------\nType: {}\n'.format(type))
                type_dir = os.path.join(res_dir, type)
                type_ious = list()
                type_paccs = list()
                for mask in os.listdir(type_dir):
                    mask_id = mask.split('.')[0]
                    gt_id = '{}_mask'.format(mask_id)
                    if type != 'good':
                        gt = cv2.imread(os.path.join(gt_dir, type, '{}.png'.format(gt_id)))
                        mask = cv2.imread(os.path.join(type_dir, mask))
                        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        _h, _w = gt.shape
                        mask = cv2.resize(mask, (_w, _h))
                        labels.append(0)

                        type_ious.append(cal_iou(mask, gt))
                        type_bad_index+=(1-cal_good_index(mask,800))
                        gt_re_list.append(gt.reshape(_w*_h, 1))
                        num_bad += 1

                    else:
                        mask = cv2.imread(os.path.join(type_dir, mask))
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        gt = np.zeros(shape=mask.shape, dtype=np.uint8)
                        labels.append(1)
                        num_good += 1
                        type_good_index += (cal_good_index(mask, 800))
                    type_paccs.append(cal_pixel_accuracy(mask, gt))
                if type == 'good':
                    log_file.write('mean IoU: nan\n')
                else:
                    log_file.write('mean IoU:{:.2f}\n'.format(np.array(type_ious).mean() * 100))
                log_file.write('mean Pixel Accuracy:{:2f}\n'.format(np.array(type_paccs).mean() * 100))
                ious += type_ious
                paccs += type_paccs
            mIoU = np.array(ious).mean()
            mPAc = np.array(paccs).mean()
            s_map_all = np.array(threshold_dict[item]).reshape(-1, 1)
            gt_re = np.array(gt_re_list)
            gt_re = gt_re.reshape(-1,1)
            for threshold in np.arange(0,1,0.005):
                FPR_list.append(cal_FPR(s_map_all, gt_re,threshold))
                TPR_list.append(cal_TPR(s_map_all, gt_re,threshold))

            auc = cal_AUC(TPR_list, FPR_list)
            plt.figure()
            plt.plot(FPR_list, TPR_list, '.-')
            plt.savefig('./eval_result/ROC_curve/' + item + '.jpg')
            acc_good = type_good_index / num_good
            acc_bad = type_bad_index / num_bad
            log_file.write('--------------------------\n')
            log_file.write('Total mean IoU:{:.2f}\n'.format(mIoU*100))
            log_file.write('Total mean Pixel Accuracy:{:.2f}\n'.format(mPAc*100))
            log_file.write('AUC of defect samples: {:.2f}\n'.format(auc * 100))
            log_file.write('acc of good samples: {:.2f}\n'.format(acc_good * 100))
            log_file.write('acc of bad samples: {:.2f}\n'.format(acc_bad*100))
            summary_file.write('mIoU:{:.2f}     mPAcc:{:.2f}    auc:{:.2f}  acc_good:{:.2f}  acc_bad:{:.2f}\n'.format(mIoU*100, mPAc*100,auc*100,acc_good*100,acc_bad*100))

            log_file.write('\n')
            log_file.close()
            pass

# test
# if __name__ == '__main__':
#     mvtec = MVTEC(root='D:/DataSet/mvtec_anomaly_detection', set='train', preproc=None)
#     for i in range(len(mvtec)):
#         img = mvtec.__getitem__(i)