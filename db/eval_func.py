"""Evaluation function

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cal_iou(mask, gt):
    mask_defect = (mask == 255)
    gt_defect = (gt == 255)
    overlap = (mask_defect == gt_defect) & (mask == 255) & (gt == 255)
    mask_defect_sum = float(mask_defect.sum())
    gt_defect_sum = float(gt_defect.sum())
    overlap_sum = float(overlap.sum())
    iou = overlap_sum / (mask_defect_sum + gt_defect_sum - overlap_sum)

    return iou


def cal_pixel_accuracy(mask, gt):
    _h, _w = gt.shape
    positive = (mask == gt)
    positive_sum = float(positive.sum())
    pixel_acc = positive_sum / (_h * _w)

    return pixel_acc


def cal_TPR(s_map,gt,threshold):
    # True Postive Rate
    gt_defect = (gt == 255)
    overlap = (s_map<threshold) & (gt == 255)
    gt_defect_sum = float(gt_defect.sum())
    overlap_sum = float(overlap.sum())
    TPR= overlap_sum /gt_defect_sum
    return TPR


def cal_FPR(s_map,gt,threshold):
    # False Postive Rate
    mask_defect = (s_map<threshold)
    gt_good = (gt == 0)
    overlap = (s_map<threshold) & (gt == 255)
    mask_defect_sum = float(mask_defect.sum())
    gt_good_sum = float(gt_good.sum())
    overlap_sum = float(overlap.sum())
    FPR = (mask_defect_sum-overlap_sum)/gt_good_sum
    return FPR


def cal_AUC(TPR_arr, FPR_arr):
    # compute AUC
    TPR_arr = np.array(TPR_arr)
    FPR_arr = np.array(FPR_arr)
    AUC = 0
    for i in range(TPR_arr.size - 1):
        AUC += np.abs(FPR_arr[i+1] - FPR_arr[i]) * (TPR_arr[i+1] + TPR_arr[i]) / 2
    return AUC


def cal_good_index(mask,area_threshold=100):
    #compute the accuracy for good samples of each class,return 0 for bad samples, return 1 for good samples
    mask_defect = (mask == 255)
    if float(mask_defect.sum()) < area_threshold:
        good_index = 1
    else:
        good_index = 0
    return good_index


if __name__ == '__main__':
    mask = cv2.imread('./mask.png')
    gt = cv2.imread('./gt.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    iou = cal_iou(mask, gt)
    pixel_acc = cal_pixel_accuracy(mask, gt)
    good_index=cal_good_index(mask,area_threshold=10)
