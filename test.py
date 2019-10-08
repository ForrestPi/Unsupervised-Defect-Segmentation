import os
import cv2
import json
import argparse
import numpy as np
from db import Transform
from model.rebuilder import Rebuilder
from model.segmentation import ssim_seg, seg_mask
from tools import Timer
from factory import *
from db.eval_func import cal_good_index


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection base on anchor.')
    parser.add_argument('--cfg', help="Path of config file", type=str, required=True)
    parser.add_argument('--model_path', help="Path of model", type=str,required=True)
    parser.add_argument('--gpu_id', help="ID of GPU", type=int, default=0)
    parser.add_argument('--res_dir', help="Directory path of result", type=str, default='./eval_result')
    parser.add_argument('--retest', default=False, type=bool)

    return parser.parse_args()

def val_mvtec(val_set, rebuilder, transform):
    threshold_seg_dict = dict()
    for item in val_set.val_dict:
        item_list = list()
        item_list = val_set.val_dict[item]
        good_count = 0
        for threshold_temp in range(0, 256):
            for path in item_list:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                ori_h, ori_w, _ = image.shape
                ori_img, input_tensor = transform(image)
                out = rebuilder.inference(input_tensor)
                re_img = out.transpose((1, 2, 0))
                s_map = ssim_seg(ori_img, re_img)
                s_map = cv2.resize(s_map, (ori_w, ori_h))
                mask = seg_mask(s_map, threshold_temp)
                good_count += cal_good_index(mask, 400)
            if good_count >= int(0.99*len(item_list)):
                threshold_seg_dict[item] = threshold_temp
                break
        print('validation: Item:{} finishes'.format(item))
    return threshold_seg_dict


def test_mvtec(test_set, rebuilder, transform, save_dir, threshold_seg_dict, val_index):
    _t = Timer()
    cost_time = list()
    threshold_dict = dict()
    if not os.path.exists(os.path.join(save_dir, 'ROC_curve')):
        os.mkdir(os.path.join(save_dir, 'ROC_curve'))
    for item in test_set.test_dict:
        threshold_list = list()
        item_dict = test_set.test_dict[item]

        if not os.path.exists(os.path.join(save_dir, item)):
            os.mkdir(os.path.join(save_dir, item))
            os.mkdir(os.path.join(save_dir, item, 'ori'))
            os.mkdir(os.path.join(save_dir, item, 'gen'))
            os.mkdir(os.path.join(save_dir, item, 'mask'))
        for type in item_dict:
            if not os.path.exists(os.path.join(save_dir, item, 'ori', type)):
                os.mkdir(os.path.join(save_dir, item, 'ori', type))
            if not os.path.exists(os.path.join(save_dir, item, 'gen', type)):
                os.mkdir(os.path.join(save_dir, item, 'gen', type))
            if not os.path.exists(os.path.join(save_dir, item, 'mask', type)):
                os.mkdir(os.path.join(save_dir, item, 'mask', type))
            _time = list()
            img_list = item_dict[type]
            for path in img_list:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                ori_h, ori_w, _ = image.shape
                _t.tic()
                ori_img, input_tensor = transform(image)
                out = rebuilder.inference(input_tensor)
                re_img = out.transpose((1, 2, 0))
                s_map = ssim_seg(ori_img, re_img, win_size=11, gaussian_weights=True)
                s_map = cv2.resize(s_map, (ori_w, ori_h))
                if val_index == 1:
                    mask = seg_mask(s_map, threshold=threshold_seg_dict[item])
                elif val_index == 0:
                    mask = seg_mask(s_map, threshold=threshold_seg_dict)
                else:
                    raise Exception("Invalid val_index")

                inference_time = _t.toc()
                img_id = path.split('.')[0][-3:]
                cv2.imwrite(os.path.join(save_dir, item, 'ori', type, '{}.png'.format(img_id)), ori_img)
                cv2.imwrite(os.path.join(save_dir, item, 'gen', type, '{}.png'.format(img_id)), re_img)
                cv2.imwrite(os.path.join(save_dir, item, 'mask', type, '{}.png'.format(img_id)), mask)
                _time.append(inference_time)

                if type != 'good':
                    threshold_list.append(s_map)
                else:
                    pass

            cost_time += _time
            mean_time = np.array(_time).mean()
            print('Evaluate: Item:{}; Type:{}; Mean time:{:.1f}ms'.format(item, type, mean_time*1000))
            _t.clear()
        threshold_dict[item] = threshold_list
    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num*0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time*1000))

    # evaluate results
    print('Evaluating...')
    test_set.eval(save_dir,threshold_dict)


def test_chip(test_set, rebuilder, transform, save_dir):
    _t = Timer()
    cost_time = list()
    for type in test_set.test_dict:
        img_list = test_set.test_dict[type]
        if not os.path.exists(os.path.join(save_dir, type)):
            os.mkdir(os.path.join(save_dir, type))
        for k, path in enumerate(img_list):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _t.tic()
            ori_img, input_tensor = transform(image)
            out = rebuilder.inference(input_tensor)
            re_img = out[0]
            s_map = ssim_seg(ori_img, re_img, win_size=11, gaussian_weights=True)
            mask = seg_mask(s_map, threshold=32)
            inference_time = _t.toc()
            cat_img = np.concatenate((ori_img, re_img, mask), axis=1)
            cv2.imwrite(os.path.join(save_dir, type, '{:d}.png'.format(k)), cat_img)
            cost_time.append(inference_time)
            if (k+1) % 20 == 0:
                print('{}th image, cost time: {:.1f}'.format(k+1, inference_time*1000))
            _t.clear()
    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num*0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time*1000))


if __name__ == '__main__':
    args = parse_args()

    # load config file
    cfg_file = os.path.join('./config', args.cfg + '.json')
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)


    # load data set
    test_set = load_data_set_from_factory(configs, 'test')
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # retest
    if args.retest is True:
        print('Evaluating...')
        test_set.eval(args.res_dir)
        exit(0)

    # init and load Rebuilder
    # load model
    transform = Transform(tuple(configs['db']['resize']))
    net = load_test_model_from_factory(configs)
    rebuilder = Rebuilder(net, gpu_id=args.gpu_id)
    rebuilder.load_params(args.model_path)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    threshold_seg_dict = {}
    val_index = 0
    if configs['db']['name'] == 'mvtec':
        if configs['db']['use_validation_set'] is True:
            # load validation set
            val_index = 1
            val_set = load_data_set_from_factory(configs, 'validation')
            print('Data set: {} has been loaded'.format(configs['db']['name']))
            # validation for threshold selection
            print('Start Validation... ')
            threshold_seg_dict = val_mvtec(val_set, rebuilder, transform)
        elif configs['db']['use_validation_set'] is False:
            val_index = 0
        else:
            raise Exception("Invalid input")
    elif configs['db']['name'] == 'chip':
        pass
    else:
        raise Exception("Invalid set name")

    # test each image
    print('Start Testing... ')
    if configs['db']['name'] == 'mvtec':
        if configs['db']['use_validation_set'] is True:
            test_mvtec(test_set, rebuilder, transform, args.res_dir, threshold_seg_dict, val_index)
        elif configs['db']['use_validation_set'] is False:
            test_mvtec(test_set, rebuilder, transform, args.res_dir, 64, val_index)
        else:
            raise Exception("Invalid input")
    elif configs['db']['name'] == 'chip':
        test_chip(test_set, rebuilder, transform, args.res_dir)
    else:
        raise Exception("Invalid set name")