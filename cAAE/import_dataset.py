import random
import numpy as np
import nibabel as nib
import os
import glob
from funcs import preproc

def create_datasets(retrain=False, task=None, labels=False, ds_scale=0):
    wd = "./Data/CamCAN_unbiased/CamCAN/T2w"
    #os.chdir(wd)
    subject_id = np.array([i for i in glob.glob(wd+"/*") if '.txt' not in i and 'normalized' in i])

    subject_train_idx = random.sample(range(0, 652), 400)
    subject_train = subject_id[subject_train_idx]
    subject_test = [i for i in subject_id if i not in subject_train][:50]
    if not retrain:
        np.savetxt(str(task)+"subject_train.txt", subject_train, "%s", delimiter=',')
        np.savetxt(str(task)+"subject_test.txt", subject_test, "%s", delimiter=',')

    else:
        subject_train = np.genfromtxt(str(task)+"subject_train.txt", dtype=str,delimiter=',')
        subject_test = np.genfromtxt(str(task)+"subject_test.txt", dtype=str,delimiter=',')

    print("retrain is {}".format(retrain))
    print(str(task)+"training subject ids: {}".format(subject_train))
    print(str(task)+"testing subject ids: {}".format(subject_test))


    X_train_input = []
    X_train_target = []
    X_train_target_all = []
    X_dev_input = []
    X_dev_target = []
    X_dev_target_all = []

    for i in subject_train:
        print(i)
        pathx=i
        img = nib.load(pathx).get_data()
        img = np.transpose(img, [2, 0, 1])
        idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        img = img[idx]
        _, x, y = img.shape
        max_xy = max(x,y)

        a = (max_xy-x)/2
        b = (max_xy-y)/2

        if len(X_train_input)==0:
            print(img.shape)

        img = np.pad(img, ((0, 0),((a,a)), (b,b)), mode = 'edge')
        img = preproc.resize(img, 256 /233.0)

        if labels:
            z = np.genfromtxt(os.path.join(wd, str(i) + '/' + str(i) + "_label.txt"))
            assert len(z)==len(idx)
            X_train_target.extend(z)

        if ds_scale!=0:
            img = preproc.downsample_image(img[np.newaxis,:,:,:],ds_scale)
        X_train_input.extend(img)


    for j in subject_test:
        print(j)
        pathx=j
        img = nib.load(pathx).get_data()
        img = np.transpose(img, [2, 0, 1])
        idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        img = img[idx]
        img = np.pad(img, ((0, 0), (a,a), (b,b)), mode='edge')
        img = preproc.resize(img, 256 / 233.0)

        if ds_scale!=0:
            img = preproc.downsample_image(img[np.newaxis,:,:,:],ds_scale)
        X_dev_input.extend(img)
        if labels:
            z = np.genfromtxt(os.path.join(wd, str(j) + '/' + str(j) + "_label.txt"))
            print(len(z), img.shape)
            X_dev_target.extend(z)

    if not labels:
        return np.asarray(X_train_input), np.asarray(X_dev_input)
    else:
        return X_train_input, X_train_target, X_dev_input, X_dev_target