import nibabel as nib
import glob
import os
import numpy as np
import tensorlayer as tl

'''
Before normalization, run N4 bias correction (https://www.ncbi.nlm.nih.gov/pubmed/20378467), 
then save the data under folder ./CamCAN_unbiased/CamCAN
'''

modalities = ['T1w', 'T2w']
BraTS_modalities = ['T1w']
folders = ['HGG', 'LGG']
wd = './Data/CamCAN_unbiased/CamCAN'

thumbnail_idx = [60, 70, 80, 90]

for mod in modalities:
    wd_mod = os.path.join(wd, str(mod))
    os.chdir(wd_mod)
    img_files = [i for i in glob.glob("*") if "_unbiased" in i]

    for img in img_files:
        print(img)
        img_data = nib.load(img)
        img_data = img_data.get_data()
        mask = img.split("_unbiased")[0] + "_brain_mask.nii.gz"
        mask_data = nib.load(mask).get_data()

        img_data = np.transpose(img_data, [2, 0, 1])
        mask_data = np.transpose(mask_data, [2, 0, 1])

        idx = [s for s in range(img_data.shape[0]) if mask_data[s].sum() > 1]
        img_data = img_data[idx, :, 17:215]
        mask_data = mask_data[idx, :, 17:215]

        img_data = np.pad(img_data, ((0, 0), (1, 2), (1, 1)), mode='edge')
        mask_data = np.pad(mask_data, ((0, 0), (1, 2), (1, 1)), mode='edge')
        img_data = np.rot90(img_data, 1, (2, 1))
        mask_data = np.rot90(mask_data, 1, (2, 1))

        ref_mean = np.mean(img_data[mask_data == 1])
        ref_std = np.std(img_data[mask_data == 1])

        normed_img = (img_data - ref_mean) / ref_std
        normed_img[normed_img == normed_img.min()] = -3.5

        x_nif = nib.Nifti1Image(normed_img, np.eye(4))
        nib.save(x_nif, os.path.join(img.split("_unbiased")[0] + "_normalized_cropped_mask.nii.gz"))

        x_nif = nib.Nifti1Image(mask_data, np.eye(4))
        nib.save(x_nif, os.path.join(img.split("_unbiased")[0] + "_mask_cropped_mask.nii.gz"))

        tl.visualize.save_images(normed_img[thumbnail_idx, :, :, np.newaxis], [2, 2],
                                 "/scratch_net/bmicdl01/Data/CamCAN_unbiased/preview/" + str(mod)
                                 + "/" + img.split("_unbiased")[0] + "_normed_img.png")
        print("---")